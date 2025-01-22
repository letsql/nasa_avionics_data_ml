import operator
import pickle
import time
import warnings
from threading import Lock

import ibis
import numpy as np
import pyarrow as pa
import toolz
import torch
from codetiming import Timer

import letsql as ls
from nasa_avionics_data_ml.lib import (
    pandas_to_sequence_tensor,
)
import nasa_avionics_data_ml.settings as S
import nasa_avionics_data_ml.zip_data as ZD
from letsql.common.caching import ParquetSnapshot
from letsql.common.utils.defer_utils import deferred_read_parquet
from letsql.expr.relations import into_backend
from letsql.expr.udf import (
    agg,
)


gpu_lock = Lock()


def make_rate_to_url(flight_data):
    return {
        rate: (
            "https://nasa-avionics-data-ml.s3.us-east-2.amazonaws.com"
            f"/{flight_data.parquet_dir.name}/{flight_data.flight}.{ZD.rate_to_rate_str(rate)}.parquet"
        )
        for rate in ZD.rate_to_columns
    }


def make_rate_to_parquet(flight_data):
    return {
        rate: (
            flight_data.parquet_dir.joinpath(f"{flight_data.flight}.{ZD.rate_to_rate_str(rate)}.parquet")
        )
        for rate in ZD.rate_to_columns
    }


def asof_join_flight_data(flight_data, airborne_only=True):
    """Create an expression for a particular flight's data """

    rate_to_parquet = make_rate_to_parquet(flight_data)

    con = ls.connect()
    ts = (
        deferred_read_parquet(con, parquet_path, ZD.rate_to_rate_str(rate))
        .mutate(flight=ls.literal(flight_data.flight))
        for rate, parquet_path in sorted(rate_to_parquet.items(), key=operator.itemgetter(0), reverse=True)
    )
    db_con = ls.duckdb.connect()
    (expr, *others) = (into_backend(t, db_con, name=f"flight-{flight_data.flight}-{t.op().parent.name}") for t in ts)
    for other in others:
        expr = expr.asof_join(other, on="time").drop(["time_right", "flight_right"])
    if airborne_only:
        expr = expr[lambda t: t.GS != 0]
    return expr


def union_cached_asof_joined_flight_data(*flight_datas, cls=ParquetSnapshot):
    return ls.union(*(
        asof_join_flight_data(flight_data).cache(storage=cls(path=S.parquet_cache_path))
        for flight_data in flight_datas
    ))


iter_idx = -1
pa_return_type = pa.scalar({
    "iter_idx": iter_idx,
    "batch_loss": float(),
    "model": pickle.dumps(None),
    "start_time": time.perf_counter(),
    "stop_time": time.perf_counter(),
}).type


@toolz.curry
def train_batch(df, config, scaleX, scaleT, model, loss_func, opt, error_trace, astype, return_type):
    global iter_idx

    start_time = time.perf_counter()
    exog_tensor = pandas_to_sequence_tensor(
        df[scaleX.feature_names_in_],
        config.seq_length,
        scaleX,
        # FIXME: enable per-column astype
        astype,
    ).to(config.device)
    endog_tensor = pandas_to_sequence_tensor(
        df[scaleT.feature_names_in_].iloc[config.seq_length-1:],
        None,
        scaleT,
        astype,
    ).to(config.device)
    with gpu_lock:
        train_output = model(exog_tensor)
        opt.zero_grad()
        loss = loss_func(train_output, endog_tensor)
        loss.backward()
        opt.step()
        batch_loss = loss.detach().cpu()
        torch.cuda.empty_cache()

    error_trace.append(batch_loss)
    stop_time = time.perf_counter()
    # return a struct that contains invocation#, batch_loss, model-as-binary
    iter_idx += 1
    dct = {
        "iter_idx": iter_idx,
        "batch_loss": float(batch_loss),
        "model": pickle.dumps(model),
        "start_time": start_time,
        "stop_time": stop_time,
    }
    return pa.scalar(dct, type=return_type.to_pyarrow())


def make_training_udaf(schema, return_type, config, scaleX, scaleT):
    """Create a udaf for running training of a particular model """

    if set(schema) != set(scaleX.feature_names_in_).union(scaleT.feature_names_in_):
        raise ValueError

    model = config.get_model()
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.l_rate)
    error_trace = []
    astype = np.float32

    training_udaf = agg.pandas_df(
        schema=schema,
        fn=train_batch(
            config=config,
            scaleX=scaleX,
            scaleT=scaleT,
            model=model,
            loss_func=loss_func,
            opt=opt,
            error_trace=error_trace,
            astype=astype,
            return_type=return_type,
        ),
        return_type=return_type,
        name="train_batch",
    )

    # we return all the things the user needs a reference to
    return training_udaf, model, loss_func, opt, error_trace


if __name__ == "__main__":
    import itertools

    import pandas as pd

    from nasa_avionics_data_ml.lib import (
        Config,
        read_scales,
    )

    (order_by, group_by) = ("time", "flight")
    tail = "Tail_652_1"
    n_flights = 8

    return_type = ibis.dtype(pa_return_type)
    (config, *_) = Config.get_debug_configs()
    (scaleX, scaleT) = read_scales()
    training_udaf, *rest = make_training_udaf(
        ibis.schema({name: float for names in (config.x_names, config.t_names) for name in names}),
        return_type,
        config,
        scaleX,
        scaleT,
    )
    (model, loss_func, opt, error_trace) = rest

    tail_data = next(td for td in ZD.TailData.gen_from_data_dir() if td.tail == tail)
    (flight_data, *_) = flight_datas = tuple(itertools.islice(
        tail_data.gen_parquet_exists(),
        n_flights,
    ))
    expr = union_cached_asof_joined_flight_data(*flight_datas)
    with_batch_loss = expr.group_by("flight").agg(batch_loss=training_udaf.on_expr(expr))

    with Timer("first expr execution"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"row count is {ls.execute(expr.count())}")
    with Timer("second expr execution"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"row count is {ls.execute(expr.count())}")
    with Timer("first full run"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch_loss_df = (
                ls.execute(with_batch_loss)
                .pipe(lambda t: t.drop(columns="batch_loss").join(t.batch_loss.apply(pd.Series)))
            )
