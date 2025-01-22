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
return_type = ibis.schema({
    "iter_idx": type(iter_idx),
    "batch_loss": float,
    "model": bytes,
    "start_time": float,
    "stop_time": float,
}).as_struct()


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


def splat_struct(expr, col, do_drop=True):
    expr = expr.mutate(**{field: expr[col][field] for field in expr[col].fields})
    if do_drop:
        expr = expr.drop(col)
    return expr


if __name__ == "__main__":
    import itertools

    from nasa_avionics_data_ml.lib import (
        Config,
        read_scales,
    )


    tail = "Tail_652_1"
    n_flights = 64

    (config, *_) = Config.get_debug_configs()
    (scaleX, scaleT) = read_scales()
    (training_udaf, model, *rest) = make_training_udaf(
        ibis.schema({name: float for name in (*config.x_names, *config.t_names)}),
        return_type,
        config,
        scaleX,
        scaleT,
    )

    tail_data = next(td for td in ZD.TailData.gen_from_data_dir() if td.tail == tail)
    flight_datas = tuple(itertools.islice(
        tail_data.gen_parquet_exists(),
        n_flights,
    ))
    data_expr = union_cached_asof_joined_flight_data(*flight_datas)
    model_expr = (
        data_expr
        .pipe(lambda t, col="col": (
            t
            .group_by("flight")
            .agg(**{col: training_udaf.on_expr(t)})
            .pipe(splat_struct, col)
        ))
    )

    if not any(p for storage in data_expr.ls.storages for p in storage.path.iterdir()):
        with Timer("data_expr caching"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(f"row count is {ls.execute(data_expr.count())}")
    with Timer("data_expr cached read"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"row count is {ls.execute(data_expr.count())}")
    with Timer("model training"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = ls.execute(model_expr)
