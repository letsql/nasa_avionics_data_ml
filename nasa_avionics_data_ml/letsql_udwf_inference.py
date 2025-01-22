import operator
import pprint
import warnings

import ibis
import pyarrow as pa

import letsql as ls
from nasa_avionics_data_ml.lib import (
    pyarrow_to_sequence_tensor,
)
import nasa_avionics_data_ml.settings as S
import nasa_avionics_data_ml.zip_data as ZD
from letsql.common.caching import ParquetSnapshot
from letsql.common.utils.defer_utils import deferred_read_parquet
from letsql.expr.relations import into_backend
from letsql.expr.udf import (
    pyarrow_udwf,
)


def make_rate_to_parquet(flight_data):
    return {
        rate: (
            "https://nasa-avionics-data-ml.s3.us-east-2.amazonaws.com"
            f"/{flight_data.parquet_dir.name}/{flight_data.flight}.{ZD.rate_to_rate_str(rate)}.parquet"
        )
        for rate in ZD.rate_to_columns
    }


def asof_join_flight_data(flight_data, airborne_only=True):
    """Create an expression for a particular flight's data """

    rate_to_parquet = make_rate_to_parquet(flight_data)

    con = ls.connect()
    # HAK: ensure object_storage is loaded
    con.read_csv(next(iter(rate_to_parquet.values())))
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


def make_inference_udwf(schema, return_type, model, seq_length, scaleX, scaleT):
    """Create a udwf for running inference of a particular model """

    @pyarrow_udwf(
        schema=schema,
        return_type=ibis.dtype(return_type),
    )
    def inference_udwf(self, values, num_rows):
        return predict_flight(
            values,
            model,
            seq_length,
            scaleX,
            scaleT,
            return_type,
        )
    return inference_udwf


def predict_flight(arrow_values, model, seq_length, scaleX, scaleT, return_type):

    def torch_to_payrrow(tensor, scaler):
        predicted = scaler.inverse_transform(tensor.detach().numpy())[:, 0]
        arrow = pa.Array.from_pandas(predicted)
        return arrow

    def prepad_pyarrow(n, arrow):
        return pa.concat_arrays([
            pa.array([float("nan")]*n).cast(arrow.type),
            arrow,
        ])

    tensor = pyarrow_to_sequence_tensor(arrow_values, seq_length, scaleX)
    predicted = model(tensor)
    arrow = torch_to_payrrow(predicted, scaleT)
    prepended = prepad_pyarrow(seq_length-1, arrow).cast(return_type)
    return prepended


def do_manual_batch(expr, model, seq_length, scaleX, scaleT, return_type, xlist, group_by, order_by):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = ls.execute(expr.order_by(order_by))
    predicted = (
        df
        .groupby(group_by)
        [df.columns]
        .apply(
            lambda t: (
                t
                .assign(
                    predicted=predict_flight(
                        # HAK: index is received as an extra column
                        pa.Table.from_pandas(t[xlist]).columns[:len(xlist)],
                        model,
                        seq_length,
                        scaleX,
                        scaleT,
                        return_type,
                    ),
                )
            ),
        )
        .reset_index(drop=True)
    )
    return predicted


if __name__ == "__main__":
    import itertools
    import pprint

    from nasa_avionics_data_ml.lib import (
        Config,
        read_model_and_scales,
    )

    (order_by, group_by) = ("time", "flight")
    tail = "Tail_652_1"

    return_type = "float64"
    (config, *_) = Config.get_debug_configs()
    (model, scaleX, scaleT) = read_model_and_scales()
    # (seq_length, xlist) = (config.seq_length, config.xlist)
    inference_udwf = make_inference_udwf(
        ibis.schema({name: float for name in config.x_names}),
        return_type, model, 8, scaleX, scaleT,
    )

    # get 8 flights from tail 652_1
    (flight_data, *_) = flight_datas = tuple(itertools.islice(
        next(td for td in ZD.TailData.gen_from_data_dir() if td.tail == tail).gen_parquet_exists(),
        8,
    ))
    single_expr = asof_join_flight_data(flight_data)
    expr = union_cached_asof_joined_flight_data(*flight_datas)
    window = ibis.window(
        preceding=config.seq_length-1,
        following=0,
        order_by=order_by,
        group_by=group_by,
    )
    with_prediction = (
        expr
        .mutate(predicted=inference_udwf.on_expr(expr).over(window))
    )

    pprint.pprint(make_rate_to_parquet(flight_data))
    print(ls.to_sql(single_expr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_letsql = ls.execute(with_prediction.order_by(group_by, order_by))
    from_manual = (
        do_manual_batch(expr, model, config.seq_length, scaleX, scaleT, return_type, config.xlist, group_by, order_by)
        .sort_values([group_by, order_by], ignore_index=True)
    )

    assert from_manual.equals(from_letsql)
