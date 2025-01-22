import itertools
import pickle
import warnings

import ibis
import pyarrow as pa
import torch
from codetiming import Timer

import letsql as ls
import nasa_avionics_data_ml.zip_data as ZD
from nasa_avionics_data_ml.lib import (
    Config,
    read_scales,
)
from nasa_avionics_data_ml.letsql_udwf_inference import (
    make_inference_udwf,
)
from nasa_avionics_data_ml.letsql_udaf_training import (
    make_training_udaf,
    union_cached_asof_joined_flight_data,
    splat_struct,
    return_type,
)


tail = "Tail_652_1"
n_training_flights = 64
seq_length = 8
(order_by, group_by) = ("time", "flight")


(config, *_) = Config.get_debug_configs()
(scaleX, scaleT) = read_scales()
(training_udaf, model, *rest) = make_training_udaf(
    ibis.schema({name: float for name in (*config.x_names, *config.t_names)}),
    return_type,
    config,
    scaleX,
    scaleT,
)
tail_data = ZD.TailData.ensure_zip(tail + ".zip")
all_flight_datas = (*training_flight_datas, inference_flight_data) = tuple(itertools.islice(
    tail_data.gen_parquet_exists(),
    n_training_flights+1,
))
for flight_data in all_flight_datas:
    flight_data.ensure_rate_parquets()
training_data_expr = union_cached_asof_joined_flight_data(*training_flight_datas)
inference_data_expr = union_cached_asof_joined_flight_data(inference_flight_data)


model_expr = (
    training_data_expr
    .pipe(lambda t, col="col": (
        t
        .group_by(group_by)
        .agg(**{col: training_udaf.on_expr(t)})
        .pipe(splat_struct, col)
    ))
)
if not any(p for storage in training_data_expr.ls.storages for p in storage.path.iterdir()):
    with Timer("training_data_expr caching"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"row count is {ls.execute(training_data_expr.count())}")
with Timer("training_data_expr cached read"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"row count is {ls.execute(training_data_expr.count())}")
with Timer("model training"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        training_df = ls.execute(model_expr)
device = torch.device("cpu")
model = pickle.loads(training_df.sort_values("iter_idx").model.iloc[-1]).to(device)
model.device = device


inference_udwf = make_inference_udwf(
    ibis.schema({name: float for name in config.x_names}),
    pa.float64(),
    model, seq_length, scaleX, scaleT,
)
window = ibis.window(
    preceding=config.seq_length-1,
    following=0,
    order_by=order_by,
    group_by=group_by,
)
inference_expr = (
    inference_data_expr
    .mutate(predicted=inference_udwf.on_expr(inference_data_expr).over(window))
)
with Timer("model training"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inference_df = ls.execute(inference_expr)
