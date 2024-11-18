import pathlib


this_dir = pathlib.Path(__file__).parent.absolute()
model_path = this_dir.joinpath("cpu_model.torch")
parquet_cache_path = this_dir.joinpath("flight-cache")
