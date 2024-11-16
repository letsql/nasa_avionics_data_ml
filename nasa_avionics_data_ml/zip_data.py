import datetime
import functools
import multiprocessing
import operator
import pathlib
import warnings
import zipfile

import numpy as np
import pandas as pd
import toolz
from scipy.io import loadmat
from attr import (
    field,
    frozen,
)


default_data_dir = pathlib.Path("/mnt/nasa-data-download/data")


rate_to_rate_str = toolz.compose(operator.methodcaller("replace", ".", "p"), str)


rate_to_columns = {
    0.25: ('VAR_1107', 'VAR_2670', 'VAR_5107', 'VAR_6670', 'DATE_YEAR', 'DATE_MONTH', 'DATE_DAY', 'DVER_1', 'ACID', 'ESN_4', 'ECYC_1', 'ECYC_2', 'FRMC', 'DVER_2', 'ESN_3', 'ECYC_3', 'ECYC_4', 'EHRS_1', 'EHRS_4', 'EHRS_3', 'EHRS_2', 'ESN_1', 'ESN_2'),
    1.0: ('POVT', 'MW', 'DFGS', 'FQTY_1', 'OIT_1', 'OIT_2', 'APFD', 'PH', 'EVNT', 'MRK', 'VHF1', 'VHF2', 'LGDN', 'LGUP', 'VHF3', 'PUSH', 'HYDY', 'HYDG', 'SMOK', 'CALT', 'ACMT', 'FQTY_2', 'OIT_3', 'OIT_4', 'BLV', 'EAI', 'PACK', 'WSHR', 'WOW', 'TAT', 'SAT', 'FQTY_3', 'OIP_1', 'OIP_2', 'FQTY_4', 'CRSS', 'HDGS', 'ALTS', 'SNAP', 'CASS', 'N1CO', 'VSPS', 'MNS', 'VMODE', 'LMOD', 'A_T', 'OIP_3', 'OIP_4', 'LOC', 'GLS', 'LONP', 'ABRK', 'AIL_1', 'AIL_2', 'SPL_1', 'SPL_2', 'ELEV_1', 'ELEV_2', 'FLAP', 'PTRM', 'HF1', 'HF2', 'SMKB', 'SPLY', 'SPLG', 'BPGR_1', 'BPGR_2', 'BPYR_1', 'BPYR_2', 'TCAS', 'GPWS', 'TMAG', 'TAI', 'WAI_1', 'WAI_2', 'DWPT', 'OIPL', 'FADF', 'FADS', 'TMODE', 'ATEN', 'LATP', 'FIRE_1', 'FIRE_2', 'FIRE_3', 'FIRE_4', 'FGC3', 'ILSF'),
    2.0: ('PSA', 'PI', 'PS', 'PT', 'SHKR', 'MSQT_2', 'GMT_HOUR', 'GMT_MINUTE', 'GMT_SEC', 'APUF', 'TOCW', 'RUDD', 'MSQT_1', 'CCPC', 'CCPF', 'RUDP', 'CWPC', 'CWPF'),
    4.0: ('TH', 'MH', 'EGT_1', 'EGT_2', 'EGT_3', 'EGT_4', 'GS', 'TRK', 'TRKM', 'DA', 'WS', 'WD', 'ALT', 'NSQT', 'ALTR', 'AOA1', 'AOA2', 'FF_1', 'FF_2', 'FF_3', 'FF_4', 'N1_1', 'N1_2', 'MACH', 'CAS', 'CASM', 'TAS', 'LATG', 'N1_3', 'VIB_1', 'VIB_2', 'VIB_3', 'LONG', 'PLA_1', 'N1_4', 'VIB_4', 'PLA_2', 'PLA_3', 'PLA_4', 'AOAI', 'AOAC', 'BAL1', 'BAL2', 'N2_1', 'N2_2', 'N2_3', 'N2_4', 'N1T', 'N1C'),
    8.0: ('RALT', 'PTCH', 'ROLL', 'VRTG'),
    16.0: ('FPAC', 'BLAC', 'CTAC', 'IVV'),
}


def mat_2_df(mdata, param, samplesize):
    """This NASA matlab data is a dictionary of arrays of objects and this function
    takes in a single dictionary word and parses the data into a pandas dataframe
    """
    # create an array filled with NaNs so we don't fill in data with bad data yet.
    d = np.empty(samplesize).reshape(-1, 1)
    d[:] = np.nan
    rate = mdata[param]["Rate"][0, 0][0, 0]
    i = int(16 / rate)
    d[::i] = mdata[param]["data"][0, 0].reshape(-1, 1)

    return d


def process_d(d):
    hour = d["GMT_HOUR"]["data"][0, 0].reshape(-1, 1).astype(int)
    # skip the file if the plane never gets airborne for the duration of the file
    if min(hour) < 24:
        minute = d["GMT_MINUTE"]["data"][0, 0].reshape(-1, 1).astype(int)
        second = d["GMT_SEC"]["data"][0, 0].reshape(-1, 1).astype(int)
        # set time array as large as 16hz signal as NaNs then add actual time in GMT
        GMT = np.array(hour * 3600 + minute * 60 + second)
        # matches the fastest 16 Hz (adding an extra amount above 16Hz found in the data
        t_delta = 0.0625 + 0.5 / 549 / 16  # seconds.
        samplesize_16hz = d["FPAC"]["data"][0, 0].shape[0]
        # the first valid GMTsecs
        ti = np.argmin(GMT)
        t0 = GMT[ti]
        irange = ti + 13
        # skip the file if it doesn't have much airborne data after the first GMT
        if irange < GMT.shape[0]:
            # find the first time GMTsecs transitions to a new time, this is reliable GMT
            # it normally stays the same time for 6 seconds.
            i = min([i for i in range(ti, irange, 1) if not t0 == GMT[i]], default=None)
            if i is None:
                return ("Not enough airborne data", None)
            # GMTsecs is 2 Hz or 0.5 seconds
            timeoffset = 0.5 * i

            # offset from the first reliable GMT
            starttime = GMT[i, 0] - timeoffset

            stoptime = starttime + t_delta * (samplesize_16hz - 1)

            # create a 16 Hz time column to line everything up and make it a pandas dataframe
            t16hz = np.linspace(starttime, stoptime, samplesize_16hz)
            df1 = pd.DataFrame(data={"time": t16hz})

            # adding GMT in seconds
            t = np.empty(samplesize_16hz).reshape(-1, 1)
            t[:] = np.nan
            t[::8] = GMT  # every eighth element
            df1["GMTsecs"] = t

            # looping through all of the keys and creating a full Pandas DataFrame
            for k in d.keys():
                # don't print the first "__xxxx__" parameters
                if type(d[k]) == np.ndarray:
                    df1[k] = mat_2_df(d, k, samplesize_16hz)
            # removing the first set of unreliable GMT data so it starts cleanly
            df1 = df1[i * 8 : -1]
            df1.reset_index(inplace=True)
            df1.pop("index")
            return (None, df1)
        else:
            return ("Not enough airborne data", None)
    else:
        return ("No airborne data", None)


def process_mat_file(f):
    fdir = f.parent
    pdir = fdir.with_name(fdir.name + "_parquet")
    pdir.mkdir(exist_ok=True)
    pfile = pdir.joinpath(f.with_suffix(".parquet").name)
    if pfile.exists():
        print(f"skipping, file exists: {pfile}")
        return
    d = loadmat(f)
    (msg, df) = process_d(d)
    if msg is None and df is None:
        raise ValueError
    elif msg is not None and df is not None:
        raise ValueError
    elif msg:
        print(f"{msg}: {f.name}")
    elif df is not None:
        # writing to a parquet file in the parquet directory
        # overwriting file if it already exists
        df.to_parquet(path=pfile, compression="gzip")
        print(f"Wrote parquet: {pfile}")


def process_zipfile_element(zip_path, filename, pfile):
    zf = zipfile.ZipFile(zip_path)
    with zf.open(filename) as fh:
        d = loadmat(fh)
        (msg, df) = process_d(d)
    if msg is None and df is None:
        raise ValueError
    elif msg is not None and df is not None:
        raise ValueError
    elif msg:
        return (f"{msg}: {zip_path},{filename}", None)
    elif df is not None:
        # writing to a parquet file in the parquet directory
        # overwriting file if it already exists
        df.to_parquet(path=pfile, compression="gzip")
        return (None, pfile)


def process_zipfile(zip_path):
    zf = zipfile.ZipFile(zip_path)
    pdir = zip_path.with_name(zip_path.with_suffix("").name + "_parquet")
    pdir.mkdir(exist_ok=True)
    results = ()
    with multiprocessing.Pool() as pool:
        for file in zf.filelist:
            filename = file.filename
            pfile = pdir.joinpath(filename).with_suffix(".parquet")
            if not pfile.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = pool.apply_async(
                        process_zipfile_element,
                        (zip_path, filename, pfile),
                    )
                    results += (result,)
        results = tuple(result.get() for result in results)
    return results


@frozen
class TailData:
    zip_path = field()

    @property
    def tail(self):
        return self.zip_path.with_suffix("").name

    def __attrs_post_init__(self):
        assert self.zip_path.exists()
        # verify that we have a proper zip file
        zipfile.ZipFile(self.zip_path)

    @property
    @functools.cache
    def filenames(self):
        zf = zipfile.ZipFile(self.zip_path)
        return tuple(file.filename for file in zf.filelist)

    @property
    @functools.cache
    def flight_datas(self):
        return tuple(FlightData(self.zip_path, filename) for filename in self.filenames)

    def ensure_parquets(self):
        results = ()
        with multiprocessing.Pool() as pool:
            for flight_data in self.flight_datas:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = pool.apply_async(flight_data.ensure_parquet)
                    results += (result,)
            results = tuple(result.get() for result in results)
        return results

    def exists(self):
        return all(flight_data.exists() for flight_data in self.flight_datas)

    def gen_parquet_exists(self):
        yield from (flight_data for flight_data in self.flight_datas if flight_data.parquet_path.exists())

    @classmethod
    def gen_from_data_dir(cls, data_dir=default_data_dir):
        yield from (cls(p) for p in sorted(data_dir.iterdir()) if p.suffix == ".zip")

    @classmethod
    def ensure_all_parquets(cls, data_dir=default_data_dir):
        dct = {}
        tail_datas = cls.gen_from_data_dir(data_dir=data_dir)
        for (i, tail_data) in enumerate(tail_datas):
            dct[tail_data] = tail_data.ensure_parquets()
            print(f"{datetime.datetime.now()} :: done :: {i} / {len(tail_datas)} :: {tail_data.zip_path}")
        return dct


@frozen
class FlightData:
    zip_path = field()
    filename = field()

    def __attrs_post_init__(self):
        assert self.filename in TailData(self.zip_path).filenames

    def get_d(self):
        zf = zipfile.ZipFile(self.zip_path)
        with zf.open(self.filename) as fh:
            d = loadmat(fh)
        return d

    def get_df(self):
        d = self.get_d()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (msg, df) = process_d(d)
        if msg is None and df is None:
            raise ValueError
        elif msg is not None and df is not None:
            raise ValueError
        elif msg:
            return (f"{msg}: {self.zip_path},{self.filename}", None)
        return (msg, df)

    @staticmethod
    def split_df_by_rate(df):
        rate_to_df = {
            rate: df.set_index("time")[list(columns)].dropna().reset_index()
            for rate, columns in rate_to_columns.items()
        }
        return rate_to_df

    @property
    def flight(self):
        n = -4
        (head, tail) = (self.filename[:n], self.filename[n:])
        if tail != ".mat":
            raise ValueError
        return head

    def get_rate_to_columns(self):
        d = self.get_d()
        rates = pd.Series({
            k: float(v["Rate"][0][0][0][0])
            for k, v in d.items()
            if hasattr(v, "dtype")
        })
        rate_to_columns = (
            rates
            .groupby(rates)
            .apply(lambda s: tuple(s.index))
            .to_dict()
        )
        return rate_to_columns

    @property
    def parquet_dir(self):
        return self.zip_path.with_name(self.zip_path.with_suffix("").name + "_parquet")

    @property
    def _base_path(self):
        return self.parquet_dir.joinpath(self.filename)

    @property
    def parquet_path(self):
        return self._base_path.with_suffix(".parquet")

    @property
    def err_path(self):
        return self._base_path.with_suffix(".err")

    def ensure_parquet(self):
        parquet_path = self.parquet_path
        msg = None
        if not parquet_path.exists():
            if not self.err_path.exists():
                # neither exist: try it
                parquet_path.parent.mkdir(exist_ok=True)
                (msg, df) = self.get_df()
                if df is not None:
                    df.to_parquet(path=parquet_path, compression="gzip")
                elif msg is not None:
                    self.err_path.write_text(msg)
            else:
                # err_path contains the error message
                msg = self.err_path.read_text()
        return (msg, parquet_path)

    def ensure_rate_parquets(self):
        rate_to_parquet = {
            rate: self.parquet_path.with_suffix(f".{rate_to_rate_str(rate)}.parquet")
            for rate in rate_to_columns
        }
        msg = None
        if not all(p.exists() for p in rate_to_parquet.values()):
            (msg, parquet_path) = self.ensure_parquet()
            if msg is None:
                rate_to_df = self.split_df_by_rate(pd.read_parquet(parquet_path))
                for rate, df in rate_to_df.items():
                    rate_parquet = rate_to_parquet[rate]
                    df.to_parquet(rate_parquet, compression="gzip")
        return (msg, rate_to_parquet)

    def exists(self):
        return self.parquet_path.exists() or self.err_path.exists()

    @classmethod
    def from_parquet_path(cls, parquet_path):
        suffix = ".parquet"
        if parquet_path.suffix != suffix:
            raise ValueError
        filename = parquet_path.with_suffix(".mat").name
        zip_path = parquet_path.parent.with_name(parquet_path.parent.name[:-len("_parquet")]).with_suffix(".zip")
        return cls(zip_path, filename)


def clear_empty_zips(data_dir, dry_run=True):
    unlinked = ()
    for path in data_dir.iterdir():
        if path.suffix == ".zip" and not path.stat().st_size:
            unlinked += (path,)
    if not dry_run:
        for path in unlinked:
            path.unlink()
    return unlinked


def clear_bad_zips(data_dir, dry_run=True):
    unlinked = ()
    for path in data_dir.iterdir():
        if path.suffix == ".zip":
            try:
                zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                unlinked += (path,)
    if not dry_run:
        for path in unlinked:
            path.unlink()
    return unlinked


def get_first_existing():
    tail_data = next(TailData.gen_from_data_dir())
    flight_data = next(flight_data for flight_data in tail_data.flight_datas if flight_data.parquet_path.exists())
    (msg, rate_to_parquet) = flight_data.ensure_rate_parquets()
    if msg is not None:
        raise ValueError
    return rate_to_parquet
