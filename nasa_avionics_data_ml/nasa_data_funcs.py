import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def remove_outliers(df):
    for name, cont in (
        ("VRTG", float(0.03)),
        ("LATG", float(0.015)),
        ("LONG", float(0.015)),
    ):
        # using the Isolation Forest model to remove outliers in some of the data
        # contamination values that have been manually chosen
        # Isolation Forest model
        model = IsolationForest(
            n_estimators=100, max_samples=256, contamination=cont, max_features=1.0
        )
        outliers = model.fit_predict(df[[name]].values)
        # filtering out the outliers
        df.loc[outliers < 0, name] = np.NaN
    # pad all NaNs created by the outlier filter and backfill all NaN's at the beginning of the datafile
    df = df.ffill().bfill()
    return df


def strip_n_fill(df, xlist, tlist, outliers=False, add_elev=False, airborne_only=False):
    """This function only keeps the variables that appear to be needed.
    1. It strips extra data out of the file
    2. fills in NaNs
    3. if filter is set to True then it performs an Isolation Forest filter
    4. if add_elev is true then call the function to add elevation for each lat/lon
    5. if VRTG = False then this will exclude VRTG, LATG, and LONG in the input.
    """

    # remove data when it's on the ground before padding any other data.
    # this is done since the NSQT switch has a periodic 0 signal when airborne and I don't want to pad
    # this noise everywhere at 16hz then strip it.  This should also happen before the elevation data is
    # added.
    df = df[xlist + tlist].ffill().bfill()
    if airborne_only:
        # this replaces strip_ground_data
        df = df[lambda t: t["GS"] != 0]

    # if you would like to keep the VRTG, LATG, and LONG then filter them with Isolation Forest method
    (X, T) = (df[xlist], df[tlist])
    if outliers:
        X = remove_outliers(X)

    X = X.reset_index(drop=True)
    T = T.reset_index(drop=True)
    if add_elev:
        raise ValueError
    return X, T


def scale_data(X, T, scaleX=None, scaleT=None):
    """This function turns the dataframe into an np.array and scales it between values of 0 & 1.
    It returns the new arrays and the scale objects so it can un-scale data when needed.
    This also returns an un-scaled time to be used for plots
    1. converts df to np.array
    2. scales the values between 0 & 1
    3. If a scaleX and scaleY are given as inputs then this will use those to scale the data
       turning the dataframes into numpy arrays"""

    time = X["time"]

    # check to see if scaleX was defined or not
    if not scaleX:
        # Normalizing/Scaling the Training data
        scaleX = MinMaxScaler()
        scaleT = MinMaxScaler()

        # scale the whole dataset based on it's min/max
        Xs = scaleX.fit_transform(X)
        Ts = scaleT.fit_transform(T)
    else:
        Xs = scaleX.transform(X)
        Ts = scaleT.transform(T)

    Xs = pd.DataFrame(Xs, index=X.index, columns=X.columns)
    Ts = pd.DataFrame(Ts, index=T.index, columns=T.columns)
    return Xs, Ts, scaleX, scaleT, time


def read_filtered(
    paths,
    xlist,
    tlist,
    outliers=False,
    add_elev=False,
    airborne_only=False,
):
    gen = (
        strip_n_fill(
            pd.read_parquet(path=path),
            xlist,
            tlist,
            outliers=outliers,
            add_elev=add_elev,
            airborne_only=airborne_only,
        )
        for path in paths
    )
    (X, T) = (
        pd.concat(
            el,
            ignore_index=True,
        )
        for el in zip(*gen)
    )
    return (X, T)


def read_filtered_scaled(
    paths,
    xlist,
    tlist,
    scaleX=None,
    scaleT=None,
    outliers=False,
    add_elev=False,
    airborne_only=False,
):
    (X, T) = read_filtered(paths, xlist, tlist, outliers=outliers, add_elev=add_elev, airborne_only=airborne_only)
    (Xs, Ts, scaleX, scaleT, time) = scale_data(X, T, scaleX=scaleX, scaleT=scaleT)
    return (Xs, Ts, scaleX, scaleT, time)


def read_filtered_scaled_windowed(
    paths,
    seq_length,
    xlist,
    tlist,
    scaleX=None,
    scaleT=None,
    outliers=False,
    add_elev=False,
    airborne_only=False,
):
    (Xs, Ts, scaleX, scaleT, time) = read_filtered_scaled(paths, xlist, tlist, scaleX=scaleX, scaleT=scaleT, outliers=outliers, add_elev=add_elev, airborne_only=airborne_only)
    Xwin, Twin, Timetrain = sliding_window(Xs, Ts, time, seq_length)
    return Xwin, Twin, Timetrain, scaleX, scaleT


def get_sliding_window(X, T, time, seq_length, i):
    (start, stop) = (i, i+seq_length)
    if len(X) <= stop:
        raise ValueError
    # if time is None:
    #     return (
    #         X[start:stop, :],
    #         T[stop, :],
    #         None,
    #     )
    # else:
    #     return (
    #         X[start:stop, :],
    #         T[stop, :],
    #         time[stop],
    #     )
    if time is None:
        return (
            X.values[start:stop, :],
            T.values[stop, :],
            None,
        )
    else:
        return (
            X.values[start:stop, :],
            T.values[stop, :],
            time.values[stop],
        )


def partition(n, seq):
    gen = iter(seq)
    while True:
        value = tuple(itertools.islice(gen, n))
        if value:
            yield value
        else:
            break


def gen_sliding_windows(X, T, time, seq_length):
    # FIXME: partition by flight: its all the features and they're all floats
    for i in range(len(X)-seq_length):
        yield get_sliding_window(X, T, time, seq_length, i)


def gen_batches(X, T, seq_length, batch_size):
    gen = gen_sliding_windows(X, T, None, seq_length)
    for batch in partition(batch_size, gen):
        x, t, _ = map(np.array, zip(*batch))
        yield x, t
