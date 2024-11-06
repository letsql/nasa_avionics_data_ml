import copy
import functools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from attrs import (
    field,
    frozen,
)
from attrs.validators import (
    deep_iterable,
    instance_of,
)

import nasa_avionics_data_ml.nasa_data_funcs as ndf
from nasa_avionics_data_ml.LSTM import LSTM


default_pdir = pathlib.Path("/mnt/nasa-data-download/data/Tail_666_1_parquet/")
default_t_names = ("ALT",)
base_x_names = (
    "time",
    "RALT",
    "PSA",
    "PI",
    "PT",
    "ALTR",
    "IVV",
    "VSPS",
    "FPAC",
    "BLAC",
    "CTAC",
    "TAS",
    "CAS",
    "GS",
    "CASS",
    "WS",
    "PTCH",
    "ROLL",
    "DA",
    "TAT",
    "SAT",
    "LATP",
    "LONP",
)


@frozen
class Config:
    n_rand_loops = field(validator=instance_of(int), default=3)
    n_files = field(validator=instance_of(int), default=3)
    batch_size = field(validator=instance_of(int), default=50_000)
    seq_length = field(validator=instance_of(int), default=8)
    l_rate = field(validator=instance_of(float), default=0.01)
    epoch = field(validator=instance_of(int), default=1000)
    n_h_unit = field(validator=instance_of(int), default=10)
    n_h_layer = field(validator=instance_of(int), default=1)
    pdir = field(validator=instance_of(pathlib.Path), converter=pathlib.Path, default=default_pdir)
    t_names = field(validator=deep_iterable(instance_of(str), instance_of(tuple)), default=default_t_names)
    x_names = field(validator=deep_iterable(instance_of(str), instance_of(tuple)), init=False)
    airborne_only = field(validator=instance_of(bool), default=True)
    vrtg = field(validator=instance_of(bool), default=True)
    train_frac = field(validator=instance_of(float), default=2/3)

    def __attrs_post_init__(self):
        if not self.pdir.exists():
            raise ValueError
        x_names = base_x_names
        if self.airborne_only or self.vrtg:
            x_names += ("LATG", "LONG", "VRTG")
        object.__setattr__(self, "x_names", x_names)

    @property
    def xlist(self):
        return list(self.x_names)

    @property
    def tlist(self):
        return list(self.t_names)

    @classmethod
    def get_debug_configs(cls):
        fixed = dict(
            n_rand_loops=1,
            n_files=3,
        )
        varying = dict(
            seq_length=(8,8),
            l_rate=(.01, .05),
            epoch=(150, 150),
            n_h_unit=(10, 10),
            n_h_layer=(1, 1),
        )
        yield from (
            cls(
                **dict(zip(varying.keys(), values)),
                **fixed,
            )
            for values in zip(*varying.values())
        )

    @classmethod
    def get_default_configs(cls):
        varying = dict(
            seq_length=(8, 8, 16, 16, 16, 16, 16, 32),
            l_rate=(0.01, 0.05, 0.01, 0.01, 0.03, 0.05, 0.05, 0.05),
            epoch=(1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
            n_h_unit=(10, 20, 10, 15, 20, 20, 25, 20),
            n_h_layer=(1, 1, 1, 2, 1, 1, 1, 1),
        )
        yield from (
            cls(**dict(zip(varying.keys(), values)))
            for values in zip(*varying.values())
        )

    @property
    @functools.cache
    def paths(self):
        return tuple(p for p in self.pdir.iterdir() if p.suffix == ".parquet")[:self.n_files]

    def get_train_paths(self, seed):
        # shuffle the files around in this list to get different results for the bootstrapping
        paths = shuffle(list(self.paths), seed)
        n_train = int(len(paths) * self.train_frac)
        train_paths = paths[:n_train]
        return train_paths

    def get_train_data(self, seed, **read_kwargs):
        train_paths = self.get_train_paths(seed=seed)
        X, T, scaleX, scaleT, time = ndf.read_filtered_scaled(
            train_paths,
            self.xlist,
            self.tlist,
            airborne_only=self.airborne_only,
            **read_kwargs,
        )
        return X, T, scaleX, scaleT, time

    def get_test_paths(self, seed):
        # shuffle the files around in this list to get different results for the bootstrapping
        paths = shuffle(list(self.paths), seed)
        n_train = int(len(paths) * self.train_frac)
        test_paths = paths[n_train:]
        return test_paths

    def get_test_data(self, seed, scaleX, scaleT, **read_kwargs):
        test_paths = self.get_test_paths(seed=seed)
        X, T, _, _, time = ndf.read_filtered_scaled(
            test_paths,
            self.xlist,
            self.tlist,
            scaleX=scaleX,
            scaleT=scaleT,
            airborne_only=self.airborne_only,
            **read_kwargs
        )
        return X, T, time

    def get_model(self):
        model = LSTM(
            len(self.x_names),
            len(self.t_names),
            self.n_h_unit,
            self.n_h_layer,
            self.device,
        )
        model = model.to(self.device)
        return model

    def train_model(self, seed, scaleX=None, scaleT=None, **read_kwargs):
        X, T, scaleX, scaleT, _ = self.get_train_data(seed=seed, **read_kwargs)
        model = self.get_model()
        loss_func = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.l_rate)
        error_trace = []
        for _ in range(self.epoch):
            for _, (x, t) in enumerate(
                ndf.gen_batches(X, T, self.seq_length, batch_size=self.batch_size)
            ):
                # create tensors from the data frames
                Xtrain = torch.from_numpy(x.astype(np.float32)).to(self.device)
                Ttrain = torch.from_numpy(t.astype(np.float32)).to(self.device)

                ##run input forward through model
                train_output = model(Xtrain)

                # zero out gradients of optimized torch.Tensors to zero before back-propagation happens
                opt.zero_grad()

                # calculate the loss of the model output (Y) to the training Target (Ttrain)
                loss = loss_func(train_output, Ttrain)

                # back-propagation of the loss
                loss.backward()

                # performs a single optimization step (parameter update)
                opt.step()

                # keeping track of the history of the error in order to plot the convergence.
                error_trace.append(loss.detach().cpu())

        # do the last batch one last time
        loss = loss_func(train_output, Ttrain).detach().cpu()
        error_train = loss.item()
        torch.cuda.empty_cache()
        return (
            model,
            scaleX,
            scaleT,
            loss_func,
            opt,
            error_trace,
            error_train,
        )

    def test_model(self, seed, scaleX, scaleT, model, loss_func, **read_kwargs):
        X, T, _ = self.get_test_data(seed=seed, scaleX=scaleX, scaleT=scaleT, **read_kwargs)
        x, t, _ = map(np.array, zip(*ndf.gen_sliding_windows(X, T, None, self.seq_length)))

        # create tensors from the data frames
        Xtest = torch.from_numpy(x.astype(np.float32)).to(self.device)
        Ttest = torch.from_numpy(t.astype(np.float32)).to(self.device)

        # Run the test data through the trained model
        test_output = model(Xtest)
        # calculate the loss of the model output (Y) to the training Target (Ttrain)
        loss = loss_func(test_output, Ttest).detach().cpu()

        error_test = loss.item()
        torch.cuda.empty_cache()
        return error_test

    @property
    @functools.cache
    def device(self):
        if torch.cuda.is_available():
            (device, *_) = (torch.device(i) for i in range(torch.cuda.device_count()))
        else:
            device = torch.device("cpu")
            raise("we can't set_device to cpu")
        return device

    @property
    def to_kwargs(self):
        return {
            "device": self.device,
            # "dtype": torch.float32,
            # "non_blocking": False,
        }


def set_device(config):
    torch.cuda.set_device(config.device)


def print_cuda_settings():
    print(f"Total Number of CUDA Devices: {torch.cuda.device_count()}")
    print(
        f"Current CUDA Device: GPU{torch.cuda.current_device()} --> {torch.cuda.get_device_name()}"
    )
    print(f"Device Properties:\n  {torch.cuda.get_device_properties(torch.cuda.current_device())}")


def shuffle(x, seed=0):
    from numpy.random import Generator, PCG64
    rng = Generator(PCG64(seed))
    x = copy.copy(x)
    rng.shuffle(x)
    return x


def get_paths(pdir, n, seed=0):
    paths = tuple(p for p in pdir.iterdir() if p.suffix == ".parquet")
    paths = paths[:n]
    paths = shuffle(paths, seed)
    return paths


(config, *_) = configs = tuple(Config.get_debug_configs())
set_device(config)
print_cuda_settings()


print('All configurations trained and tested on "n_files".')
print(
    "These files were randomly selected and this process was repeated for each configuration"
)
print(
    f"{config.n_rand_loops} times with the MSE being averaged over this repeated set training models."
)


Xtrain, Ttrain, scaleX, scaleT, timetrain = config.get_train_data(seed=0)
Xtest, Ttest, timetest = config.get_test_data(seed=0, scaleX=scaleX, scaleT=scaleT)


avg_error_train_list = []
avg_error_test_list = []
# loop through all the different configurations
for (config_i, config) in enumerate(configs):
    print(f"\n\n\n----------------------\nConfiguration #{config_i}\n----------------------")
    print(f"seq_length:{config.seq_length}")
    print(f"learning_rate:{config.l_rate}")
    print(f"n_epochs:{config.epoch}")
    print(f"hidden_units:{config.n_h_unit}")
    print(f"n_hidden_layers:{config.n_h_layer}")
    print(f"n_files:{config.n_files}\n-----------------")

    sum_error_train = 0
    sum_error_test = 0
    # bootstrap loop
    for rf in range(config.n_rand_loops):
        # print('Loading Data...')
        (model, scaleX, scaleT, loss_func, opt, error_trace, error_train) = config.train_model(seed=rf)
        sum_error_train += error_train

        # testing the model
        # --------------------------
        # print('Testing the Model...')
        error_test = config.test_model(
            seed=rf, scaleX=scaleX, scaleT=scaleT, model=model, loss_func=loss_func,
        )
        sum_error_test += error_test
        # print('after test delete')
        #!nvidia-smi

    # calculating error over all the random runs
    avg_error_train = sum_error_train / config.n_rand_loops
    avg_error_train_list.append(avg_error_train)
    avg_error_test = sum_error_test / config.n_rand_loops
    avg_error_test_list.append(avg_error_test)

    # run full set of the last training data through model to plot it
    # FIXME: change
    X, T, Timetest = config.get_test_data(seed=rf, scaleX=scaleX, scaleT=scaleT)
    Xs, Ts = next(ndf.gen_batches(X, T, config.seq_length, None))
    Xtest = torch.from_numpy(Xs.astype(np.float32)).to(**config.to_kwargs)
    test_output = model(Xtest)
    # removing the scaling factor to plot in real units
    Ttest_alt = scaleT.inverse_transform(Ts)
    Ytest_alt = scaleT.inverse_transform(test_output.detach().cpu().numpy())

    # free up some GPU memory by deleting the tensors on the GPU
    del Xtest, test_output
    torch.cuda.empty_cache()

    # print('after test_alt delete')
    #!nvidia-smi

    # run full set of the last training data through model to plot it
    # FIXME: change
    (X, T, scaleX, scaleT, Timetrain) = config.get_train_data(seed=rf)
    Xs, Ts = next(ndf.gen_batches(X, T, config.seq_length, None))
    Xtrain = torch.from_numpy(Xs.astype(np.float32)).to(**config.to_kwargs)
    train_output = model(Xtrain)

    # removing the scaling factor to plot training and testing data in real units
    Ttrain_alt = scaleT.inverse_transform(Ts)
    Ytrain_alt = scaleT.inverse_transform(train_output.detach().cpu().numpy())

    # free up some GPU memory by deleting the tensors on the GPU
    del Xtrain, train_output
    torch.cuda.empty_cache()

    # print('after train_alt delete')
    #!nvidia-smi

    # ----------------------------------------------------------------------------------
    # plotting training convergence
    # -------------------------------

    print(f"---------------")
    print(f"\n\nAverage Training Error: {avg_error_train:.9f}")
    print(f"Average Testing Error: {avg_error_test:.9f}")

    fig = plt.figure(figsize=(20, 8))
    plt.semilogy(error_trace, "-b")
    ax1 = plt.gca()
    ax1.set_title("Training Error Convergence", fontsize=20)
    plt.ylabel("MSE Loss ", fontsize=14)
    plt.xlabel("# of Epochs", fontsize=14)
    plt.grid(visible=True, which="major", color="k", linestyle="-")
    plt.grid(visible=True, which="minor", color="0.9", linestyle="-")
    plt.xlim(0, 1000)
    plt.ylim(0.00001, 1)
    plt.show()
    # plotting Training Data
    # ----------------------
    fig = plt.figure(figsize=(20, 15))
    # fig.suptitle('Model vs Data',fontsize=20)

    # number of vertical subplots
    nv = 2

    # subplot 1
    ax1 = plt.subplot(nv, 1, 1)
    ax1.set_title("Training Data", fontsize=20)
    # plt.plot(Ttrain.detach().numpy()[42000:43000],      'ro', label='Target Altitude')
    plt.plot(Timetrain[config.seq_length:], Ttrain_alt, "ro", label="Target Altitude")
    plt.plot(Timetrain[config.seq_length:], Ytrain_alt, "b.", label="Model Altitude")
    plt.legend(fontsize=14)
    plt.ylabel("Altitude (ft) ", fontsize=14)

    # subplot 2
    istart = 42500
    istop = 48500

    ax2 = plt.subplot(nv, 1, 2)
    ax2.set_title("Training Data (Zoomed In)", fontsize=20)
    plt.plot(
        Timetrain[istart:istop], Ttrain_alt[istart:istop], "ro", label="Target Altitude"
    )
    plt.plot(
        Timetrain[istart:istop], Ytrain_alt[istart:istop], "b.", label="Model Altitude"
    )
    plt.legend(fontsize=14)
    plt.ylabel("Altitude (ft) ", fontsize=14)

    plt.xlabel("Time (GMTsecs)", fontsize=14)
    plt.show()
    # plotting Testing Data
    # ----------------------

    fig = plt.figure(figsize=(20, 15))
    # fig.suptitle('Model vs Test Data',fontsize=20)

    # number of vertical subplots
    nv = 2

    # subplot 1
    ax1 = plt.subplot(nv, 1, 1)
    ax1.set_title("Testing Data", fontsize=20)
    plt.plot(Timetest[config.seq_length:], Ttest_alt, "ro", label="Target Altitude")
    plt.plot(Timetest[config.seq_length:], Ytest_alt, "b.", label="Model Altitude")
    plt.legend(fontsize=14)
    plt.ylabel("Altitude (ft) ", fontsize=14)

    # subplot 2
    istart = 42500
    istop = 48500

    ax2 = plt.subplot(nv, 1, 2)
    ax2.set_title("Testing Data (Zoomed In)", fontsize=20)
    plt.plot(
        Timetest[istart:istop], Ttest_alt[istart:istop], "ro", label="Target Altitude"
    )
    plt.plot(
        Timetest[istart:istop], Ytest_alt[istart:istop], "b.", label="Model Altitude"
    )
    plt.legend(fontsize=14)
    plt.ylabel("Altitude (ft) ", fontsize=14)

    plt.xlabel("Time (GMTsecs)", fontsize=14)
    plt.show()


fig = plt.figure(figsize=(20, 15))
plt.plot(avg_error_train_list, "ko-", label="Training Data Avg MSE")
plt.plot(avg_error_test_list, "go-", label="Testing Data Avg MSE")
ax = plt.gca()
ax.set_title("LSTM Configuration Comparisons (Airborne data only) ", fontsize=30)
plt.xlabel("Configuration #", fontsize=18)
plt.ylabel("Avg Mean Sq Error", fontsize=18)
plt.xlim(0, 7)
plt.ylim(0, 0.02)
plt.legend(fontsize=18)


print(avg_error_train_list, avg_error_test_list)
# ntest
# pathlib.Path("model.pkl").write_bytes(pickle.dumps(model))
# model = pickle.loads(pathlib.Path("model.pkl").read_bytes())
# predict, score


def get_data(config, seed):
    Xtraindf, Ttraindf, _, scaleX, scaleT = config.get_train_data(seed=seed)
    Xtestdf, Ttestdf, *_ = config.get_test_data(
        seed=seed,
        scaleX=scaleX,
        scaleT=scaleT,
    )
    # Xtrain = torch.from_numpy(X.astype(np.float32)).to(self.device)
    # Ttrain = torch.from_numpy(T.astype(np.float32)).to(self.device)
    return (Xtraindf, Ttraindf, Xtestdf, Ttestdf)


def score(model, config, seed):
    (Xtrain, Ttrain, Xtest, Ttest) = (
        torch.from_numpy(el.astype(np.float32)).to(config.device)
        for el in get_data(config, seed)
    )
    # FIXME: make `get_loss` part of config
    train_loss = torch.nn.MSELoss()(model(Xtrain), Ttrain)
    test_loss = torch.nn.MSELoss()(model(Xtest), Ttest)
    return (train_loss, test_loss)



# need cached read of model into memory
# we also need the scaleX, scaleT for the model
@functools.cache
def cached_read_model(model_path):
    import pickle
    (model, scaleX, scaleT) = pickle.loads(model_path.read_bytes())
    return (model, scaleX, scaleT)


def predict_lstm(model_path, row):
    (model, scaleX, _) = cached_read_model(model_path)
    scaled = scaleX(row)
    return model(scaled)


# register udf with letsql
