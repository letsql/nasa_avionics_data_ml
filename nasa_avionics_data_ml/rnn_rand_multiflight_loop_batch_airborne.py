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
    instance_of,
)

import nasa_avionics_data_ml.nasa_data_funcs as ndf
from nasa_avionics_data_ml.LSTM import LSTM


default_pdir = pathlib.Path("/mnt/nasa-data-download/data/Tail_666_1_parquet/")


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
    train_frac = field(validator=instance_of(float), default=2/3)

    def __attrs_post_init__(self):
        if not self.pdir.exists():
            raise ValueError

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
        paths = shuffle(list(self.paths), seed)
        n_train = int(len(paths) * self.train_frac)
        train_paths = paths[:n_train]
        return train_paths

    def get_test_paths(self, seed):
        paths = shuffle(list(self.paths), seed)
        n_train = int(len(paths) * self.train_frac)
        test_paths = paths[n_train:]
        return test_paths

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


avg_error_train_list = []
avg_error_test_list = []
# loop through all the different configurations
for (config_i, config) in enumerate(configs):
    seq_length = config.seq_length
    learning_rate = config.l_rate
    n_epochs = config.epoch
    hidden_units = config.n_h_unit
    n_hidden_layers = config.n_h_layer

    print(f"\n\n\n----------------------\nConfiguration #{config_i}\n----------------------")
    print(f"seq_length:{seq_length}")
    print(f"learning_rate:{learning_rate}")
    print(f"n_epochs:{n_epochs}")
    print(f"hidden_units:{hidden_units}")
    print(f"n_hidden_layers:{n_hidden_layers}")
    print(f"n_files:{config.n_files}\n-----------------")

    avg_error_train = 0
    avg_error_test = 0
    # bootstrap loop
    for rf in range(config.n_rand_loops):
        # shuffle the files around in this list to get different results for the bootstrapping
        train_paths = config.get_train_paths(seed=rf)

        # print('Loading Data...')
        Xtraindf, Ttraindf, Timetrain, scaleX, scaleT = ndf.read_parquet_flight_merge(
            train_paths,
            seq_length,
            scaleX=None,
            scaleT=None,
            VRTG=True,
            airborne_only=True,
        )

        ## Training
        # ----------------------------------

        # varibles described at the beginning of this notebook.
        # number of samples, samples per sequence, components per sample
        N, S, I = Xtraindf.shape
        O = Ttraindf.shape[1]
        n_input = I
        n_out = O

        # calling the LSTM definition above and initializing the model
        model = LSTM(n_input, n_out, hidden_units, n_hidden_layers, config.device)
        # set computational resource to the "device"
        model = model.to(config.device)

        # Define the loss function used to calculate the difference between the model output and the training Target
        loss_func = torch.nn.MSELoss()

        # optimization function
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        error_trace = []

        # training the model
        # print('Training the Model...')
        for epoch in range(n_epochs):
            for i, (X, T) in enumerate(
                ndf.get_batch(Xtraindf, Ttraindf, batch_size=config.batch_size)
            ):
                # create tensors from the data frames
                Xtrain = torch.from_numpy(X.astype(np.float32)).to(**config.to_kwargs)
                Ttrain = torch.from_numpy(T.astype(np.float32)).to(**config.to_kwargs)

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
        avg_error_train += loss.item()

        # print('after epoch loop')
        #!nvidia-smi

        del Xtrain, Ttrain, train_output, loss
        torch.cuda.empty_cache()
        # print('after epoch loop delete')
        #!nvidia-smi

        # testing the model
        # --------------------------
        # print('Testing the Model...')
        test_paths = config.get_test_paths(seed=rf)
        Xtestdf, Ttestdf, Timetest, scaleX, scaleT = ndf.read_parquet_flight_merge(
            test_paths,
            seq_length,
            scaleX=scaleX,
            scaleT=scaleT,
            VRTG=True,
            airborne_only=True,
        )

        # create tensors from the data frames
        Xtest = torch.from_numpy(Xtestdf.astype(np.float32)).to(**config.to_kwargs)
        Ttest = torch.from_numpy(Ttestdf.astype(np.float32)).to(**config.to_kwargs)

        # Run the test data through the trained model
        test_output = model(Xtest)
        # calculate the loss of the model output (Y) to the training Target (Ttrain)
        loss = loss_func(test_output, Ttest).detach().cpu()

        avg_error_test += loss.item()

        del Xtest, Ttest, test_output, loss
        torch.cuda.empty_cache()
        # print('after test delete')
        #!nvidia-smi

    # calculating error over all the random runs
    avg_error_train = avg_error_train / config.n_rand_loops
    avg_error_train_list.append(avg_error_train)
    avg_error_test = avg_error_test / config.n_rand_loops
    avg_error_test_list.append(avg_error_test)

    # run full set of the last training data through model to plot it
    Xtest = torch.from_numpy(Xtestdf.astype(np.float32)).to(**config.to_kwargs)
    test_output = model(Xtest)
    # removing the scaling factor to plot in real units
    Ttest_alt = scaleT.inverse_transform(Ttestdf)
    Ytest_alt = scaleT.inverse_transform(test_output.detach().cpu().numpy())

    # free up some GPU memory by deleting the tensors on the GPU
    del test_output, Ttestdf, Xtest, Xtestdf
    torch.cuda.empty_cache()

    # print('after test_alt delete')
    #!nvidia-smi

    # run full set of the last training data through model to plot it
    Xtrain = torch.from_numpy(Xtraindf.astype(np.float32)).to(**config.to_kwargs)
    train_output = model(Xtrain)

    # removing the scaling factor to plot training and testing data in real units
    Ttrain_alt = scaleT.inverse_transform(Ttraindf)
    Ytrain_alt = scaleT.inverse_transform(train_output.detach().cpu().numpy())

    # free up some GPU memory by deleting the tensors on the GPU
    del Xtrain, train_output, Xtraindf, Ttraindf
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
    plt.plot(Timetrain, Ttrain_alt, "ro", label="Target Altitude")
    plt.plot(Timetrain, Ytrain_alt, "b.", label="Model Altitude")
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
    plt.plot(Timetest, Ttest_alt, "ro", label="Target Altitude")
    plt.plot(Timetest, Ytest_alt, "b.", label="Model Altitude")
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
