import matplotlib.pyplot as plt
import numpy as np
import torch

import nasa_avionics_data_ml.nasa_data_funcs as ndf
from nasa_avionics_data_ml.lib import (
    Config,
    print_cuda_settings,
    set_device,
)


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
