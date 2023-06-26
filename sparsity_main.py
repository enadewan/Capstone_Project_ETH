# Sampling and drawing from analytic field
import torch as torch
import matplotlib.pyplot as plt
import numpy as np
import time

from helpers.gp_gpytorch import EGPModel, IpGpModel
from helpers.gp_manual import ManualSsGpModel, train_hyperparams
import gpytorch

from helpers.data_generation import create_data
from helpers.plotting import plot_accuracy
from sklearn.metrics import mean_squared_error


VERBOSE = True
N_ITER = 10
DIM = 2
N_TRAIN = 1000
FIELD = "Liu"

kernels = [
    "ipgp_5",
    "ipgp_20",
    "ipgp_50",
    "ipgp_100",
    "ssgp_3",
    "ssgp_5",
    "ssgp_10",
    "ssgp_20",
    "egp_1",
]


accuracy = {}
timing = {}
for kernel in kernels:
    accuracy[kernel] = {}
    accuracy[kernel]["train"] = np.zeros(shape=N_ITER)
    accuracy[kernel]["test"] = np.zeros(shape=N_ITER)
    timing[kernel] = {}
    timing[kernel]["instant"] = np.zeros(shape=N_ITER)
    timing[kernel]["train"] = np.zeros(shape=N_ITER)
    timing[kernel]["pred"] = np.zeros(shape=N_ITER)

likelihoods = {}
gps = {}
states = {}
train_pred = {}
test_pred = {}
grid_pred = {}


for i in range(N_ITER):
    (
        X_train,
        y_train,
        gridPoints,
        mean_true,
        var_true,
        limits,
    ) = create_data(dim=DIM, n_train=N_TRAIN, field=FIELD)

    for kernel in kernels:
        print(f"--- {i}: {kernel} ---")
        k_split = kernel.split("_")
        n_sparse = int(k_split[1])
        s = time.time()
        if k_split[0] == "egp":
            likelihoods[kernel] = gpytorch.likelihoods.GaussianLikelihood()
            gps[kernel] = EGPModel(X_train, y_train, likelihoods[kernel])
        elif k_split[0] == "ipgp":
            likelihoods[kernel] = gpytorch.likelihoods.GaussianLikelihood()
            gps[kernel] = IpGpModel(
                X_train, y_train, likelihoods[kernel], n_ind=n_sparse
            )
        elif k_split[0] == "ssgp":
            gps[kernel] = ManualSsGpModel(X_train, y_train, n_freqs=n_sparse)
            likelihoods[kernel] = None

        timing[kernel]["instant"][i] = time.time() - s

        s = time.time()
        # train hyperparameters
        train_hyperparams(
            gps[kernel],
            likelihoods[kernel],
            X_train,
            y_train,
            n_steps=40,
            VERBOSE=VERBOSE,
        )

        timing[kernel]["train"][i] = (time.time() - s) / 40.0

        # prediction
        # train_pred[kernel] = gps[kernel].predict(X_train)
        s = time.time()
        test_pred[kernel] = gps[kernel].predict(gridPoints)
        timing[kernel]["pred"][i] = time.time() - s

        # accuracy[kernel]["train"][i] = mean_squared_error(
        #     y_train, train_pred[kernel]["mean"]
        # )
        accuracy[kernel]["test"][i] = mean_squared_error(
            mean_true, test_pred[kernel]["mean"]
        )


############################################################## PLOTTING

fig = plot_accuracy(accuracy, timing, kernels)
plt.show(block=True)

pic_name = "sparsity.pdf"
fig.savefig(pic_name, format="pdf", dpi=300)
