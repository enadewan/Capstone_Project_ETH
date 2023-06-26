# Sampling and drawing from analytic field
import torch as torch
import matplotlib.pyplot as plt
import numpy as np

from helpers.gp_gpytorch import EGPModel
from helpers.gp_manual import train_hyperparams
import gpytorch

from helpers.data_generation import create_data
from helpers.plotting import print_accuracy, kernel_comp_plot_1d
from sklearn.metrics import mean_squared_error


VERBOSE = True
N_ITER = 10
DIM = 1
N_TRAIN = 300
FIELD = "Liu"

kernels = ["rbf", "poly", "cos", "matern"]


accuracy = {}
for kernel in kernels:
    accuracy[kernel] = {}
    accuracy[kernel]["train"] = np.zeros(shape=N_ITER)
    accuracy[kernel]["test"] = np.zeros(shape=N_ITER)


likelihoods = {}
gps = {}
states = {}
train_pred = {}
test_pred = {}
pred = {}

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
        likelihoods[kernel] = gpytorch.likelihoods.GaussianLikelihood()
        gps[kernel] = EGPModel(X_train, y_train, likelihoods[kernel], kernel=kernel)

        if i == 0:
            # train and save hyperparameters
            train_hyperparams(
                gps[kernel], likelihoods[kernel], X_train, y_train, 40, VERBOSE
            )
            states[kernel] = gps[kernel].state_dict()
        else:
            # load hyperparameters
            gps[kernel].load_state_dict(states[kernel])

        # prediction
        train_pred[kernel] = gps[kernel].predict(X_train)
        test_pred[kernel] = gps[kernel].predict(gridPoints)

        # accuracy
        accuracy[kernel]["train"][i] = mean_squared_error(
            y_train, train_pred[kernel]["mean"]
        )
        accuracy[kernel]["test"][i] = mean_squared_error(
            mean_true, test_pred[kernel]["mean"]
        )


############################################################## GRID PREDICTION
pred_data = dict()
pred_data["true_mean"] = mean_true.numpy()
pred_data["true_var"] = var_true.numpy()

for kernel in kernels:
    pred_data[kernel] = gps[kernel].predict(gridPoints)

    print_accuracy(f"{kernel}", accuracy[kernel])

############################################################## PLOTTING
fig = kernel_comp_plot_1d(
    gridPoints.numpy(), pred_data, X_train, y_train, limits, FIELD
)
plt.show(block=True)

pic_name = "kernel_{}.pdf".format(FIELD)
fig.savefig(pic_name, format="pdf", dpi=300)
