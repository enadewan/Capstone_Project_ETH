import torch as torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from helpers.gp_gpytorch import EGPModel, IpGpModel
from helpers.gp_manual import ManualSsGpModel, train_hyperparams
from helpers.gboost import GboostModel
import gpytorch


from helpers.data_generation import create_data
from helpers.plotting import (
    print_accuracy,
    regression_plot_1d,
    regression_plot_2d,
    condense_to_2d,
)

from sklearn.metrics import mean_squared_error

TRAIN_NEW_PARAMS = False
VERBOSE = True
DIM = 3
FIELD = "Liu"
N_ITER = 10

if DIM == 1:
    n_train = 50
    n_ip = 10
    n_bf = 5
    n_epoch = 30
elif DIM == 2:
    n_train = 500
    n_ip = 50
    n_bf = 20
    n_epoch = 20
elif DIM == 3:
    n_train = 1000
    n_ip = 100
    n_bf = 60
    i_z = 14
    n_epoch = 15
else:
    raise ValueError("The dimension needst to be between 1 and 3.")


models = ["egp", "ipgp", "ssgp", "gboost"]

accuracy = {}
timing = {}
for model in models:
    accuracy[model] = {}
    accuracy[model]["train"] = np.zeros(shape=(N_ITER, 1))
    accuracy[model]["test"] = np.zeros(shape=(N_ITER, 1))
    timing[model] = {}
    timing[model]["pred"] = np.zeros(shape=(N_ITER, 1))


predictors = {}
likelihoods = {}
states = {}
for i in range(N_ITER):
    (
        X_train,
        y_train,
        gridPoints,
        mean_true,
        var_true,
        limits,
    ) = create_data(dim=DIM, n_train=n_train, field=FIELD)

    for model in models:
        if model == "egp":
            likelihoods[model] = gpytorch.likelihoods.GaussianLikelihood()
            predictors[model] = EGPModel(X_train, y_train, likelihoods[model])
            file_name = f"egp_{DIM}_{FIELD}.pth"
        elif model == "ipgp":
            likelihoods[model] = gpytorch.likelihoods.GaussianLikelihood()
            predictors[model] = IpGpModel(X_train, y_train, likelihoods[model], n_ip)
            file_name = f"ipgp_{DIM}_{n_ip}_{FIELD}.pth"
        elif model == "ssgp":
            likelihoods[model] = None
            predictors[model] = ManualSsGpModel(X_train, y_train, n_bf)
            file_name = f"ssgp_{DIM}_{n_bf}_{FIELD}.pth"
        elif model == "gboost":
            likelihoods[model] = None
            predictors[model] = GboostModel(X_train, y_train)
            file_name = f"gboost_{DIM}_{FIELD}.pkl"

        if i == 0:
            full_name = f"results\\{model}_{DIM}_{FIELD}"
            # train and save hyperparameters
            if TRAIN_NEW_PARAMS:
                if model == "gboost":
                    states[model] = predictors[model].train_hyperparams()
                    with open(full_name + ".pkl", "wb") as f:
                        pickle.dump(states[model], f)
                else:
                    train_hyperparams(
                        predictors[model],
                        likelihoods[model],
                        X_train,
                        y_train,
                        n_epoch,
                        VERBOSE,
                    )

                    states[model] = predictors[model].state_dict()
                    # save hyperparameters
                    torch.save(states[model], full_name + ".pth")

            else:  # don't train new hyperparams but load existing ones
                if model == "gboost":  # from pickle file
                    with open(full_name + ".pkl", "rb") as f:
                        states[model] = pickle.load(f)
                    predictors[model].load_params(states[model])
                else:  # from pth file
                    states[model] = torch.load(full_name + ".pth")
                    predictors[model].load_state_dict(states[model])
                    if model == "ssgp":
                        predictors[model].update_kernel()

        else:  # for iterations > 0,
            if model == "gboost":
                predictors[model].load_params(states[model])
            else:
                # load hyperparameters
                predictors[model].load_state_dict(states[model])
                if model == "ssgp":
                    predictors[model].update_kernel()

        # prediction
        s = time.time()
        test_pred = predictors[model].predict(gridPoints)
        timing[model]["pred"][i] = time.time() - s
        train_pred = predictors[model].predict(X_train)

        # accuracy
        accuracy[model]["train"][i] = mean_squared_error(y_train, train_pred["mean"])
        accuracy[model]["test"][i] = mean_squared_error(mean_true, test_pred["mean"])


############################################################## SAVE DATA IN CSV
timing_df = pd.read_csv("results\\timing.csv")
for model in models:
    timing_df[f"{model}_{DIM}_{FIELD}"] = timing[model]["pred"]

timing_df.to_csv("results\\timing.csv", index=False)

test_error_df = pd.read_csv("results\\test_error.csv")
for model in models:
    test_error_df[f"{model}_{DIM}_{FIELD}"] = accuracy[model]["test"]

test_error_df.to_csv("results\\test_error.csv", index=False)

############################################################## PRINT ACCURACY


print_accuracy("EGP   ", accuracy["egp"])
print_accuracy("IPGP  ", accuracy["ipgp"])
print_accuracy("SSGP  ", accuracy["ssgp"])
print_accuracy("GBOOST", accuracy["gboost"])

############################################################## GRID PREDICTION
pred_data = dict()

pred_data["true_mean"] = mean_true.numpy()
pred_data["true_var"] = var_true.numpy()

for model in models:
    pred_data[model] = predictors[model].predict(gridPoints)

############################################################## PLOT RESULTS


if DIM == 1:
    fig = regression_plot_1d(
        gridPoints.numpy(), pred_data, X_train.numpy(), y_train.numpy(), limits, FIELD
    )
elif DIM == 2:
    fig = regression_plot_2d(
        gridPoints.numpy(), pred_data, X_train.numpy(), y_train.numpy(), limits, FIELD
    )
elif DIM == 3:
    gridPoints_2d, pred_data_2d, X_train_2d, y_train_2d = condense_to_2d(
        gridPoints.numpy(), pred_data, X_train.numpy(), y_train.numpy(), i_z
    )
    fig = regression_plot_2d(
        gridPoints_2d, pred_data_2d, X_train_2d, y_train_2d, limits, FIELD
    )
else:
    raise ValueError("The dimension of the problem must be between 1 and 3.")
plt.show(block=True)


pic_name = "reg_main_{}D_{}.pdf".format(DIM, FIELD)
fig.savefig(pic_name, format="pdf", dpi=300, bbox_inches="tight")
