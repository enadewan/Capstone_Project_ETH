import torch as torch

import matplotlib.pyplot as plt
from helpers.plotting import plot_1d_mean, plot_1d_var

from helpers.gp_gpytorch import IpGpModel
from helpers.gp_manual import ManualIpGpModel, train_hyperparams
import gpytorch

from helpers.data_generation import create_data

FIELD = "Liu"
(
    X_train,
    y_train,
    grid_points,
    mean_true,
    var_true,
    limits,
) = create_data(dim=1, n_train=100, field=FIELD)

######################### GPYTORCH ###############################
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = IpGpModel(X_train, y_train, likelihood, 5)

# hyperparameter training
train_hyperparams(model, likelihood, X_train, y_train, 25)

# save parameters for manual GP implementation
likeihood_noise_variance = model.likelihood.noise[0]
kernel_std = model.covar_module.base_kernel.lengthscale[0][0]
inducing_pints = model.covar_module.inducing_points

print("GPyTorch")
print(inducing_pints)
print(torch.sqrt(likeihood_noise_variance))
print(kernel_std)


# predict
model_pred = model.predict(grid_points)


######################### MANUAL GP ###############################

manual = ManualIpGpModel(X_train, y_train, X_train[:5, :].clone())

# Option 1: use parameters from gpytorch model
# manual.inducing_x = inducing_pints
# manual.sigma_data = torch.nn.Parameter(torch.sqrt(likeihood_noise_variance))
# manual.sigma_kernel = torch.nn.Parameter(kernel_std)

# Option 2:  train hyperparameters
train_hyperparams(manual, None, X_train, y_train, 50)

manual.update_kernel()
print("Manual")
print(manual.inducing_x)
print(manual.sigma_data)
print(manual.sigma_kernel)

manual_pred = manual.predict(grid_points)

######################### PLOTTING ###############################

fig = plt.figure(figsize=(8, 5))

if FIELD == "Liu":
    ylim = (-1.5, 2.5)
elif FIELD == "Goldberg":
    ylim = (-4, 4)

ax0 = fig.add_subplot(1, 2, 1)
ax0.set_title("GPyTorch")
plot_1d_mean(ax0, grid_points, mean_true, model_pred["mean"])
plot_1d_var(ax0, grid_points, model_pred)
ax0.scatter(X_train, y_train, color="b", s=5.0, label="Training data")
ax0.set_ylabel(r"y")
ax0.set_xlabel(r"x")
ax0.set_xlim((limits["min"], limits["max"]))
ax0.set_ylim(ylim)
# ax0.legend()

ax1 = fig.add_subplot(1, 2, 2)
ax1.set_title("Manual")
plot_1d_mean(ax1, grid_points, mean_true, manual_pred["mean"])
plot_1d_var(ax1, grid_points, manual_pred)
ax1.scatter(X_train, y_train, color="b", s=5.0, label="Training data")
ax1.set_ylabel(r"y")
ax1.set_xlabel(r"x")
ax1.set_xlim((limits["min"], limits["max"]))
ax1.set_ylim(ylim)

plt.show(block=True)
