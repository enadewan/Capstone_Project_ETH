import torch as torch
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

from helpers.gp_manual import ManualEGpModel, train_hyperparams
import gpytorch
from helpers.gp_gpytorch import EGPModel

from helpers.data_generation import create_data
from helpers.plotting import plot_1d_mean, plot_1d_var


FIELD = "Liu"
(
    X_train,
    y_train,
    grid_points,
    mean_true,
    var_true,
    limits,
) = create_data(dim=1, n_train=50, field=FIELD)


######################### GPyTorch ###############################
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = EGPModel(X_train, y_train, likelihood)

# hyperparameter training
train_hyperparams(model, likelihood, X_train, y_train, 25)

likeihood_noise_variance = model.likelihood.noise[0]
kernel_std = model.covar_module.lengthscale[0][0]

print("GPyTorch")
print(torch.sqrt(likeihood_noise_variance))
print(kernel_std)


model_pred = model.predict(grid_points)


######################### MANUAL GP ###############################
manual = ManualEGpModel(X_train, y_train)


# Option 1: use parameters from gpytorch model
# manual.sigma_data = torch.nn.Parameter(torch.sqrt(likeihood_noise_variance))
# manual.sigma_kernel = torch.nn.Parameter(kernel_std.detach())

# Option 2:  train hyperparameters
train_hyperparams(manual, None, X_train, y_train, n_steps=25)

manual.update_kernel()
print("Manual")
print(manual.sigma_data)
print(manual.sigma_kernel)

manual_pred = manual.predict(grid_points)

######################### PLOTTING ###############################

if FIELD == "Liu":
    ylim = (-1.5, 2.5)
elif FIELD == "Goldberg":
    ylim = (-4, 4)

fig = plt.figure(figsize=(16, 9))


ax0 = fig.add_subplot(1, 2, 1)
ax0.set_title("GPyTorch")
plot_1d_mean(ax0, grid_points, mean_true, model_pred["mean"])
plot_1d_var(ax0, grid_points, model_pred)
ax0.scatter(X_train, y_train, color="b", s=5.0, label="Training data")
ax0.set_ylabel(r"y")
ax0.set_xlabel(r"x")
ax0.set_xlim((limits["min"], limits["max"]))
ax0.set_ylim(ylim)

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
