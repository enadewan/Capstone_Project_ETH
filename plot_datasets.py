import torch as torch
import matplotlib.pyplot as plt


from helpers.data_generation import create_data
from helpers.plotting import plot_true_mean_3d, plot_true_mean_2d, plot_true_mean_1d

# This script creates plots for both data fields in 1D, 2D, and 2D for the report.
for dim in [1, 2, 3]:
    for field in ["Liu", "Goldberg"]:
        (
            X_train,
            y_train,
            gridPoints,
            mean_true,
            var_true,
            limits,
        ) = create_data(dim=dim, n_train=1000, field=field)

        if dim == 1:
            var_plot = {}
            var_plot["lower_data"] = (mean_true + 2 * torch.sqrt(var_true)).numpy()
            var_plot["upper_data"] = (mean_true - 2 * torch.sqrt(var_true)).numpy()
            fig = plot_true_mean_1d(
                gridPoints.numpy(), mean_true.numpy(), var_plot, limits, field
            )
        elif dim == 2:
            fig = plot_true_mean_2d(
                gridPoints.numpy(),
                mean_true,
                var_true.numpy(),
                limits,
                field,
            )
        elif dim == 3:
            fig = plot_true_mean_3d(
                gridPoints.numpy(), mean_true.numpy(), limits, field
            )
        else:
            raise ValueError("The dimension of the problem must be between 1 and 3.")
        plt.show(block=True)

        pic_name = "true_mean_{}D_{}.pdf".format(dim, field)
        fig.savefig(pic_name, format="pdf", dpi=300, bbox_inches="tight")
