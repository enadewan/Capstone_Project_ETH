# Capstone project: (Sparse) GP in low dimensions
This repository was created for the the Capstone Project **Sparse techniques for real-time regression** by Ena Dewan and Gregor Ochsner at ETH Zurich. The project was supervised by Victor Cohen.

## Folders

### helpers
The folder *helpers* contains several files with auxiliary functions (check Helper fuctions below)
### results
The folder *results* contains the .csv files with the main regression results data, and all the files containing the trained hyperparameters for the different models. Additionally, the folder contains the script *print_latex_tables.py* that loads the .csv files and prints out the final results table in LaTeX code.
## Main experiments
Each of the main experiments in the report can be reproduced by running a separate python script.
### Experiment 1: Kernel comparison
The file *kernel_function_main.py* allows reproducing the kernel comparison experiment, where different types of kernels are comprared on 1D data.

### Experiment 2: Sparsity analysis
The file *sparsity_main.py* allows reproducing the sparsity experiment, where the number of inducing points (Inducing Point GP) and the number of trigonometric basis functions (Sparse Spectrum GP) are varied to analyze their effect on the prediction accuracy and prediciton time in 2D.

### Experiment 3: Regression
The file *regression_main.py* allows running the regression experiment for 1D, 2D, and 3D data for different data fields.

## Addional experiments
The files *manual_comp_egp.py* and *manual_comp_ipgp.py* compare the manual GP implementations of the exact GP and the inducing point GP with the implementations based on GPyTorch. These files are not required for the experiments in the report.

## Helper functions
The file *gp_gpytroch.py* contains all GP models that are based on the GPyTorch module.

The file *gp_manual.py* contains the self-implemented GP models and many auxiliary functions for the GPs.

The file *gboost.py* contains all functions related to the gradient boosting regressor.

The file *data_generation.py* with it's function *createData()* allows creating datasets from different artificicial fields in 1D, 2D, or 3D.

The file *plotting.py* contains all the plotting functions to generate the plots for the project report.
