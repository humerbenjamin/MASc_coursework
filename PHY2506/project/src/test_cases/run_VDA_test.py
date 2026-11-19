# add the src header to pythonpath
############################################################
import sys
sys.path.append(".")

# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
    # sublibraries
from scipy.sparse import csr_matrix


# import personal files
############################################################
    # specific functions
from src.data_assimilation.generate_DA_data import generate_data
from src.data_assimilation.transient_data_assimilation import variational_DA_transient
from src.source_functions.source_functions import gaussian_x_multiplied_by_u
from src.source_functions.source_function_derivatives import gaussian_x_multiplied_by_u_du, gaussian_x_multiplied_by_u_dk




############################################################
############################################################
# Generate data
############################################################
############################################################

# direct solver inputs
n_points=200
x = np.linspace(-1,1,n_points)
u = np.zeros(len(x)) # np.append(1*np.ones(int(n_points/2)), -1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1

# source function
source_parameters = [-2,0,0.5]

# system parameters
kin_viscosity = 0
BCs = [2,-2]

# solver parameters
n_iter = 10000
C = 0.2
convergence_threshold = 1e-6

# type of solve (steady or transient)
solve_type='steady'
solve_type='transient'

# measurement locations
measurement_locations = np.linspace(-1,1,40)

generate_data(  
                solve_type, u, x,                                                           # direct functional inputs
                gaussian_x_multiplied_by_u, source_parameters,                              # source function and parameters
                kin_viscosity, BCs,                                                         # additional system parameters
                n_iter, C, convergence_threshold,                                           # solver parameters for steady solver
                t_f=1.5, data_save_interval=0.1,                                            # solver parameters for transient solve
                verbose=False, plot_values=True,                                            # midstream outputs
                output_filename='data/test',                                                # data locations
                measurement_locations=measurement_locations, random_seed=42, SNR=0.1        # noisy measurement generation
             )



############################################################
############################################################
# VDA test case
############################################################
############################################################

S_fcn = gaussian_x_multiplied_by_u
S_fcn_du = gaussian_x_multiplied_by_u_du
S_fcn_dk = gaussian_x_multiplied_by_u_dk

source_parameters_initial = [-1.5, 0.4, 0.25]

verbose=True

parameter_covariance_matrix = np.eye(3)
obs_cov_function = np.eye(40)


variational_DA_transient(   
                            "data/test", S_fcn, source_parameters_initial, S_fcn_du, S_fcn_dk,
                            parameter_covariance_matrix, obs_cov_function, C, kin_viscosity, BCs,
                            verbose=verbose
                        )