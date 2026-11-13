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
from time import time



############################################################
# calculate adjoint varaibles
############################################################

# calculate the adjoint variables for the n=N case
############################################################
def calc_terminal_adjoint_variables(H, HT, model_state, obs_vector, obs_cov_inverse, has_observations=False):
    if has_observations:
        adjoint_vars = -HT @ obs_cov_inverse @ (H@model_state - obs_vector)
    else:
        adjoint_vars = np.zeros(H.size[0])

    return adjoint_vars


# calculate the adjoint variables for the n={1,...,N-1} case
############################################################
def calc_non_terminal_adjoint_variables(H, HT, model_state, obs_vector, obs_cov_inverse, model_jacobian_transpose, prev_adjoint_vars, has_observations=False):
    print("\t",np.size(model_jacobian_transpose), np.size(prev_adjoint_vars))
    if has_observations:
        adjoint_vars = model_jacobian_transpose @ prev_adjoint_vars - HT @ obs_cov_inverse @ (H@model_state - obs_vector)
    else:
        adjoint_vars = model_jacobian_transpose @ prev_adjoint_vars

    return adjoint_vars