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
# calculate adjoint variables
############################################################

# calculate the adjoint variables for the n=N case
############################################################
def calc_terminal_adjoint_variables(H, HT, model_state, obs_vector, obs_cov_inverse, has_observations=False):
    if has_observations:
        adjoint_vars = -HT @ obs_cov_inverse @ (H@model_state - obs_vector)
    else:
        adjoint_vars = np.zeros(len(model_state))

    return adjoint_vars


# calculate the adjoint variables for the n={1,...,N-1} case
############################################################
def calc_non_terminal_adjoint_variables(H, HT, model_state, obs_vector, obs_cov_inverse, model_jacobian_transpose, prev_adjoint_vars, has_observations=False):
    if has_observations:
        adjoint_vars = model_jacobian_transpose @ prev_adjoint_vars - HT @ obs_cov_inverse @ (H@model_state - obs_vector)
    else:
        adjoint_vars = model_jacobian_transpose @ prev_adjoint_vars

    return adjoint_vars