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
from scipy.sparse import linalg

# import personal files
############################################################
    # specific functions
from src.data_assimilation.calc_covariance_matrices import diagonal_obs_cov_matrix


############################################################
# calculate adjoint variables
############################################################

# calculate the adjoint variables for the n=N case
############################################################
def calc_terminal_adjoint_variables(
        H, HT, model_state, obs_vector,
        df_noisy_singletime, dt, has_observations=False
    ):

    if has_observations:
        obs_cov = diagonal_obs_cov_matrix(df_noisy_singletime)
        obs_cov_inverse = linalg.inv(obs_cov)

        adjoint_vars = -dt * HT @ obs_cov_inverse @ (H @ model_state - obs_vector)

    else:
        adjoint_vars = np.zeros(len(model_state))

    return adjoint_vars


# calculate the adjoint variables for the n={1,...,N-1} case
############################################################
def calc_non_terminal_adjoint_variables(
        H, HT, model_state, obs_vector,
        df_noisy_singletime, model_jacobian_transpose,
        prev_adjoint_vars, dt_previous, dt_next,
        has_observations=False
    ):

    if has_observations:
        obs_cov = diagonal_obs_cov_matrix(df_noisy_singletime)
        obs_cov_inverse = linalg.inv(obs_cov)

        adjoint_vars = dt_previous * (
            model_jacobian_transpose @ (prev_adjoint_vars / dt_next)
            - HT @ obs_cov_inverse @ (H @ model_state - obs_vector)
        )

    else:
        adjoint_vars = dt_previous * (
            model_jacobian_transpose @ (prev_adjoint_vars / dt_next)
        )

    return adjoint_vars