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
# implement standard cost function
############################################################
def cost_function_with_parameter_regularization(df_model, df_obs, H, params, params_b, obs_cov, params_cov):

    # get list of times that contribute to DA
    unique_times = np.sort(np.unique(df_obs.t))

    meas_delta_cost = 0
    for t_ind in range(len(unique_times)):
        u = df_model[np.isclose(df_model.t, unique_times[t_ind], atol=1e-10)].u
        obs = df_obs[np.isclose(df_obs.t, unique_times[t_ind], atol=1e-10)].u_noisy

        meas_delta_cost += (H@u - obs).T @ obs_cov @ (H@u - obs)

    meas_delta_cost *= 1/2

    param_diff_cost = 1/2 * (params - params_b).T @ params_cov @ (params - params_b)

    total_cost = meas_delta_cost + param_diff_cost

    return total_cost


############################################################
# function to determine the step size
############################################################
def backtracking_line_search(
        k, k0, gradient,
        cost_function,
        df_model, df_obs, H, obs_cov, params_cov,
        alpha_initial=1.0, beta=0.5, c=1e-4,
        max_iterations=500
    ):

    alpha = alpha_initial
    J_current = cost_function(df_model, df_obs, H, k, k0, obs_cov, params_cov)

    gradient_norm_squared = np.dot(gradient, gradient)

    for _ in range(max_iterations):

        k_trial = k - alpha * gradient

        J_trial = cost_function(df_model, df_obs, H, k, k0, obs_cov, params_cov)

        if J_trial <= J_current - c * alpha * gradient_norm_squared:
            return k_trial, alpha, J_trial

        alpha *= beta

    raise RuntimeError("Line search failed to find an acceptable step.")