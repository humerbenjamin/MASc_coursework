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
def cost_function(df_model, df_obs, H, params, params_b, obs_cov, params_cov):

    # get list of times that contribute to DA
    unique_times = np.sort(np.unique(df_obs.t))

    meas_delta_cost = 0
    for t_ind in range(len(unique_times)):
        u = df_model[df_model.t == unique_times[t_ind]].u
        obs = df_obs.u_noisy

        meas_delta_cost += (H@u - obs).T @ obs_cov @ (H@u - obs)

    meas_delta_cost *= 1/2

    param_diff_cost = 1/2 * (params - params_b).T @ params_cov @ (params - params_b)

    total_cost = meas_delta_cost + param_diff_cost

    return total_cost