# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
    # sublibraries
from scipy.sparse import diags
from time import time



############################################################
# generate different forms of observation covariance matrices
############################################################


# diagonal matrix
############################################################
def diagonal_obs_cov_matrix(df_noisy_singletime):
    diag_vec = df_noisy_singletime.stdev.to_numpy()
    obs_cov_matrix = diags(diag_vec**2, offsets=0, format='csc')

    return obs_cov_matrix