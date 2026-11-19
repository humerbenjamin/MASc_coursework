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
from src.postprocessing.plot_parameter_evolution import plot_source_function_evolution

    # source functions
from src.source_functions.source_functions import gaussian_x_multiplied_by_u

# load in dataframes
k_hist_df = pd.read_csv("data/test_k_history.csv")
df_ic = pd.read_csv("data/test_rawdata_IC.csv")

x = df_ic.x.to_numpy()

k_exact = [-2,0,0.5]

s_function = gaussian_x_multiplied_by_u

plot_source_function_evolution(x, s_function, k_exact, k_hist_df)