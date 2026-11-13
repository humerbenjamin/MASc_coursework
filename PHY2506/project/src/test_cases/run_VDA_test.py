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
from src.data_assimilation.transient_data_assimilation import variational_DA_transient
from src.source_functions.source_functions import gaussian_x_multiplied_by_u
from src.source_functions.source_function_derivatives import gaussian_x_multiplied_by_u_du, gaussian_x_multiplied_by_u_dk



############################################################
############################################################
# VDA test case
############################################################
############################################################

S_fcn = gaussian_x_multiplied_by_u
S_fcn_du = gaussian_x_multiplied_by_u_du
S_fcn_dk = gaussian_x_multiplied_by_u_dk

source_parameters = [-2, 0, 0.5]

C = 0.2

nu = 0
BCs = [2,-2]

verbose=False

variational_DA_transient("data/test", S_fcn, source_parameters, S_fcn_du, S_fcn_dk, C, nu, BCs, verbose)