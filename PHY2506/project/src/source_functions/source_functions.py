# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
    # sublibraries



# import personal files
############################################################
    # specific functions



############################################################
############################################################
# GENERAL FORMAT OF SOURCE FUNCTIONS
############################################################
############################################################
'''
All source functions must have the following form:
    source_function(u,x,t,source_params)
        where u is the solution vector for all points x
        x is the solution domain
        t is the current solution time for the given solution vector u
        and source_params is an array of arbitrary length that are used to parametrize the source function

This form is required because it is how the function is called throughout the solver as these source functions are passed through the code.
If there is no time dependence, or the source function is not parametrized, the variables must still be included.

There may be optional hyperparameters specified after the standard source_params.
'''




############################################################
############################################################
# define source functions
############################################################
############################################################

############################################################
# source functions that are gaussian in x
############################################################

# gaussian in x multiplied by u, weighted by constant a
############################################################
def gaussian_x_multiplied_by_u(u,x,t,source_params):
    S = source_params[0]*u *  1 / np.sqrt(2*np.pi*source_params[2]**2) * np.exp((-(x-source_params[1])**2) / (2*source_params[2]**2))

    return S