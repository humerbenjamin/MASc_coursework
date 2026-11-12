# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
    # sublibraries
from warnings import warn


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
# define source function derivatives
############################################################
############################################################

############################################################
# source functions that are gaussian in x
############################################################

# gaussian in x multiplied by u, weighted by constant a
############################################################

# diff wrt u
def dS_du(u, x, t, source_params):
    a, mu, sigma = source_params
    return (a / np.sqrt(2 * np.pi * sigma**2)
            * np.exp(-(x - mu)**2 / (2 * sigma**2)))

# diff wrt k
def dS_dk(u, x, t, source_params, k_index=0):
    a, mu, sigma = source_params
    if k_index == 0:
        return (u / np.sqrt(2 * np.pi * sigma**2)
                * np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    elif k_index == 1:
        return (a * u / np.sqrt(2 * np.pi * sigma**2)
                * np.exp(-(x - mu)**2 / (2 * sigma**2))
                * (x - mu) / sigma**2)

    elif k_index == 2:
        return (a * u / np.sqrt(2 * np.pi * sigma**2)
                * np.exp(-(x - mu)**2 / (2 * sigma**2))
                * (-1 / sigma + (x - mu)**2 / sigma**3))

    else:
        raise ValueError("Invalid value of k_index.\n\tPlease choose k=[1,2,3]")