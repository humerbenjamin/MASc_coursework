# add the src header to pythonpath
############################################################
import sys
sys.path.append(".")


# import standard modules
############################################################
    # full libraries
import numpy as np
import matplotlib.pyplot as plt

# import personal files
############################################################
    # specific functions
from src.data_assimilation.generate_DA_data import generate_data

# direct solver inputs
n_points=500
x = np.linspace(0,1,n_points)
u = np.zeros(len(x)) # np.append(1*np.ones(int(n_points/2)), -1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1
def S(u):
    return 0 # sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x)) #sp.stats.norm.pdf(x)

# system parameters
kin_viscosity = 0
BCs = [1,-1]

# solver parameters
n_iter = 100000
C = 0.1
convergence_threshold = 1e-6

# type of solve (steady or unsteady)
solve_type='steady'

# measurement locations
measurement_locations = np.linspace(0,1,20)

generate_data(  solve_type, u, x, S,                                # direct functional inputs
                kin_viscosity, BCs,                                 # additional system parameters
                n_iter, C, convergence_threshold,                   # solver parameters
                verbose=False, plot_values=True,
                measurement_locations=measurement_locations
             )