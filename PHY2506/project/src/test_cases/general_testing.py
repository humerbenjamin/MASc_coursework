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
from src.source_functions.source_functions import gaussian_x_multiplied_by_u

from src.viscous_burgers_solver.transient_solver import make_data_save_times


############################################################
# Testing of minor functionality and submodules
############################################################

# burger's solver
############################################################
print(make_data_save_times(1.35,0.25))

