# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
    # sublibraries

# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.semi_discrete_solver import L, make_A, make_b
from src.viscous_burgers_solver.ssp_rk2 import ssp_rk2, calc_timestep


############################################################
# main function for transient solve of IBVP
############################################################
def transient_solve(u0, x, t_f, S_fcn, C=0.1, kin_viscosity=1.5e-5, BCs=[0,0], verbose=True, data_save_interval=0.1):
    # make matrices and vectors for finite difference
    A = make_A(u0)
    b = make_b(u0, BCs)
    t_c = 0
    dx = np.abs(x[1]-x[0])

    # make initial state vectors for solver
    u = u0
    S = S_fcn(u)

    # make dataframe for storing simulation data
    df = pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})
    t_saving = data_save_interval


    # iterate until end time t_f

    

    while t_c < t_f:
        # calculate timestep
        dt = min(calc_timestep(u, BCs, kin_viscosity, dx, C), t_f - t_c)
        if verbose:
            print(
                f"t = {t_c:.6e}, " f"min(u) = {np.min(u):.6e}, " f"max(u) = {np.max(u):.6e}, " f"dt = {dt:.6e}"
            )

        # save data if timestep condition is met
        if t_c > t_saving:
            df = pd.concat([df, pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})])
            t_saving += data_save_interval


        # ssp-rk2 algorithm
        u = ssp_rk2(u, x, S, t_c, A, b, BCs, dt, kin_viscosity)
        
        # update to next timestep
        t_c += dt
        S = S_fcn(u)

    # check if there has been a save recently
    if (t_c - np.max(df.t)) > data_save_interval / 2:
        df = pd.concat([df, pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})])


    return df