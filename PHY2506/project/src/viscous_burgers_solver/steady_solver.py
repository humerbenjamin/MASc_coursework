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
# main function for steady solve of the IBVP
############################################################
def steady_solve(u0, x, t_f, S, 
                    C=0.1, kin_viscosity=1.5e-5, BCs=[0,0],
                    verbose=True, data_save_interval=0.1, residual_threshold=1e-6
                ):
    
    # make matrices and vectors for finite difference
    A = make_A(u0)
    b = make_b(u0, BCs)
    t_c = 0
    dx = np.abs(x[1]-x[0])

    # make initial state vectors for solver
    u_prev = u0.copy()
    u = u0.copy

    # make dataframe for storing simulation data
    df = pd.DataFrame({"u":u0, "x":x, "t":np.ones(len(u0))*t_c})
    t_saving = data_save_interval

    # iterate until end time t_f

    
    timestep_index = 0
    while t_c < t_f:
        # calculate timestep
        dt = min(calc_timestep(u_prev, kin_viscosity, dx, C), t_f - t_c)
        if verbose:
            print(
                f"t = {t_c:.6e}, "
                f"min(u) = {np.min(u_prev):.6e}, "
                f"max(u) = {np.max(u_prev):.6e}, "
                f"dt = {dt:.6e}"
            )

        # save data if timestep condition is met
        if t_c > t_saving:
            df = pd.concat([df, pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})])
            t_saving += data_save_interval

        # ssp-rk2 algorithm application
        u = ssp_rk2(u_prev, x, S, t_c, A, b, dt, kin_viscosity)
        
        # update to next timestep and calculate residual
        t_c += dt
        R = np.sum(np.abs(u-u_prev))

        # initialize first residual
        if timestep_index == 0:
            first_residual = R
            df_resid = pd.DataFrame({"R":[R], "R_norm":[1], "t":[t_c], "n":[timestep_index+1]})
        
        else:
            R_norm = R / first_residual
            df_resid.loc[len(df_resid)] = [R, R_norm, t_c, timestep_index+1]
            # check convergence
            if R_norm < residual_threshold:
                return df, df_resid

        # increase timestep index and update what is defined as the previous timestep
        timestep_index += 1
        u_prev = u.copy()



    # check if there has been a save recently
    if (t_c - np.max(df.t)) > data_save_interval / 2:
        df = pd.concat([df, pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})])


    return df, df_resid