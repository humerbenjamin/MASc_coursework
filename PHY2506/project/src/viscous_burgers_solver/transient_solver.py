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
def transient_solve(u0, x, t_f, S_fcn, source_params=[], C=0.1, kin_viscosity=1.5e-5, BCs=[0,0], verbose=True, data_save_interval=0.1, strict_data_saving=False):
    # make matrices and vectors for finite difference
    A = make_A(u0)
    b = make_b(u0, BCs)
    t_c = 0
    dx = np.abs(x[1]-x[0])

    # make initial state vectors for solver
    u = u0.copy()
    u_old = u.copy()
    S = S_fcn(u, x, t_c, source_params)

    # make df for initial condition passed to solver
    df_ic = pd.DataFrame({"u":u0, "x":x, "t":np.ones(len(u0))*t_c})

    # make dataframe for storing simulation data
    df = pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})
    t_saving = data_save_interval

    # make array of times when data is being saved
    data_save_times = make_data_save_times(t_f, data_save_interval)

    # iterate until end time t_f
    n = 0
    while t_c < t_f:
        # calculate timestep
        dt, is_saving_step, cur_save_time = calc_timestep(u, BCs, kin_viscosity, dx, C, t_c, data_save_times, strict_data_saving)

        # calculate whether this would correspond to a data saving timestep


        if verbose:
            print(
                f"t = {t_c:.6e}, " f"min(u) = {np.min(u):.6e}, " f"max(u) = {np.max(u):.6e}, " f"dt = {dt:.6e}"
            )

        # save data if timestep condition is met
        if is_saving_step:
            df = pd.concat([df, pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*cur_save_time})])
            t_saving += data_save_interval


        # ssp-rk2 algorithm
        u = ssp_rk2(u, x, S, t_c, A, b, BCs, dt, kin_viscosity)

                # calculate residual
        R = np.sum(np.abs(u - u_old))
        if n == 0:
            R_baseline = R
            R_norm = 1
        else:
            R_norm = R/R_baseline

        # save data for residual convergence plot
        if n == 0:
            n_array = [n]
            t_array = [t_c]
            R_array = [R]
            R_norm_array = [R_norm]
        else:
            n_array.append(n)
            t_array.append(t_c)
            R_array.append(R)
            R_norm_array.append(R_norm)
        
        # update to next timestep
        t_c += dt
        S = S_fcn(u, x, t_c, source_params)
        n += 1
        u_old = u.copy()

    df_resid = pd.DataFrame({"n":n_array,"t":t_array, "R":R_array, "R_norm":R_norm_array})


    # check if there has been a save recently
    if (t_c - np.max(df.t)) > data_save_interval / 2:
        df = pd.concat([df, pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})])


    return df, df_ic, df_resid



############################################################
# helper function to make data saving times
############################################################
def make_data_save_times(t_f, data_save_interval):
    i = 1
    data_save_times = [0]
    while t_f > data_save_times[-1] + data_save_interval:
        data_save_times.append(data_save_interval*i)
        i += 1

    data_save_times.append(t_f)

    return np.array(data_save_times)