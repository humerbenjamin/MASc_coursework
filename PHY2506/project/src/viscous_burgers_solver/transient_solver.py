# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
    # sublibraries
from time import time
from scipy.sparse import diags, linalg

# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.semi_discrete_solver import make_A, make_b
from src.viscous_burgers_solver.ssp_rk2 import ssp_rk2, calc_timestep
from src.data_assimilation.calculate_jacobians import calc_dM_dU, calc_dtR_dk
from src.data_assimilation.calc_covariance_matrices import diagonal_obs_cov_matrix


############################################################
# main function for transient solve of IBVP
############################################################
def transient_solve(
                        u0, x, t_f,
                        S_fcn, source_params=[],
                        C=0.1, kin_viscosity=1.5e-5, BCs=[0,0],
                        verbose=True, data_save_interval=0.1, printout_solve_time=False, solve_number=1, save_alldata=False,
                        calculate_gauss_newton_matrix=False, S_fcn_dU=None, S_fcn_dk=None,
                        obs_cov_function=None, param_cov_matrix=None, H_matrix=None, df_noisy=None
                   ):
    t_start = time()

    # make matrices and vectors for finite difference
    A = make_A(u0)
    b = make_b(u0, BCs)
    t_c = 0
    dx = np.abs(x[1]-x[0])

    # make initial state vectors for solver
    u = np.array(u0.copy())
    u_old = np.array(u.copy())
    S = S_fcn(u, x, source_params)

    # make df for initial condition passed to solver
    df_ic = pd.DataFrame({"u":u0.copy(), "x":x, "t":np.ones(len(u0))*t_c})

    # make dataframe for storing simulation data
    # df = pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})
    onevector = np.ones(len(u))
    u_data, x_data, t_data = [u.copy()], x.copy(), [(onevector*t_c).copy()]

    # make dataframe for saving all data in case the inverse problem needs to be calculated
    # df_alldata = pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})
    u_alldata, u_tilde_alldata, t_alldata = [u.copy()], [], [(onevector*t_c).copy()]

    # make array of times when data is being saved
    data_save_times = make_data_save_times(t_f, data_save_interval)

    # initialize gauss-newton matrix
    if calculate_gauss_newton_matrix:
        sens_matrix = np.zeros([len(u), len(source_params)])
        G = param_cov_matrix.copy()
        H = H_matrix.copy()


    # iterate until end time t_f
    n = 0
    while t_c < t_f:
        # calculate timestep
        dt, is_saving_step, cur_save_time = calc_timestep(u, BCs, kin_viscosity, dx, C, t_c, data_save_times)

        # calculate whether this would correspond to a data saving timestep


        if verbose:
            print(
                f"t = {t_c:.6e}, " f"min(u) = {np.min(u):.6e}, " f"max(u) = {np.max(u):.6e}, " f"dt = {dt:.6e}"
            )

        # ssp-rk2 algorithm
        u, u_tilde = ssp_rk2(u, x, S, t_c, A, b, BCs, dt, kin_viscosity)

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


        # calculate gauss newton matrix
        if calculate_gauss_newton_matrix:
            # get jacobians
            dM_dU = calc_dM_dU(u_old, u_tilde, x, kin_viscosity, dx, dt, S_fcn_dU, source_params)
            dtR_dk = calc_dtR_dk(u_old, u_tilde, x, dt, kin_viscosity, dx, S_fcn_dk, S_fcn_dU, source_params)

            # calculate matrices
            sens_matrix = dM_dU @ sens_matrix - dt * dtR_dk

            # contribute to G on data saving steps
            if is_saving_step:
                df_noisy_singletime = df_noisy[np.isclose(df_noisy.t, cur_save_time, atol=1e-10)]
                obs_cov_matrix = diagonal_obs_cov_matrix(df_noisy_singletime)
                obs_cov_matrix_inv = linalg.inv(obs_cov_matrix)
                observation_sensitivity = H @ sens_matrix
                G += ( observation_sensitivity.T @ obs_cov_matrix_inv @ observation_sensitivity )


        # update to next timestep
        t_c += dt
        S = S_fcn(u, x, source_params)
        n += 1
        u_old = u.copy()


        # Save scheduled data
        if is_saving_step:
            u_data.append(u.copy())
            t_data.append((onevector*cur_save_time).copy())

        # save data
        if save_alldata:
            u_tilde_alldata.append(u_tilde.copy())
            u_alldata.append(u.copy())

            if is_saving_step:
                t_alldata.append((onevector*cur_save_time).copy())
            else:
                t_alldata.append((onevector*t_c).copy())


    u_tilde_alldata.append(np.full(len(u), np.nan))


    # make dataframes
    df_resid = pd.DataFrame({"n":n_array,"t":t_array, "R":R_array, "R_norm":R_norm_array})
    df = pd.DataFrame({"u":np.concatenate(u_data), "x":np.tile(x, len(u_data)), "t":np.concatenate(t_data)})
    df_alldata = pd.DataFrame({"u":np.concatenate(u_alldata), "u_tilde":np.concatenate(u_tilde_alldata), "x":np.tile(x, len(u_alldata)), "t":np.concatenate(t_alldata)})

    # printout elapsed time if relevant
    if printout_solve_time:
        print(f"Forward solve {solve_number} complete in {(time() - t_start):.2f} seconds.")


    if calculate_gauss_newton_matrix:
        return df, df_ic, df_resid, df_alldata, G

    else:
        return df, df_ic, df_resid, df_alldata



############################################################
# helper function to make data saving times
############################################################
def make_data_save_times(t_f, data_save_interval):
    i = 1
    data_save_times = [0]
    while t_f > data_save_times[-1] + data_save_interval:
        data_save_times.append(data_save_interval*i)
        i += 1

    data_save_times.append(round(t_f, 4))

    return np.array(data_save_times)