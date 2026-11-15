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
from time import time


# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.transient_solver import transient_solve
from src.data_assimilation.generate_DA_data import make_H_matrix
from src.data_assimilation.adjoint_variable_calculation import calc_terminal_adjoint_variables, calc_non_terminal_adjoint_variables
from src.data_assimilation.calculate_jacobians import calc_dM_dU, calc_dtR_dk
from src.data_assimilation.cost_function import cost_function_with_parameter_regularization, backtracking_line_search



############################################################
############################################################
# main function for 4Dvar of transient solutions
############################################################
############################################################

# 4Dvar Function
############################################################
def variational_DA_transient(
                                data_loc,                                                               # location of data to be read in
                                source_function, source_parameters_initial,                             # source functions
                                source_function_derivative_u, source_function_derivative_k,             # source function derivatives
                                parameter_cov_matrix, observation_cov_matrix,                           # covariance matrices
                                C, kin_viscosity, BCs, verbose                                          # 
                            ):
    '''
    data_loc="data/test"
    '''

    # load data
    df = pd.read_csv(f"{data_loc}_rawdata.csv")
    df_ic = pd.read_csv(f"{data_loc}_rawdata_IC.csv")
    df_resid = pd.read_csv(f"{data_loc}_rawdata_resid.csv")
    df_noisy = pd.read_csv(f"{data_loc}_measurements_noisy.csv")

    # set up initial conditions for solve
        # get sorted data
    sorted_data_times = np.sort(np.unique(df.t))
        # get initial condition vectors and final time
    u0 = df_ic.u
    x = df_ic.x
    t_f = sorted_data_times[-1]
        # get source parameters as numpy array
    k0 = np.array(source_parameters_initial)
    k = k0.copy()
        # get data save timings
    data_save_interval = sorted_data_times[1] - sorted_data_times[0]
        # set up counter for solve number
    solve_number = 1
        # set up whether time updates are all included
    printout_solve_time = verbose

    # load in observation mapping operator
    H, measurement_indices = make_H_matrix(df_noisy[df_noisy.t == 0].measurement_locations, df_ic.x)
    HT = H.T.tocsr()

    # set up conditions for each solve within loop

    for i in range(1):
        # do the forward transient solve
        df, df_ic, df_resid, df_alldata = transient_solve(
                                                            u0, x, t_f,
                                                            S_fcn=source_function, source_params=k,
                                                            C=C, kin_viscosity=kin_viscosity, BCs=BCs,
                                                            verbose=verbose,
                                                            data_save_interval=data_save_interval,
                                                            printout_solve_time=True, solve_number=solve_number,
                                                            save_alldata=True
                                                         )

        

        # plot the forward transient run if required
        plot_transient_solve_single_run_forward_plot(df, df_ic, df_resid)


        # do the inverse solve
        # FIX THIS FIRST
        inverse_problem_solve()


        print(accumulated_parameter_gradient)

        print(k, k_new)
        
        print(len(reverse_sorted_times), print(len(adjoint_vars_all_times)))

        plt.plot(df_ic.x, adjoint_vars)
        plt.show()


        # run the inverse problem

        

        

    return



############################################################
############################################################
# subfunctions
############################################################
############################################################

# function for inverse solve
############################################################
def inverse_problem_solve():
    # precalculate data that is required for the inverse problem
    reverse_sorted_times = np.sort(np.unique(df_alldata.t))[::-1]
    obs_cov_inverse = np.eye(len(df_noisy[df_noisy.t == 0].u_noisy))
    x = df_alldata[df_alldata.t == 0].x
    dx = x[1] - x[0]

    # set up variables for the accumulated parameter gradient
    accumulated_parameter_gradient = 0

    # calculate the adjoint variables and then store them all for computing the gradient later
    adjoint_vars_all_times = []
    for time_ind in range(len(reverse_sorted_times) - 1):
        # get time information
        time = reverse_sorted_times[time_ind]
        dt = reverse_sorted_times[time_ind] - reverse_sorted_times[time_ind + 1]

        # separate arrays at this specific time from the dfs
        df_singletime = df_alldata[np.isclose(df_alldata.t, reverse_sorted_times[time_ind], atol=1e-10)]
        df_obs_singletime = df_noisy[np.isclose(df_noisy.t, reverse_sorted_times[time_ind], atol=1e-10)]
    
        # print(time_ind, reverse_sorted_times[time_ind], len(df_obs_singletime), len(df_singletime), dt)

        # use the previous adjoint variables to calculate the accumulated parameter gradient
            # calculate the residual jacobian wrt k
        dtR_dk = calc_dtR_dk(    
                                df_singletime.u.to_numpy(), df_singletime.u_tilde.to_numpy(), df_singletime.x.to_numpy(), 
                                dt, kin_viscosity, dx, source_function_derivative_k, source_function_derivative_u, k
                            )
            # calculate the accumulated gradient
        if time_ind != 0:
            accumulated_parameter_gradient += dtR_dk.T @ adjoint_vars

        # calculate the adjoint variables
            # initial time case
        if time == reverse_sorted_times[0]:
            # handle the case containing observations
            if len(df_obs_singletime) > 0:
                adjoint_vars = calc_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), obs_cov_inverse, True)
                adjoint_vars_all_times.append(adjoint_vars)
            
            # handle the case which does not contain observations
            else:
                adjoint_vars = calc_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), obs_cov_inverse, False)
                adjoint_vars_all_times.append(adjoint_vars)
        
            # all other times in this case
        else:
            # calculate the model jacobian transpose
            model_jacobian_transpose = calc_dM_dU(df_singletime.u.to_numpy(), df_singletime.u_tilde.to_numpy(), df_singletime.x.to_numpy(), kin_viscosity, dx, dt, source_function_derivative_u, k).T

            # handle the case containing observations
            if len(df_obs_singletime) > 0:
                adjoint_vars = calc_non_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), obs_cov_inverse, model_jacobian_transpose, adjoint_vars, True)  
                adjoint_vars_all_times.append(adjoint_vars)

            # handle the case which does not contain observations
            else:
                adjoint_vars = calc_non_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), obs_cov_inverse, model_jacobian_transpose, adjoint_vars, False)  
                adjoint_vars_all_times.append(adjoint_vars)


    # filter dataframes for t=0
    df_singletime = df_alldata[np.isclose(df_alldata.t, reverse_sorted_times[time_ind+1], atol=1e-10)]
    df_obs_singletime = df_noisy[np.isclose(df_noisy.t, reverse_sorted_times[time_ind+1], atol=1e-10)]

    # last addition to the accumulated parameter gradient
    dtR_dk = calc_dtR_dk(    
                    df_singletime.u.to_numpy(), df_singletime.u_tilde.to_numpy(), df_singletime.x.to_numpy(), 
                    dt, kin_viscosity, dx, source_function_derivative_k, source_function_derivative_u, k
                )
    accumulated_parameter_gradient += parameter_cov_matrix @ (k-k0) + dtR_dk.T @ adjoint_vars

    # calculate what the parameter should be for updating the parameters
    k_new, alpha, J_new = backtracking_line_search(
                                                        k, k0, accumulated_parameter_gradient,
                                                        cost_function_with_parameter_regularization,
                                                        df_alldata, df_noisy, H,
                                                        observation_cov_matrix, parameter_cov_matrix
                                                    )


    return


############################################################
############################################################
# plotting functions
############################################################
############################################################

############################################################
# plot the transient solve data for a single forward run
############################################################
def plot_transient_solve_single_run_forward_plot(df, df_ic, df_resid):
    # get the figure and the axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # get data required for the plotting loop
    times = np.unique(df.t)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=times.min(), vmax=times.max())

    # add plot for each time that data was saved
    for t in times:
        df_t = df[df.t == t]
        ax1.plot(df_t.x, df_t.u, color=cmap(norm(t)))

    # set up axis labels
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$u$")
    ax1.set_title(r"$u(x)$ for a Developing Shock Wave Modeled"+"\n"+"Using Inviscid Burger's Equation")

    # Right plot
    ax2.plot(df_resid.n, df_resid.R_norm)
    ax2.set_ylabel(r"$R$")
    ax2.set_xlabel(r"$N_\mathrm{iter}$")
    ax2.set_title("Residual Over Number of Iterations")
    ax2.set_xscale('linear')
    ax2.set_yscale('log')

    # Add colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax1, label=r"$t$")

    # fix layout and show plot
    plt.tight_layout()
    plt.show()

    return