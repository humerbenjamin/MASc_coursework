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
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import inv as sparse_inv


# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.transient_solver import transient_solve
from src.data_assimilation.generate_DA_data import make_H_matrix
from src.data_assimilation.generate_DA_data import make_transient_plot as make_transient_plot_with_data
from src.data_assimilation.adjoint_variable_calculation import calc_terminal_adjoint_variables, calc_non_terminal_adjoint_variables
from src.data_assimilation.calculate_jacobians import calc_dM_dU, calc_dtR_dk
from src.data_assimilation.cost_function import cost_function_with_parameter_regularization, backtracking_line_search, update_bfgs_inverse_hessian, calculate_bfgs_step, calculate_gauss_newton_update
from src.data_assimilation.calc_covariance_matrices import diagonal_obs_cov_matrix



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
                                parameter_cov_matrix, observation_cov_function,                         # covariance matrices
                                C, kin_viscosity, BCs,                                                  # physical and solver parameters
                                num_VDA_iterations=10,
                                verbose=False,                                                          # 

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
    k_prev = k0.copy()
    k = k0.copy()
        # get data save timings
    data_save_interval = sorted_data_times[1] - sorted_data_times[0]
        # set up whether time updates are all included
    printout_solve_time = verbose

    # load in observation mapping operator
    H, measurement_indices = make_H_matrix(df_noisy[df_noisy.t == 0].measurement_locations, df_ic.x)
    HT = H.T.tocsr()

    # set up array for k_history
    k_history = [k0]

    for i in range(num_VDA_iterations):
        # set up solve number
        solve_number=i+1
        # do the forward transient solve
        df, df_ic, df_resid, df_alldata, G = transient_solve(
                                                                u0, x, t_f,
                                                                S_fcn=source_function, source_params=k,
                                                                C=C, kin_viscosity=kin_viscosity, BCs=BCs,
                                                                verbose=False,
                                                                data_save_interval=data_save_interval,
                                                                printout_solve_time=printout_solve_time, solve_number=solve_number,
                                                                save_alldata=True,
                                                                calculate_gauss_newton_matrix=True,
                                                                S_fcn_dU=source_function_derivative_u, S_fcn_dk=source_function_derivative_k,
                                                                obs_cov_function=observation_cov_function, param_cov_matrix=parameter_cov_matrix,
                                                                H_matrix=H, df_noisy=df_noisy
                                                            )

        
        # plot the forward transient run if required
        # plot_transient_solve_single_run_forward_plot(df, df_ic, df_resid)
        # make_transient_plot_with_data(df, df_ic, df_resid, df_noisy, BCs)


        # do the inverse solve
        accumulated_parameter_gradient = inverse_problem_solve(
                                                                        df_alldata, df_noisy, kin_viscosity,
                                                                        source_function_derivative_k, source_function_derivative_u,
                                                                        k, k0,
                                                                        H, HT,
                                                                        parameter_cov_matrix, observation_cov_function,
                                                                        printout_solve_time=printout_solve_time, solve_number=solve_number
                                                              )

        # set up values for next forward solve before updating k
        k_prev = k.copy()

        # update k
        dk = calculate_gauss_newton_update(G, accumulated_parameter_gradient)
        k = k + dk

        print("\t\tAccumulated parameter gradient:",accumulated_parameter_gradient)
        print("\t\tk:",k_prev, "k0:", k)

        k_history.append(k)

    # Save parameter history
        # reshape the array
    k_history = np.asarray(k_history).T
    n_parameters = k_history.shape[0]
        # make dataframe
    df_k_history = pd.DataFrame(k_history.T, columns=[f"k_{i+1}" for i in range(n_parameters)])
    df_k_history.index.name = "Iteration"
    df_k_history.to_csv(f"{data_loc}_k_history.csv", index=True)

    plot_parameter_history(k_history)

    return



############################################################
############################################################
# subfunctions
############################################################
############################################################

# function for inverse solve
############################################################
def inverse_problem_solve(
                                df_alldata, df_noisy, 
                                kin_viscosity, 
                                source_function_derivative_k, source_function_derivative_u,         # source function derivatives
                                k, k0,
                                H, HT,
                                parameter_cov_matrix, observation_cov_function,
                                printout_solve_time=False, solve_number=1
                          ):
    # get starting time for inverse solve
    t_i_inv_solve = time()
    
    # precalculate data that is required for the inverse problem
    reverse_sorted_times = np.sort(np.unique(df_alldata.t))[::-1]
    x = df_alldata[df_alldata.t == 0].x
    dx = x[1] - x[0]

    # set up variables for the accumulated parameter gradient
    accumulated_parameter_gradient = 0

    # calculate the adjoint variables and then store them all for computing the gradient later
    adjoint_vars_all_times = []
    for time_ind in range(len(reverse_sorted_times) - 1):
        # get time information
        t = reverse_sorted_times[time_ind]
        dt = reverse_sorted_times[time_ind] - reverse_sorted_times[time_ind + 1]

        # separate arrays at this specific time from the dfs
        df_singletime = df_alldata[np.isclose(df_alldata.t, reverse_sorted_times[time_ind], atol=1e-10)]
        df_obs_singletime = df_noisy[np.isclose(df_noisy.t, reverse_sorted_times[time_ind], atol=1e-10)]
    
        # print(time_ind, reverse_sorted_times[time_ind], len(df_obs_singletime), len(df_singletime), dt)

        # use the previous adjoint variables to calculate the accumulated parameter gradient
        if time_ind != 0:
            # calculate the residual jacobian wrt k
            dtR_dk = calc_dtR_dk(    
                                    df_singletime.u.to_numpy(), df_singletime.u_tilde.to_numpy(), df_singletime.x.to_numpy(), 
                                    dt_prev, kin_viscosity, dx, source_function_derivative_k, source_function_derivative_u, k
                                )
            # calculate the accumulated gradient
            accumulated_parameter_gradient += dtR_dk.T @ adjoint_vars

        # calculate the adjoint variables
            # initial time case
        if time_ind == 0:
            # handle the case containing observations
            if len(df_obs_singletime) > 0:
                adjoint_vars = calc_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), df_obs_singletime, dt, True)
                adjoint_vars_all_times.append(adjoint_vars)
            
            # handle the case which does not contain observations
            else:
                adjoint_vars = calc_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), df_obs_singletime, dt, False)
                adjoint_vars_all_times.append(adjoint_vars)

            # all other times in this case
        else:
            # calculate the model jacobian transpose
            model_jacobian_transpose = calc_dM_dU(df_singletime.u.to_numpy(), df_singletime.u_tilde.to_numpy(), df_singletime.x.to_numpy(), kin_viscosity, dx, dt_prev, source_function_derivative_u, k).T

            # handle the case containing observations
            if len(df_obs_singletime) > 0:
                adjoint_vars = calc_non_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), df_obs_singletime, model_jacobian_transpose, adjoint_vars, dt, dt_prev, True)  
                adjoint_vars_all_times.append(adjoint_vars)

            # handle the case which does not contain observations
            else:
                adjoint_vars = calc_non_terminal_adjoint_variables(H, HT, df_singletime.u.to_numpy(), df_obs_singletime.u_noisy.to_numpy(), df_obs_singletime, model_jacobian_transpose, adjoint_vars, dt, dt_prev, False)  
                adjoint_vars_all_times.append(adjoint_vars)


        # save previous timestep
        dt_prev = dt

    # filter dataframes for t=0
    df_singletime = df_alldata[np.isclose(df_alldata.t, reverse_sorted_times[time_ind+1], atol=1e-10)]
    df_obs_singletime = df_noisy[np.isclose(df_noisy.t, reverse_sorted_times[time_ind+1], atol=1e-10)]


    # last addition to the accumulated parameter gradient
    dtR_dk = calc_dtR_dk(    
                    df_singletime.u.to_numpy(), df_singletime.u_tilde.to_numpy(), df_singletime.x.to_numpy(), 
                    dt, kin_viscosity, dx, source_function_derivative_k, source_function_derivative_u, k
                )




    if issparse(parameter_cov_matrix):
        parameter_cov_matrix_inv = sparse_inv(parameter_cov_matrix)
    else:
        parameter_cov_matrix_inv = linalg.inv(parameter_cov_matrix)
    
    accumulated_parameter_gradient += parameter_cov_matrix_inv @ (k-k0) + dtR_dk.T @ adjoint_vars


    # if required, printout the solving time for the inverse problem
    if printout_solve_time:
        print(f"\tInverse solve {solve_number} complete in {(time() - t_i_inv_solve):.2f} seconds")


    return accumulated_parameter_gradient


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


############################################################
# plot the transient solve data for a single forward run
############################################################
def plot_parameter_history(k_history):

    for i in range(len(k_history)):
        plt.plot(k_history[i], label=f"Parameter {i}")


    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()


    return