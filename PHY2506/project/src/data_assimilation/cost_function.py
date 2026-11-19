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
from src.data_assimilation.calculate_jacobians import calc_dM_dU, calc_dtR_dk
from src.viscous_burgers_solver.semi_discrete_solver import L


############################################################
# implement standard cost function
############################################################
def cost_function_with_parameter_regularization(df_model, df_obs, H, params, params_b, obs_cov, params_cov):

    # get list of times that contribute to DA
    unique_times = np.sort(np.unique(df_obs.t))

    meas_delta_cost = 0
    for t_ind in range(len(unique_times)):
        u = df_model[np.isclose(df_model.t, unique_times[t_ind], atol=1e-10)].u
        obs = df_obs[np.isclose(df_obs.t, unique_times[t_ind], atol=1e-10)].u_noisy

        meas_delta_cost += (H@u - obs).T @ obs_cov @ (H@u - obs)

    meas_delta_cost *= 1/2

    param_diff_cost = 1/2 * (params - params_b).T @ params_cov @ (params - params_b)

    total_cost = meas_delta_cost + param_diff_cost

    return total_cost


############################################################
# optioons for different function to determine the step size
############################################################

# backtracking line search
############################################################
def backtracking_line_search(
        k, k0, gradient,
        cost_function,
        df_model, df_obs, H, obs_cov, params_cov,
        alpha_initial=1.0, beta=0.5, c=1e-4,
        max_iterations=500
    ):

    alpha = alpha_initial

    J_current = cost_function(
        df_model, df_obs, H, k, k0, obs_cov, params_cov
    )

    gradient_norm_squared = np.dot(gradient, gradient)

    for i in range(max_iterations):

        k_trial = k - alpha * gradient

        J_trial = cost_function(
            df_model, df_obs, H, k_trial, k0, obs_cov, params_cov
        )

        if J_trial <= J_current - c * alpha * gradient_norm_squared:
            print(i)
            return k_trial, alpha, J_trial

        alpha *= beta

    raise RuntimeError("Line search failed to find an acceptable step.")


# bfgs step
############################################################
def calculate_bfgs_step(
                k, gradient,
                inverse_hessian,
                alpha_initial=1.0,
                alpha_max=1e-5
                ):

    # calculate BFGS search direction
    search_direction = -inverse_hessian @ gradient

    # ensure descent direction
    if np.dot(gradient, search_direction) >= 0:
        inverse_hessian = np.eye(len(k))
        search_direction = -gradient

    # calculate approximate Hessian
    hessian = np.linalg.solve(inverse_hessian, np.eye(len(k)))

    numerator = -np.dot(gradient, search_direction)
    denominator = search_direction @ hessian @ search_direction

    if denominator > 0:
        alpha = numerator / denominator
    else:
        alpha = alpha_initial

    # impose upper bound
    alpha = min(alpha, alpha_max)

    return search_direction, alpha, inverse_hessian


# hessian update for bfgs
def update_bfgs_inverse_hessian(
                inverse_hessian,
                k_old, k_new,
                gradient_old, gradient_new
                ):

    # calculate parameter and gradient differences
    s = k_new - k_old
    y = gradient_new - gradient_old

    # calculate curvature
    sy = np.dot(s, y)

    # check curvature condition
    if sy <= 1e-12:
        return inverse_hessian

    rho = 1.0 / sy

    # identity matrix
    I = np.eye(len(k_old))

    # BFGS inverse-Hessian update
    V = I - rho * np.outer(s, y)

    inverse_hessian = (
        V @ inverse_hessian @ V.T
        + rho * np.outer(s, s)
    )

    return inverse_hessian


# gauss newton matrix
############################################################
def calculate_gauss_newton_matrix(
                df_model, df_obs,
                x, dt, nu,
                H,
                obs_cov_inverse,
                params_cov_inverse,
                source_function,
                source_derivative_function_u,
                source_derivative_function_k,
                source_parameters,
                A, b, BCs, dx
                ):

    # initialise parameter sensitivity
    S = np.zeros((len(x), len(source_parameters)))

    # initialise Gauss-Newton matrix
    G = params_cov_inverse.copy()

    # retrieve unique observation times
    unique_times = np.sort(np.unique(df_obs.t))

    # loop through observation times
    for t_ind, time in enumerate(unique_times):

        # retrieve model solution at observation time
        df_singletime = df_model[
            np.isclose(df_model.t, time, atol=1e-10)
        ]

        u = df_singletime.u.to_numpy()

        # accumulate observation contribution
        if t_ind > 0:

            # retrieve previous model solution
            previous_time = unique_times[t_ind - 1]

            df_previous = df_model[
                np.isclose(df_model.t, previous_time, atol=1e-10)
            ]

            u_previous = df_previous.u.to_numpy()

            # calculate intermediate RK2 solution
            # NOTE: this must use the same implementation as your forward solver
            u_tilde = u_previous + dt * L(
                u_previous, x, source_function,
                previous_time, A, b, BCs, nu
            )

            # calculate RK2 model Jacobian
            dM_dU = calc_dM_dU(
                u_previous, u_tilde, x, nu, dx, dt,
                source_derivative_function_u,
                source_parameters
            )

            # calculate RK2 residual parameter Jacobian
            dtR_dk = calc_dtR_dk(
                u_previous, u_tilde, x, nu, dx, dt,
                source_derivative_function_u,
                source_derivative_function_k,
                source_parameters
            )

            # propagate parameter sensitivities
            S = dM_dU @ S - dt * dtR_dk

        # calculate observation sensitivity
        observation_sensitivity = H @ S

        # calculate Gauss-Newton contribution
        G += (
            observation_sensitivity.T
            @ obs_cov_inverse
            @ observation_sensitivity
        )

    return G



def calculate_gauss_newton_update(
                G,
                accumulated_parameter_gradient,
                damping=0.0
                ):

    # calculate damped Gauss-Newton matrix
    G_damped = G + damping * np.eye(G.shape[0])

    # calculate parameter update
    delta_k = np.linalg.solve(
        G_damped,
        -accumulated_parameter_gradient
    )

    return delta_k