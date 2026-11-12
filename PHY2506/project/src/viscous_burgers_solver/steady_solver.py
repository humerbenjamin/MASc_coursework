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
# main function for steady solve of IBVP
############################################################
def steady_solve(u0, x, S_fcn, source_params=[], n_iter=1000, C=0.1, kin_viscosity=1.5e-5, BCs=[0,0], verbose=True, convergence_threshold=1e-6):
    # make matrices and vectors for finite difference
    A = make_A(u0)
    b = make_b(u0, BCs)
    t_c = 0
    dx = np.abs(x[1]-x[0])

    # make initial state vectors for solver
    u_old = u0.copy()
    S = S_fcn(u_old, x, t_c, source_params)

    # iterate until convergence or num_iter is reached
    n = 0
    converged = False

    df_ic = pd.DataFrame({"u":u0, "x":x, "t":np.ones(len(u0))*t_c})
    
    # iterate until an exit condition is met
    while (n < n_iter) and (not converged):
        # calculate timestep
        dt = calc_timestep(u_old, BCs, kin_viscosity, dx, C)

        # ssp-rk2 algorithm
        u = ssp_rk2(u_old, x, S, t_c, A, b, BCs, dt, kin_viscosity)

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

        # check convergence criteria
        if R_norm < convergence_threshold:
            converged = True

        # plot verbose diagnostics
        if verbose:
            print(
                f"n = {n:.6e}, " f"t = {t_c:.6e}, " f"R = {R}, " f"R_norm = {R_norm}, " f"dt = {dt:.6e}"
            )

        # update to next timestep
        t_c += dt
        S = S_fcn(u, x, t_c, source_params)
        n += 1

        # update the previous solution vector u
        u_old = u.copy()

    df = pd.DataFrame({"u":u, "x":x, "t":np.ones(len(u))*t_c})
    df_resid = pd.DataFrame({"n":n_array,"t":t_array, "R":R_array, "R_norm":R_norm_array})

    if converged:
        print("\nConverged")
    else:
        print(f"\nRan out of iterations before convergence.\n\tLast R_norm={R_norm}")


    return df, df_ic, df_resid
