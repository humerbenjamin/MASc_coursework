# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
    # sublibraries
from scipy.sparse import diags


# import personal files
############################################################
    # specific functions


############################################################
############################################################
# main jacobian calculation functions
############################################################
############################################################

############################################################
# calculation of overall model jacobians
############################################################

# calculation of model jacobian with respect to u
############################################################

def calc_dM_dU(
                    u, u_tilde, x,                            # original and intermediate RK2 solution vectors
                    nu, dx, dt,
                    source_derivative_function_u, source_parameters
              ):

    # make arrays for pentadiagonal matrix
    upper2 = np.zeros(len(u) - 2)
    upper1 = np.zeros(len(u) - 1)
    middle = np.zeros(len(u))
    lower1 = np.zeros(len(u) - 1)
    lower2 = np.zeros(len(u) - 2)

    # fill arrays with proper values for pentadiagonal matrix
    for ind in range(len(u)):

        middle[ind] = calc_dM_du_singlepoint(
            u, u_tilde, x, ind, ind, nu, dx, dt,
            source_derivative_function_u, source_parameters
        )

        if ind < len(u) - 1:
            upper1[ind] = calc_dM_du_singlepoint(
                u, u_tilde, x, ind, ind+1, nu, dx, dt,
                source_derivative_function_u, source_parameters
            )

        if ind < len(u) - 2:
            upper2[ind] = calc_dM_du_singlepoint(
                u, u_tilde, x, ind, ind+2, nu, dx, dt,
                source_derivative_function_u, source_parameters
            )

        if ind >= 1:
            lower1[ind-1] = calc_dM_du_singlepoint(
                u, u_tilde, x, ind, ind-1, nu, dx, dt,
                source_derivative_function_u, source_parameters
            )

        if ind >= 2:
            lower2[ind-2] = calc_dM_du_singlepoint(
                u, u_tilde, x, ind, ind-2, nu, dx, dt,
                source_derivative_function_u, source_parameters
            )

    dM_dU = diags(
        [lower2, lower1, middle, upper1, upper2],
        offsets=[-2, -1, 0, 1, 2],
        format='csc'
    )

    return dM_dU




############################################################
# calculation of model jacobians at singular points
############################################################

# calculation of dMdu for one single point
############################################################
def calc_dM_du_singlepoint(
                                u, u_tilde, x,                              # solution description vectors
                                i,j,                                        # indices
                                nu, dx, dt,                                 # specified functions in solver
                                source_derivative_function_u,               # specified analytical function for the derivative of the source terms
                                source_parameters
                          ):

    # calculate residual gradient with respect to u
    dR_du_singlepoint = dunp1i_duj(
        u, u_tilde, x, i, j, nu, dx, dt,
        source_derivative_function_u, source_parameters
    )

    return dR_du_singlepoint



############################################################
# calculation of overall residual jacobians
############################################################

# calculation of residual jacobian with respect to u
############################################################
def calc_dR_dU(dM_dU, u, dt):
    return -dM_dU / dt


# calculation of dRdunp1 matrix
############################################################
def calc_dR_dUnp1(u, dt):
    return sp.sparse.eye(len(u), format='csc') / dt


# calculation of dtRdk matrix
############################################################
def calc_dtR_dk(
                        u, u_tilde, x,                              # solution description vectors
                        dt, nu, dx,                                 # specified constants in solver
                        source_derivative_function_k,               # specified analytical function for the derivative of the source terms wrt k
                        source_derivative_function_u,               # specified analytical function for the derivative of the source terms wrt u
                        source_parameters,                          # source parameters
               ):

    # calculate jacobians in U
    dtL_dU_jacobian = dL_dU(u_tilde, x, nu, dx, source_derivative_function_u, source_parameters)

    # calculate jacobians in k column by column
    dL_dk_columns = []
    dtL_dk_columns = []
    for k_ind in range(len(source_parameters)):
        dL_dk_columns.append(dL_dk_single_column(u, x, source_derivative_function_k, source_parameters, k_ind))
        dtL_dk_columns.append(dL_dk_single_column(u_tilde, x, source_derivative_function_k, source_parameters, k_ind))

    # merge columns
    dL_dk_jacobian = np.column_stack(dL_dk_columns)
    dtL_dk_jacobian = np.column_stack(dtL_dk_columns)

    # account for fixed boundary conditions in jacobians wrt k
    dL_dk_jacobian[[0, -1], :] = 0
    dtL_dk_jacobian[[0, -1], :] = 0

    # make residual jacobian
    dtR_dk = -1/2 * ( (sp.sparse.eye(len(u), format='csr') + dt*dtL_dU_jacobian) @ dL_dk_jacobian + dtL_dk_jacobian )

    return dtR_dk



############################################################
# calculation of du^{n+1}/du
############################################################

# calculation of one specific entry of du^{n+1}/du
############################################################
def dunp1i_duj(
                u, u_tilde, x,                              # solution description vectors
                i,j,                                        # indices
                nu, dx, dt,                                 # residual calculators
                source_derivative_function_u,               # specified analytical function for the derivative of the source terms
                source_parameters
              ):

    # calculate the reused term so we don't need to recalculate it
    dtu_du_val_ij = dtu_du(u, x, i, j, nu, dx, dt, source_derivative_function_u, source_parameters)


    # set up variable for sum of jacobian terms of the operator
    sum_dL_duk_dtuk_duj = 0
    for k in range(max(0, i-1), min(len(u), i+2)):
        # calculate derivative of operator
        sum_dL_duk_dtuk_duj += dLi_duj(u_tilde, x, i, k, nu, dx, source_derivative_function_u, source_parameters) * dtu_du(u, x, k, j, nu, dx, dt, source_derivative_function_u, source_parameters)


    # main calculation of the quantity
    dunplus1_duj = 0.5 * (dtu_du_val_ij + dt * sum_dL_duk_dtuk_duj)


    # get the dirac delta *1/2 for i == j case
    if i == j:
        dunplus1_duj += 0.5


    return dunplus1_duj



############################################################
# calculation of the semi-discrete jacobian in u and k
############################################################

# calculation of full semi-discrete jacobian in u
############################################################
def dL_dU(
                u, x,                                     # solution description vectors
                nu, dx,                                   # residual calculators
                source_derivative_function_u,             # specified analytical function for the derivative of the source terms
                source_parameters                         # source parameters
            ):

    n = len(u)

    # initialize the three diagonals
    upper = np.zeros(n-1)
    middle = np.zeros(n)
    lower = np.zeros(n-1)

    # calculate the interior entries of the jacobian
    for i in range(1, n-1):

        lower[i-1] = dLi_duj(u, x, i, i-1, nu, dx,
                             source_derivative_function_u, source_parameters)

        middle[i] = dLi_duj(u, x, i, i, nu, dx,
                            source_derivative_function_u, source_parameters)

        upper[i] = dLi_duj(u, x, i, i+1, nu, dx,
                           source_derivative_function_u, source_parameters)

    # construct sparse tridiagonal matrix
    J_L = diags(
        [lower, middle, upper],
        offsets=[-1, 0, 1],
        format='csr'
    )

    return J_L

# calculation of one entry of the semi-discrete jacobian in u
############################################################
def dLi_duj(
                u, x,                                       # solution description vectors
                i,j,                                        # indices
                nu, dx,                                     # residual calculators
                source_derivative_function_u,               # specified analytical function for the derivative of the source terms
                source_parameters                           # source parameters for the 
            ):

    if (i == 0) or (i == len(u)-1):
        dL_du = 0

    # J{i,j} = J_{i,i-1}
    elif i == (j+1):
        dL_du = (dFG_dUL(u[i-1],u[i])) / dx + nu / (dx**2)

    # J{i,j} = J_{i,i}
    elif i == j:
        dL_du = (dFG_dUR(u[i-1],u[i]) - dFG_dUL(u[i],u[i+1])) / dx - (2 * nu) / (dx**2) + source_derivative_function_u(u[i], x[i], source_parameters)

    # J{i,j} = J_{i,i+1}
    elif i == (j-1):
        dL_du = - (dFG_dUR(u[i], u[i+1])) / dx + nu / (dx**2)

    # warning to not calculate non-tridiagonal entries
    else:
        dL_du = 0
        warnings.warn("The function dLi_duj was called on an entry that is not tridiagonal.\n\tThe derivative has been set to zero, but the computational efficiency of utilizing sparse matrix algorithms may be lost!")


    return dL_du


# calculation of one column of the semi-discrete jacobian in k
############################################################
def dL_dk_single_column(
                u, x,                                       # solution description vectors
                source_derivative_function_k,               # specified analytical function for the derivative of the source terms
                source_parameters, source_parameter_index   # source parameters for the 
            ):

    dL_dk_column = source_derivative_function_k(u, x, source_parameters, source_parameter_index)

    return dL_dk_column



############################################################
# calculation of d\tilde{u}/du
############################################################

# calculation of one specific entry of d\tilde{u}/du
############################################################
def dtu_du(
                u, x,                                       # solution description vectors
                i,j,                                        # indices
                nu, dx, dt,                                 # residual calculators
                source_derivative_function_u,               # specified analytical function for the derivative of the source terms
                source_parameters
          ):

    dtilde_u_du = dt * dLi_duj(u, x, i, j, nu, dx, source_derivative_function_u, source_parameters)

    # in i == j case add in dirac delta
    if i == j:
        dtilde_u_du += 1

    return dtilde_u_du




############################################################
############################################################
# jacobian calculation helper / subcomponent functions
############################################################
############################################################

############################################################
# analytical differentiation of FG
############################################################

# calculation of derivative with respect to UL
############################################################
def dFG_dUL(uL, uR):
        # Case A: shock
    if uL > uR:
        s = (uL + uR) / 2
        if s > 0:
            df_dUL = uL
        else:
            df_dUL = 0
    
    # Case B: rarefaction
    elif uL < uR:
        if uL > 0:
            df_dUL = uL
        elif uR < 0:
            df_dUL = 0
        else:
            df_dUL = 0

    # Case C: no wave
    else:
        if uL > 0:
            df_dUL = uL
        else:
            df_dUL = 0

    return df_dUL


# calculation of derivative with respect to UR
############################################################
def dFG_dUR(uL, uR):
    # Case A: shock
    if uL > uR:
        s = (uL + uR) / 2
        if s > 0:
            df_dUR = 0
        else:
            df_dUR = uR
    
    # Case B: rarefaction
    elif uL < uR:
        if uL > 0:
            df_dUR = 0
        elif uR < 0:
            df_dUR = uR
        else:
            df_dUR = 0

    # Case C: no wave
    else:
        if uR < 0:
            df_dUR = uR
        else:
            df_dUR = 0

    return df_dUR