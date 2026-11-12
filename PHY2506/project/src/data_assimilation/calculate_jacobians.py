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
# calculation of overall residual jacobians
############################################################

# calculation of residual jacobian with respect to u
############################################################
def calc_dR_du(
                    u,                                          # solution description vectors
                    nu, dx, dt,                                 # residual calculators
                    source_derivative_function                  # specified analytical function for the derivative of the source terms
              ):

    # make arrays for tridiagonal matrix
    upper = np.zeros(len(u) - 1)
    middle = np.zeros(len(u))
    lower = np.zeros(len(u) - 1)

    # fill arrays with proper values for tridiagonal matrix
    for ind in range(len(u)):
        # top left corner of jacobian
        if ind == 0:
            upper[ind] = calc_dR_dU_singlepoint(u, ind, ind+1, nu, dx, dt, source_derivative_function)
            middle[ind] = calc_dR_dU_singlepoint(u, ind, ind, nu, dx, dt, source_derivative_function)

        # bottom right corner of jacobian
        elif ind == (len(u)-1):
            middle[ind] = calc_dR_dU_singlepoint(u, ind, ind, nu, dx, dt, source_derivative_function)
            lower[ind-1] = calc_dR_dU_singlepoint(u, ind, ind-1, nu, dx, dt, source_derivative_function)

        # all other jacobian entries
        else:
            upper[ind] = calc_dR_dU_singlepoint(u, ind, ind+1, nu, dx, dt, source_derivative_function)
            middle[ind] = calc_dR_dU_singlepoint(u, ind, ind, nu, dx, dt, source_derivative_function)
            lower[ind-1] = calc_dR_dU_singlepoint(u, ind, ind-1, nu, dx, dt, source_derivative_function)

    dR_dU = diags(
        [lower, middle, upper],
        offsets=[-1,0,1],
        format='csc'
    )

    return dR_dU



############################################################
# calculation of residual jacobians at singular points
############################################################

# calculation of dRdU for one single point
############################################################
def calc_dR_dU_singlepoint(
                                u,                                          # solution description vectors
                                i,j,                                        # indices
                                nu, dx, dt,                                 # specified functions in solver
                                source_derivative_function_u                # specified analytical function for the derivative of the source terms
                          ):

    # calculate residual gradient with respect to u
    dR_du_singlepoint = dunp1i_duj(u,i,j,nu,dx,dt,source_derivative_function_u)

    # add in dirac delta term
    if i == j:
        dR_du_singlepoint -= 1


    return dR_du_singlepoint


# calculation of dRdk
############################################################
def calc_dR_dk():





    return



############################################################
# calculation of du^{n+1}/du
############################################################

# calculation of one specific entry of du^{n+1}/du
############################################################
def dunp1i_duj(
                u,                                          # solution description vectors
                i,j,                                        # indices
                nu, dx, dt,                                 # residual calculators
                source_derivative_function_u                # specified analytical function for the derivative of the source terms
              ):

    # calculate the reused term so we don't need to recalculate it
    dtu_du_val_ij = dtu_du(u, i, j, nu, dx, dt, source_derivative_function_u)


    # set up variable for sum of jacobian terms of the operator
    sum_dL_duk_dtuk_duj = 0
    for k in range(i-1, i+2):
        # calculate derivative of operator
        sum_dL_duk_dtuk_duj += dLi_duj(u, i, k, nu, dx, source_derivative_function_u) * dtu_du(u, k, j, nu, dx, dt, source_derivative_function_u)


    # main calculation of the quantity
    dunplus1_duj = 0.5 * (dtu_du_val_ij + dt * sum_dL_duk_dtuk_duj)


    # get the dirac delta *1/2 for i == j case
    if i == j:
        dunplus1_duj += 0.5


    return dunplus1_duj



############################################################
# calculation of the semi-discrete jacobian
############################################################

# calculation of one entry of the semi-discrete jacobian
############################################################
def dLi_duj(
                u,                                          # solution description vectors
                i,j,                                        # indices
                nu, dx,                                     # residual calculators
                source_derivative_function_u                # specified analytical function for the derivative of the source terms
            ):

    # J{i,j} = J_{i,i-1}
    if i == (j+1):
        dL_du = (dFG_dUL(u[i-1],u[i])) / dx + nu / (dx**2) + source_derivative_function_u(i,j)

    # J{i,j} = J_{i,i}
    elif i == j:
        dL_du = (dFG_dUR(u[i-1],u[i]) - dFG_dUL(u[i],u[i+1])) / dx - (2 * nu) / (dx**2) + source_derivative_function_u(i,j)

    # J{i,j} = J_{i,i+1}
    elif i == (j-1):
        dL_du = - (dFG_dUR(u[i], u[i+1])) / dx + nu / (dx**2) + source_derivative_function_u(i,j)

    # warning to not calculate non-tridiagonal entries
    else:
        dL_du = 0
        warnings.warn("The function dLi_duj was called on an entry that is not tridiagonal.\n\tThe derivative has been set to zero, but the computational efficiency of utilizing sparse matrix algorithms may be lost!")


    return dL_du



############################################################
# calculation of d\tilde{u}/du
############################################################

# calculation of one specific entry of d\tilde{u}/du
############################################################
def dtu_du(
                u,                                          # solution description vectors
                i,j,                                        # indices
                nu, dx, dt,                                 # residual calculators
                source_derivative_function                  # specified analytical function for the derivative of the source terms
          ):

    dtilde_u_du = dt * dLi_duj(u, i, j, nu, dx, source_derivative_function)

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
        df_dUL = uL

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
        df_dUR = uR             # CHECK THIS (it is possible that this needs to be zero)

    return df_dUR