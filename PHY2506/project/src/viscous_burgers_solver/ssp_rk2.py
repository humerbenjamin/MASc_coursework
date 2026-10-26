# import standard modules
############################################################
    # full libraries
import numpy as np


# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.semi_discrete_solver import L


############################################################
# isolated ssp-rk2 function
############################################################
def ssp_rk2(u, x, S, t_c, A, b, dt, kin_viscosity):
    '''
    Strong stability preserving Runge-Kutta 2 time marching
    


    
    '''
    u_tilde = u + dt * L(u, x, S, t_c, A, b, nu=kin_viscosity)
    u_f = 1/2*u + 1/2*(u_tilde + dt*L(u_tilde, x, S, t_c, A, b, nu=kin_viscosity))

    return u_f


############################################################
# helpers
############################################################

# timestep calculation
############################################################
def calc_timestep(u, kin_viscosity, dx, C):
    max_u = np.max(np.abs(u))

    if max_u == 0 and kin_viscosity == 0:
        return 1e-7

    dt_conv = np.inf if max_u == 0 else dx / max_u

    dt_diff = (
        np.inf if kin_viscosity == 0
        else dx**2 / (2 * kin_viscosity)
    )

    return C * min(dt_conv, dt_diff)