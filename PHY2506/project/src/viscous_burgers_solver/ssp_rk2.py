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
def ssp_rk2(u, x, S, t_c, A, b, BCs, dt, kin_viscosity):
    u_tilde = u + dt * L(u, x, S, t_c, A, b, BCs, nu=kin_viscosity)
    u_f = 1/2*u + 1/2*(u_tilde + dt*L(u_tilde, x, S, t_c, A, b, BCs, nu=kin_viscosity))

    return u_f


############################################################
# helpers
############################################################

# timestep calculation
############################################################
def calc_timestep(u, BCs, kin_viscosity, dx, C, t_cur, save_times, strict_data_saving):

    max_u = np.max([np.max(np.abs(u)), np.max(np.abs(BCs))])

    if max_u == 0 and kin_viscosity == 0:
        dt = 1e-7
    else:
        dt_conv = np.inf if max_u == 0 else dx / max_u

        dt_diff = (
            np.inf if kin_viscosity == 0
            else dx**2 / (2 * kin_viscosity)
        )

        dt = C * min(dt_conv, dt_diff)

    # Limit timestep to reach the next save time exactly
    next_save_time = save_times[save_times > t_cur][0]

    dt_save = next_save_time - t_cur

    if dt_save < dt:
        saving_step = True
        
    else:
        saving_step = False

    if strict_data_saving:
        dt=min(dt, dt_save)

    return dt, saving_step, next_save_time