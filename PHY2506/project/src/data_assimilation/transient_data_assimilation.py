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


# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.transient_solver import transient_solve




############################################################
# main function for 4Dvar of transient solutions
############################################################
def variational_DA_transient(data_loc, source_function, source_parameters, C, kin_viscosity, BCs, verbose):
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
        # get initial condition vectors
    u0 = df_ic.u
    x = df_ic.x
        # get timing quantities
    t_f = sorted_data_times[-1]
    data_save_interval = sorted_data_times[1] - sorted_data_times[0]




    df, df_ic, df_resid = transient_solve(
                                                u0, x, t_f,
                                                S_fcn=source_function, source_params=source_parameters,
                                                C=C, kin_viscosity=kin_viscosity, BCs=BCs,
                                                verbose=verbose,
                                                data_save_interval=data_save_interval, strict_data_saving=True
                                         )

    plot_transient_solve_single_run_forward_plot(df, df_ic, df_resid)


    return




############################################################
# main function for 4Dvar of transient solutions
############################################################
def plot_transient_solve_single_run_forward_plot(df, df_ic, df_resid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    times = np.unique(df.t)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=times.min(), vmax=times.max())

    for t in times:
        df_t = df[df.t == t]
        ax1.plot(df_t.x, df_t.u, color=cmap(norm(t)))


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