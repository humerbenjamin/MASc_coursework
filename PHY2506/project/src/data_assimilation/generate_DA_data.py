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
from src.viscous_burgers_solver.steady_solver import steady_solve
from src.viscous_burgers_solver.transient_solver import transient_solve


############################################################
############################################################
# main DA data generation
############################################################
############################################################

# data generation function
############################################################
def generate_data(  
                    solve_type, u0, x,                                              # direct functional inputs
                    S_fcn, source_parameters,                                       # source function requirements
                    kin_viscosity=1.5e-5, BCs=[0,0],                                # additional system parameters
                    n_iter=1000, C=0.1, convergence_threshold=1e-6,                 # solver parameters for steady solve
                    t_f=2, data_save_interval=0.1, strict_data_saving=True,         # solver parameters for transient solve
                    verbose=False, plot_values=True,                                # midstream outputs
                    output_filename='data/test',                                    # data locations
                    measurement_locations=[], random_seed=42, SNR=0.1               # noisy measurement generation
                 ):

    # solve initial data

    # find steady solution if the data is steady
    if solve_type == 'steady':
        # perform the steady solve
        df, df_ic, df_resid = steady_solve(u0, x, S_fcn, source_parameters, n_iter, C=C, kin_viscosity=kin_viscosity, BCs=BCs, verbose=verbose, convergence_threshold=convergence_threshold)

        # generate mapping matrix from simulation to observation space
        H, measurement_indices = make_H_matrix(measurement_locations, df_ic.x)

        # generate random noise on data and then save it
        np.random.seed=random_seed
        stdev = np.mean(np.abs(df.u))*SNR
        u_noisy = np.random.normal(df.u, scale=stdev)
        df_noisy = pd.DataFrame({"u_original":H@df.u, "u_noisy":H@u_noisy, "x":H@df.x, "t":H@df.t, "indices":measurement_indices, "stdev":np.ones(len(measurement_indices))*stdev})

        # plot values if plot_values was specified
        if plot_values:
            make_steady_plot(df, df_ic, df_resid, df_noisy, BCs)

        # save data
        df.to_csv(f"{output_filename}_rawdata.csv", index=False)
        df_ic.to_csv(f"{output_filename}_rawdata_IC.csv", index=False)
        df_resid.to_csv(f"{output_filename}_rawdata_resid.csv", index=False)
        df_noisy.to_csv(f"{output_filename}_measurements_noisy.csv", index=False)

    # find transient soluttion if the data is transient
    elif solve_type == 'transient':
        df, df_ic, df_resid, df_alldata = transient_solve(u0, x, t_f, S_fcn, source_parameters, C, kin_viscosity, BCs, verbose=verbose, data_save_interval=data_save_interval, strict_data_saving=strict_data_saving)

        # generate mapping matrix from simulation to observation space
        H, measurement_indices = make_H_matrix(measurement_locations, df_ic.x)

        # specify random seed
        np.random.seed=random_seed
        
        # need to change how this is calculated since we will need it at every timestep, and H only maps for a single timestep
        sorted_times = np.sort(np.unique(df.t))
        for t_ind in range(len(sorted_times)):

            df_singletime = df[df.t == sorted_times[t_ind]]

            # generate random noise on data for this specific timestep
            stdev = np.mean(np.abs(df_singletime.u))*SNR
            u_noisy = np.random.normal(df_singletime.u, scale=stdev)

            if t_ind == 0:
                df_noisy = pd.DataFrame({
                                            "u_original":H@df_singletime.u, "u_noisy":H@u_noisy,                                            # different values of u
                                            "x":H@df_singletime.x, "t":H@df_singletime.t,                                                   # background values (x,t)
                                            "measurement_indices":measurement_indices, "measurement_locations":measurement_locations,       # measurement location data
                                            "stdev":np.ones(len(measurement_indices))*stdev                                                 # statistical information
                                        })
            else:
                df_noisy = pd.concat(
                                            [df_noisy, pd.DataFrame({                                                                       # set up for concatenation
                                            "u_original":H@df_singletime.u, "u_noisy":H@u_noisy,                                            # different values of u
                                            "x":H@df_singletime.x, "t":H@df_singletime.t,                                                   # background values (x,t)
                                            "measurement_indices":measurement_indices, "measurement_locations":measurement_locations,       # measurement location data
                                            "stdev":np.ones(len(measurement_indices))*stdev                                                 # statistical information
                                        })]
                                    )

        # plot values if plot_values was specified
        if plot_values:
            make_transient_plot(df, df_ic, df_resid, df_noisy, BCs)

        # save data
        df.to_csv(f"{output_filename}_rawdata.csv", index=False)
        df_ic.to_csv(f"{output_filename}_rawdata_IC.csv", index=False)
        df_resid.to_csv(f"{output_filename}_rawdata_resid.csv", index=False)
        df_noisy.to_csv(f"{output_filename}_measurements_noisy.csv", index=False)


    # throw error if not properly specified
    else:
        raise ValueError("Invalid value for string: solve_type.\n\tValid values include: 'steady' and 'transient.")




    return



############################################################
############################################################
# helper functions
############################################################
############################################################

############################################################
# data generation helper functions
############################################################

# function for generating the observation matrix H
############################################################
def make_H_matrix(measurement_locations, x):
    """
    Construct a sparse measurement matrix mapping grid values onto
    measurement locations using nearest-neighbour interpolation.

    Parameters
    ----------
    measurement_locations : ndarray, shape (m,)
        Locations where measurements are taken.
    x : ndarray, shape (n,)
        Computational grid locations.

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (m, n)
        Sparse measurement matrix.
    indices : ndarray, shape (m,)
        Index in x corresponding to each measurement location.
    """

    measurement_locations = np.asarray(measurement_locations)
    x = np.asarray(x)

    # Find nearest grid point for each measurement location
    indices = np.abs(measurement_locations[:, None] - x[None, :]).argmin(axis=1)

    # Construct sparse measurement matrix
    m = len(measurement_locations)
    H = csr_matrix(
        (np.ones(m), (np.arange(m), indices)),
        shape=(m, len(x))
    )

    return H, indices


############################################################
# plotting helper functions
############################################################

# plot for steady state data
############################################################
def make_steady_plot(df, df_ic, df_resid, df_noisy, BCs):
    # generate figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot
    ax1.plot(df_ic.x, df_ic.u, label="Initial State", color='b')
    ax1.plot(df.x, df.u, label='Converged Solution', color='k')
    ax1.scatter(df_noisy.x, df_noisy.u_noisy, label='Scattered Data With Noise Added', color='r', marker='.', alpha=0.75)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$u$")
    ax1.set_title(f"Converged Solution with BCs={BCs}")
    ax1.legend()

    # Right plot
    ax2.plot(df_resid.n, df_resid.R_norm)
    ax2.set_ylabel(r"$R$")
    ax2.set_xlabel(r"$N_\mathrm{iter}$")
    ax2.set_title("Residual Over Number of Iterations")
    ax2.set_xscale('linear')
    ax2.set_yscale('log')

    # fix layout and show plot
    plt.tight_layout()
    plt.show()

    return


# plot for transient data
############################################################
def make_transient_plot(df, df_ic, df_resid, df_noisy, BCs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    times = np.unique(df.t)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=times.min(), vmax=times.max())

    for t in times:
        df_t = df[df.t == t]
        df_tn = df_noisy[df_noisy.t == t]
        ax1.plot(df_t.x, df_t.u, color=cmap(norm(t)))
        ax1.scatter(df_tn.x, df_tn.u_noisy, color=cmap(norm(t)), marker='x', alpha=0.75)


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