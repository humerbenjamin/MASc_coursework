# import standard modules
############################################################
    # full libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
    # sublibraries
from time import time
from scipy.sparse import diags, linalg


############################################################
# plot the transient solve data for a single forward run
############################################################
def plot_source_function_evolution(x, source_function, true_parameters, df_parameter_DA_history):
    # get the figure and the axes
    fig, ax = plt.subplots(dpi=200)

    # get data required for the plotting loop
    cmap = plt.cm.plasma
    norm = plt.Normalize(1, len(df_parameter_DA_history))

    # add plot for each time that data was saved
    for i in range(len(df_parameter_DA_history)):
        print(df_parameter_DA_history.iloc[i].to_numpy()[1:])
        ax.plot(x, source_function(1, x, df_parameter_DA_history.iloc[i].to_numpy()[1:]), color=cmap(norm(i+1)), zorder=1)

    # plot truth
    ax.plot(x, source_function(1, x, true_parameters), color='k', linestyle='--', linewidth=1.5, zorder=4)

    # set up axis labels
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$S(u=1,x;\vec{k})$")
    ax.set_title(r"$S(u=1,x;\vec{k})$ Coloured by VDA Iteration"+"\n"+"for Inviscid Burger's Equation")

    # Add colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=r"VDA Iteration")

    # fix layout and show plot
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()

    return