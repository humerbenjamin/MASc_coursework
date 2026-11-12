# add the src header to pythonpath
############################################################
import sys
sys.path.append(".")


# import standard modules
############################################################
    # full libraries
import numpy as np
import matplotlib.pyplot as plt

# import personal files
############################################################
    # specific functions
from src.viscous_burgers_solver.steady_solver import steady_solve



# define source function
############################################################
def S_fcn(u,x,t, source_params):
    S = source_params[0]*u *  1 / np.sqrt(2*np.pi*source_params[2]**2) * np.exp((-(x-source_params[1])**2) / (2*source_params[2]**2))
    # S = -2*u *  1 / np.sqrt(2*np.pi*0.5**2) * np.exp((-(x-0)**2) / (2*0.5**2))
    return S

# run test case
############################################################
n_points=500
x = np.linspace(-1,1,n_points)
# t_array = np.linspace(0,1,1000)

u = np.zeros(len(x)) # np.append(1*np.ones(int(n_points/2)), -1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1

source_params = [-2,0,0.5]

n_iter = 10000
C=0.1

nu=0
BCs = [2,-2]


df, df_ic, df_resid = steady_solve(u, x, S_fcn, source_params, n_iter, C, nu, BCs, True)



# generate figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot
ax1.plot(df_ic.x, df_ic.u, label="Initial State", color='b')
ax1.plot(df.x, df.u, label='Converged Solution', color='k')
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