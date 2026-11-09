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



# run test case
############################################################
n_points=200
x = np.linspace(0,0.1,n_points)
# t_array = np.linspace(0,1,1000)
u = np.zeros(len(x)) # np.append(1*np.ones(int(n_points/2)), -1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1
def S(u):
    return 0 # sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x)) #sp.stats.norm.pdf(x)
BCs = [1,-1]

df, df_ic, df_resid = steady_solve(u, x, S, n_iter=100000, C=0.1, kin_viscosity=0, BCs=BCs, verbose=True, convergence_threshold=1e-6)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot
ax1.plot(df_ic.x, df_ic.u, label="Initial State")
ax1.plot(df.x, df.u, label='Converged Solution')
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

plt.tight_layout()
plt.show()