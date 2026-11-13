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
from src.viscous_burgers_solver.transient_solver import transient_solve



# run test case
############################################################
n_points=500
x = np.linspace(0,1,n_points)
# t_array = np.linspace(0,1,1000)
t_f = 2#1e-10 #1e-10
u = np.zeros(len(x)) # np.append(1*np.ones(int(n_points/2)), -1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1
def S(u):
    return 0 # sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x)) #sp.stats.norm.pdf(x)
BCs = [1.5,-1]

df, df_ic, df_resid, df_alldata = transient_solve(u, x, t_f, S, kin_viscosity=0, BCs=BCs, data_save_interval=0.1)



fig, ax = plt.subplots()

times = np.unique(df.t)
cmap = plt.cm.plasma
norm = plt.Normalize(vmin=times.min(), vmax=times.max())

for t in times:
    df_t = df[df.t == t]
    ax.plot(df_t.x, df_t.u, color=cmap(norm(t)))

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.set_title(r"$u(x)$ for a Moving Shock Wave Modeled"+"\n"+f"Using Inviscid Burger's Equation with BCs={BCs}")

# Add colour bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label=r"$t$")

plt.show()