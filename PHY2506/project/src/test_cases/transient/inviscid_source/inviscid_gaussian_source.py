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



# make source function
def S_fcn(u, mu=10, sig=1):
    return 1 / np.sqrt(2*np.pi*sig**2) * np.exp((-(u-mu)**2) / (2*sig**2))


# run test case
############################################################
n_points=1000
x = np.linspace(-1,1,n_points)
# t_array = np.linspace(0,1,1000)
t_f = 1.6#1e-10 #1e-10
# u = np.append(1*np.ones(int(n_points/2)), -1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1
u = x.copy()
# S = S_fcn # sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x)) #sp.stats.norm.pdf(x)
BCs = [0,0]

df = transient_solve(u, x, t_f, S_fcn, kin_viscosity=0, BCs=BCs, data_save_interval=0.2)

for t in np.unique(np.array(df.t)):
    df_t = df[df.t == t]
    plt.plot(df_t.x,df_t.u, label=f"t={str(t)[:6]}")

plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.title(r"$u(x)$ for a Rarefaction Wave Modeled Using Inviscid Burger's Equation")
plt.legend(loc='upper right')
plt.show()