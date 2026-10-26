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
n_points=1000
x = np.linspace(-1,1,n_points)
# t_array = np.linspace(0,1,1000)
t_f = 1.6#1e-10 #1e-10
u = np.sin(np.pi*x)
def S(u):
    return 0
BCs = [0,0]

df = transient_solve(u, x, t_f, S, kin_viscosity=0, BCs=BCs, data_save_interval=0.2)

for t in np.unique(np.array(df.t)):
    df_t = df[df.t == t]
    plt.plot(df_t.x,df_t.u, label=f"t={str(t)[:6]}")

plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.title(r"$u(x)$ for a Rarefaction Wave Modeled Using Inviscid Burger's Equation")
plt.legend(loc='upper right')
plt.show()