# add the src header to pythonpath
############################################################
import sys
sys.path.append("/home/humerben/Documents/uoft_MASc/coursework/MASc_coursework/PHY2506/project")


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
t_f = 1#1e-10 #1e-10
u = np.append(-1*np.ones(int(n_points/2)), 1*np.ones(int(n_points/2)))#100*sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x))#-x**2+1
S = np.zeros(len(x)) # sp.stats.norm.pdf(x, scale=0.25) #np.zeros(len(x)) #sp.stats.norm.pdf(x)
BCs = [-1,1]

df = transient_solve(u, x, t_f, S=S, kin_viscosity=0, BCs=BCs, data_save_interval=0.1)

for t in np.unique(np.array(df.t)):
    df_t = df[df.t == t]
    plt.plot(df_t.x,df_t.u, label=f"t={str(t)[:6]}")

plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.title(r"$u(x)$ for a Rarefaction Wave Modeled Using Inviscid Burger's Equation")
plt.legend()
plt.show()