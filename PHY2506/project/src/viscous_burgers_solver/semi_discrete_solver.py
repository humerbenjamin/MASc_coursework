# import standard modules
############################################################
# full libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
# sublibraries
from scipy.sparse import diags



############################################################
# main function for semi-discrete solve of du/dt
############################################################
def L(u, x, S, t, A=np.array([]), b=np.array([]), BCs=[0,0], nu=1.5e-5):
    dx = np.abs(x[1]-x[0])

    # make matrix A if it has not already been given
    if np.size(A) == 0:
        matrix_A_warning_msg="Matrix A has not been specified, and will be automatically generated each time L(u,t) is called.\nIf This is not intended, please specify matrix A."
        warnings.warn(matrix_A_warning_msg, category=RuntimeWarning)
        A = make_A(u)
    
    # make vector b if it has not already been given
    if np.size(b) == 0:
        vector_b_warning_msg="Vector b has not been specified, and will be automatically generated each time L(u,t) is called.\nIf This is not intended, please specify vector b."
        warnings.warn(vector_b_warning_msg, category=RuntimeWarning)
        b = make_b(u, BCs)

    # utilize godunov solver to obtain the convective term
    u_length = len(u)
    L_conv = np.zeros(u_length)
    for i in range(u_length):
        if i == 0:
            L_conv[i] = (F_godunov(BCs[0],u[i]) - F_godunov(u[i],u[i+1])) / dx
        elif i < u_length-1:
            L_conv[i] = (F_godunov(u[i-1],u[i]) - F_godunov(u[i],u[i+1])) / dx
        else:
            L_conv[i] = (F_godunov(u[i-1],u[i]) - F_godunov(u[i],BCs[1])) / dx

    dudt = L_conv + (nu/dx**2)*A@u + b + S

    return dudt



############################################################
# godunov solver
############################################################
def F_godunov(uL,uR):
    # Case A: shock
    if uL > uR:
        s = (uL + uR) / 2
        if s > 0:
            f = (uL**2) / 2
        else:
            f = (uR**2) / 2
    
    # Case B: rarefaction
    elif uL < uR:
        if uL > 0:
            f = (uL**2) / 2
        elif uR < 0:
            f = (uR**2) / 2
        else:
            f = 0   # check

    # Case C: no wave
    else:
        f = (uL**2)/2

    return f


############################################################
# helpers
############################################################

# generate the matrix A for FD calculation
############################################################
def make_A(u):
    n = len(u)

    lower = -np.ones(n - 1)
    main = 2 * np.ones(n)
    upper = -np.ones(n - 1)

    A = diags(
        [lower, main, upper],
        offsets=[-1, 0, 1],
        format="csr"
    )

    A[0,0]=3
    A[-1,-1]=3

    return A

# generate the matrix b for FD calculation
############################################################
def make_b(u, BCs):
    n = len(u)
    b = np.zeros(n)
    b[0] = BCs[0]
    b[-1] = BCs[-1]

    return b




if __name__ == '__main__':
    x = np.linspace(-1,1,1000)
    t_array = np.linspace(0,1,1000)
    dt = t_array[1]-t_array[0]
    u = np.zeros(len(x))#-x**2+1
    S = sp.stats.norm.pdf(x)

    # t_array = t_array[:20]

    for t in t_array:
        plt.plot(x,u, label=f"t={str(t)[:6]}")
        dudt = L(u,x,S,t)
        u = u+dudt*dt

    plt.grid(alpha=0.7)
    plt.legend()
    plt.show()