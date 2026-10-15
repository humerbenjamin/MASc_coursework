# import libraries
###########################################################
import matplotlib.pyplot as plt
import numpy as np


# main function for (a)
###########################################################
def part_a(Lx=1,Ld=0.2,J=32):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    rho = correlation_metric(X, Y, Ld)

    # plotting
    plt.contour(X, Y, rho, levels=20)
    plt.colorbar(label=r"Correlation $\rho(x,y)$")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(r"Contour Plot of the Correlation $\rho(x,y)$")
    plt.show()

    return 


# main function for (b)
###########################################################
def part_b(Lx=1,Ld=0.2,J=500):
    # create correlation metric
    dx=2*Lx/J
    x1=np.zeros(J)
    x2=Lx*np.ones(J)
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    rho1=correlation_metric(x1,y,Ld)
    rho2=correlation_metric(x2,y,Ld)

    # plotting
    plt.plot(y, rho1, label=r"$x_j=0$")
    plt.plot(y, rho2, label=r"$x_j=L_x=1$")
    plt.xlabel(r"$y$")
    plt.ylabel(r"Correlation $\rho(x_j,y)$")
    plt.title(r"Correlation $\rho(x,y)$ as a function of $y$ at fixed values of $x=x_j$")
    plt.legend()
    plt.show()

    return



# main function for (d)
###########################################################
def part_d(Lx=1,Ld=0.2,J=200):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    x1=np.zeros(J)
    x2=Lx*np.ones(J)
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    rho = correlation_metric(X, Y, Ld)
    rho1=correlation_metric(x1,y,Ld)
    rho1_old=correlation_metric(x1,y,Ld)
    rho2=correlation_metric(x2,y,Ld)
    rho2_old=correlation_metric(x2,y,Ld)

    # mask correlation metric
    Lc = 3*Ld
    rho[np.abs(X - Y) > Lc] = 0
    rho1[np.abs(x1 - y) > Lc] = 0
    rho2[np.abs(x2 - y) > Lc] = 0

    # plotting
    plt.contour(X, Y, rho, levels=20)
    plt.colorbar(label=r"Correlation $\rho(x,y)$")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(r"Contour Plot of the Correlation $\rho(x,y)$")
    plt.show()

    plt.plot(y, rho1_old, label=r"$x_j=0$", alpha=0.7, linestyle="--", zorder=2)
    plt.plot(y, rho1, label=r"$x_j=0$ with clipping to $0$", zorder=1)
    plt.plot(y, rho2_old, label=r"$x_j=L_x=1$", linestyle='--', zorder=2)
    plt.plot(y, rho2, label=r"$x_j=L_x=1$ with clipping to $0$", zorder=1)
    plt.xlabel(r"$y$")
    plt.ylabel(r"Correlation $\rho(x_j,y)$")
    plt.title(r"Correlation $\rho (x,y)$ as a function of $y$ at fixed values of $x=x_j$"+"\n"+r"$\rho(x_j,y)=0$ for $|x_j-y|>L_c$")
    plt.legend()
    plt.grid(alpha=0.8)
    plt.show()

    return


# main function for (e)
###########################################################
def part_e(Lx=1,Ld=0.2,J=200):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    x1=np.zeros(J)
    x2=Lx*np.ones(J)
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    rho = correlation_metric_triangular(X, Y)
    rho_old = correlation_metric_triangular(X, Y)
    rho1=correlation_metric_triangular(x1,y)
    rho1_old=correlation_metric_triangular(x1,y)
    rho2=correlation_metric_triangular(x2,y)
    rho2_old=correlation_metric_triangular(x2,y)

    # mask correlation metric
    Lc = 3*Ld
    rho[np.abs(X - Y) > Lc] = 0
    rho1[np.abs(x1 - y) > Lc] = 0
    rho2[np.abs(x2 - y) > Lc] = 0

    # plotting
    plt.contour(X, Y, rho_old, levels=20)
    plt.colorbar(label=r"Correlation $\rho(x,y)$")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(r"Contour Plot of the Correlation $\rho(x,y)$")
    plt.show()

    plt.contour(X, Y, rho, levels=20)
    plt.colorbar(label=r"Correlation $\rho(x,y)$")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(r"Contour Plot of the Correlation $\rho(x,y)$ With Clipping")
    plt.show()

    plt.plot(y, rho1_old, label=r"$x_j=0$", alpha=0.7, linestyle="--", zorder=2)
    plt.plot(y, rho1, label=r"$x_j=0$ with clipping to $0$", zorder=1)
    plt.plot(y, rho2_old, label=r"$x_j=L_x=1$", linestyle='--', zorder=2)
    plt.plot(y, rho2, label=r"$x_j=L_x=1$ with clipping to $0$", zorder=1)
    plt.xlabel(r"$y$")
    plt.ylabel(r"Correlation $\rho(x_j,y)$")
    plt.title(r"Correlation $\rho (x,y)$ as a function of $y$ at fixed values of $x=x_j$"+"\n"+r"$\rho(x_j,y)=0$ for $|x_j-y|>L_c$")
    plt.legend()
    plt.grid(alpha=0.8)
    plt.show()

    return

# main function for (f)
###########################################################
def part_f(Lx=1, Ld=0.2, J=200):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    x1=1
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    rho_gaussian = correlation_metric(X, Y, Ld)
    rho_triangular = correlation_metric_triangular(X, Y)
    rho_combined = rho_gaussian * rho_triangular

    # create functions based on the correlation metrics
    rho_fcn_gaussian = rho_gaussian[X == x1]
    rho_fcn_triangular = rho_triangular[X == x1]
    rho_fcn_combined = rho_combined[X == x1]


    # plotting
    plt.contour(X, Y, rho_combined, levels=20)
    plt.colorbar(label=r"Correlation $\rho(x,y)$")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(r"Contour Plot of the Correlation $\rho(x,y)$")
    plt.show()

    plt.plot(y, rho_fcn_triangular, label=r"$x_j=0$, triangular correlation function ($T$)", alpha=0.6, linestyle="--", zorder=1)
    plt.plot(y, rho_fcn_gaussian, label=r"$x_j=0$, gaussian correlation function ($B$)", alpha=0.6, linestyle="--", zorder=1)
    plt.plot(y, rho_fcn_combined, label=r"$x_j=0$, combined correlation function ($Q$)", alpha=0.9, zorder=2)
    plt.xlabel(r"$y$")
    plt.ylabel(r"Correlation $\rho(x_j,y)$")
    plt.title(r"Correlation $\rho (x,y)$ as a function of $y$ at fixed values of $x=x_j$")
    plt.legend()
    plt.grid(alpha=0.8)
    plt.show()

    return



# correlation function
###########################################################
def correlation_metric(x,y,Ld):
    return np.exp(-1/2 * (x-y)**2 / (Ld**2))

def correlation_metric_triangular(x,y):
    return 1 - np.abs(x-y)


# block for running functions
###########################################################
if __name__ == '__main__':
    do_a = False
    do_b = False
    do_d = False
    do_e = False
    do_f = True
    if do_a:
        part_a()
    if do_b:
        part_b()
    if do_d:
        part_d()
    if do_e:
        part_e()
    if do_f:
        part_f()