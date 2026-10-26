# import libraries
###########################################################
import matplotlib.pyplot as plt
import numpy as np




# function for part a
###########################################################
def part_a(Lx=1, Ld=0.2, J=32, num_evals_to_plot=6):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    x1=1
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    rho_gaussian = correlation_metric(X, Y, Ld)

    # solve the eigenproblem
    evals, evecs = np.linalg.eig(rho_gaussian)
    evals_sorting_indices = np.argsort(evals)[::-1]
    evals_sorted, evals_indices = [evals[ind] for ind in evals_sorting_indices], [ind+1 for ind in evals_sorting_indices]

    # plot the eigenvalues
    plt.plot(evals_sorted)
    plt.title(r"Eigenvalues of Correlation Matrix $Q_{ij}$ Over"+"\nEigenvalue Index in Descending Order")
    plt.ylabel(r"Eigenvalue ($\lambda_q$)")
    plt.xlabel("Eigenvalue Index")
    plt.grid(alpha=0.8)
    plt.show()

    # plot the eigenvectors
    for i in range(num_evals_to_plot):
        plt.plot(x, evecs[:, evals_sorting_indices[i]], label=rf'$\lambda_{{{i+1}}}$='+format(evals[evals_sorting_indices[i]], ".3g"))

    # add data to the plot of eigenvectors
    plt.title(r"Eigenvectors of Correlation Matrix $Q_{ij}=\rho(x_i,y_j)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Magnitude of Eigenvector Component at $x=x_j$")
    plt.grid(alpha=0.8)
    plt.legend()
    plt.show()

    return


# function for part b
###########################################################
def part_b(Lx=1, Ld=0.2, J=32, num_evals_to_plot=6, sig_r=1, sig_b=2):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    x1=1
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    Q = correlation_metric(X, Y, Ld)

    # make other matrices and then A
    R = sig_r**2*np.eye(J)
    D = sig_b*np.eye(J)
    P = np.linalg.matmul(D, np.linalg.matmul(Q,D))
    A = np.matmul(P, np.linalg.inv(P+R))

    # solve the eigenproblem
    evals_q, evecs_q = np.linalg.eig(Q)
    evals_sorting_indices_q = np.argsort(evals_q)[::-1]
    evals, evecs = np.linalg.eig(A)
    evals_sorting_indices = np.argsort(evals)[::-1]

    # plot the eigenvalues
    evals_plotting_q = [1 / (1 + ((sig_r/sig_b)**2 / evals_q[ind])) for ind in evals_sorting_indices_q]
    evals_plotting = [evals[ind] for ind in evals_sorting_indices]
    eval_indices = [i+1 for i in range(len(evals_plotting))]

    plt.plot(eval_indices, evals_plotting_q, label=r"$(1 + \alpha/\lambda_q)^{-1}$", zorder=2, linestyle=(0, (4, 5)))
    plt.plot(eval_indices, evals_plotting, label=r"$\lambda_a$", zorder=1, alpha=0.8)
    plt.title(r"Comparison of Eigenvalues of Matrices $A$ and $Q$")
    plt.ylabel(r"$\lambda$")
    plt.xlabel("Eigenvalue Index (Sorted in Descending Order)")
    plt.grid(alpha=0.8)
    plt.legend()
    plt.show()
    plt.close()

    # plot the eigenvectors
    for i in range(num_evals_to_plot):
        plt.plot(x, evecs[:, evals_sorting_indices[i]], label=rf'$\lambda_{{{i+1}}}$='+format(evals[evals_sorting_indices[i]], ".3g"))

    # add data to the plot of eigenvectors
    plt.title(r"Eigenvectors of Matrix $A$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Magnitude of Eigenvector Component at $x=x_j$")
    plt.grid(alpha=0.8)
    plt.legend()
    plt.show()

    return


# function for part c
###########################################################
def part_c(Lx=1, Ld=0.2, J=32, num_evals_to_plot=6, sig_r=1, sig_b=2, c=1):
    # create correlation metric
    dx=2*Lx/J
    x = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    x1=1
    y = np.linspace(dx*(-J/2+1), dx*(J/2), J)
    X, Y = np.meshgrid(x, y)
    Q = correlation_metric(X, Y, Ld)

    # construct matrix A
    R = sig_r**2*np.eye(J)
    D = sig_b*np.eye(J)
    P = np.linalg.matmul(np.linalg.matmul(D, Q), D)
    A = np.matmul(P, np.linalg.inv(P+R))

    # solve the eigenproblem
    evals_q, evecs_q = np.linalg.eig(Q)
    evals_sorting_indices_q = np.argsort(evals_q[::-1])
    evals, evecs = np.linalg.eig(A)
    evals_sorting_indices = np.argsort(evals)[::-1]

    # # plot the eigenvalues
    # evals_plotting_q = [(1 + ((sig_r/sig_b)**2 / evals_q[ind]))**(-1) for ind in evals_sorting_indices_q]
    # evals_plotting = [(1 + ((sig_r/sig_b)**2 / evals[ind]))**(-1) for ind in evals_sorting_indices]
        
    # plt.plot(x, evals_plotting_q, label=r"$(1 + \alpha/\lambda_q)^{-1}$")
    # plt.plot(x, evals_plotting, label=r"$\lambda_a$")
    # plt.title(r"Comparison of Eigenvalues of Matrices $A$ and $Q$")
    # plt.ylabel(r"$\lambda$")
    # plt.xlabel("Eigenvalue Index (Sorted in Descending Order)")
    # plt.grid(alpha=0.8)
    # plt.legend()
    # plt.show()
    # plt.close()

    y = np.cos(c*np.pi*x)
    d = A @ y

    # plot the eigenvectors
    # for i in range(num_evals_to_plot):
    #     plt.plot(x, evecs[:, evals_sorting_indices[i]], label=rf'$\lambda_{{{i+1}}}$='+format(evals[evals_sorting_indices[i]], ".3g"))

    plt.plot(x, y, label=r"$y$")
    plt.plot(x, d, label=r"$d$")

    # add data to the plot of eigenvectors
    plt.title(rf"Innovation Compared to Imposed Background $y=cos(c\pi x)$, $c={{{c}}}$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$ and $d$")
    plt.grid(alpha=0.8)
    plt.legend()
    plt.show()

    return


# correlation function
###########################################################
def correlation_metric(x,y,Ld):
    return np.exp(-1/2 * (x-y)**2 / (Ld**2))


###########################################################
if __name__ == '__main__':
    Lx=1
    Ld=0.2
    J=32

    do_a=False
    if do_a:
        part_a(Lx,Ld,J)

    do_b=False
    if do_b:
        part_b()

    do_c=True
    if do_c:
        for c in [1, 3, 5, 6, 10]:
            part_c(c=c, Ld=0.05)