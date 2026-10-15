# import libraries
###########################################################
import matplotlib.pyplot as plt
import numpy as np


# main function for making plots (a)
###########################################################
def make_plots(a, b, dx):
    # make linearly spaced x
    x = np.linspace(a, b, int((b-a)/dx))

    # make distributions
    y_uniform = [uniform_dist(xi) for xi in x]
    y_triangular = [triangular_dist(xi) for xi in x]
    y_gaussian = [gaussian_dist(xi) for xi in x]
    
    # make plots
        # uniform
    plt.plot(x, y_uniform)
    plt.title("Uniform Distribution")
    plt.ylabel("p(x)")
    plt.xlabel("x")
    plt.show()
        # triangular
    plt.plot(x, y_triangular)
    plt.title("Triangular Distribution")
    plt.ylabel("p(x)")
    plt.xlabel("x")
    plt.show()
        # gaussian
    plt.plot(x, y_gaussian)
    plt.title("Gaussian Distribution")
    plt.ylabel("p(x)")
    plt.xlabel("x")
    plt.show()

    return


# main function for (c)
###########################################################
def make_uniform_dist(n, n_bins):
    x = np.random.random(n)
    x = x * (2*np.sqrt(3)) - np.sqrt(3)
    y=[]
    for xi in x:
        if (xi <= (np.sqrt(3)) and (-np.sqrt(3) <= xi)):
            y.append(1/(2*np.sqrt(3)))
        else:
            y.append(0)

    # plt.scatter(x,y,marker='.')
    plt.hist(x, bins=n_bins, range=[-np.sqrt(3),np.sqrt(3)], density=True, label="Sample Data")
    plt.plot(np.linspace(-5,5,1000), [uniform_dist(xi) for xi in np.linspace(-5,5,1000)], label="Uniform Distribution")
    plt.title(f"Uniform Distribution with {n} Samples")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()

    return

# main function for (d)
###########################################################
def make_normal_dist(n, n_bins):
    x = np.random.normal(0, 1, n)
    plt.hist(x, bins=n_bins, density=True, label="Sample Data")
    plt.plot(np.linspace(-5,5,1000), [gaussian_dist(xi) for xi in np.linspace(-5,5,1000)], label="Normal Distribution")
    plt.title(f"Normal Distribution with {n} Samples")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()

    return


# uniform dist function
###########################################################
def uniform_dist(x):
    if (-np.sqrt(3) <= x) and (x <= np.sqrt(3)):
        y = 1 / (2*np.sqrt(3))
    else:
        y=0

    return y


# triangular dist function
###########################################################
def triangular_dist(x):
    if (-np.sqrt(6) <= x) and (x <= 0):
        y = (np.sqrt(6) + x) / (6)
    elif (0 <= x) and (x <= np.sqrt(6)):
        y = (np.sqrt(6) - x) / (6)
    else:
        y=0

    return y


# gaussian function
###########################################################
def gaussian_dist(x):
    y = 1 / (np.sqrt(2*np.pi)) * np.exp(-(x**2)/2)

    return y



# block for running functions
###########################################################
if __name__ == '__main__':
    do_a=False
    do_c=True
    if do_a:
        make_plots(-4,4,0.1)
    if do_c:
        n = [10,100,1000,10000]
        n_bins = [5, 10, 20, 25]
        for i in range(len(n)):
            make_uniform_dist(n[i], n_bins[i])
            make_normal_dist(n[i], n_bins[i])