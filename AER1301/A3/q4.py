import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Grad 5-moment closure
# ---------------------------------------------------------

def grad_flux(q_star, r_star=None):
    """
    Grad 5-moment closing flux.
    """
    return 10.0 * np.asarray(q_star)


# ---------------------------------------------------------
# 2. EQMOM closure
# ---------------------------------------------------------

def eqmom_a2(q_star, r_star):
    """
    Calculate a_star^2 from the EQMOM cubic polynomial.

    The polynomial is expressed in z = a_star^2:

        2z^3 - 6z^2 + (3+r_star)z - r_star + q_star^2 + 1 = 0

    Selects the real root satisfying 0 <= z <= 1.
    """

    q_star, r_star = np.broadcast_arrays(q_star, r_star)

    a2 = np.empty(q_star.shape)

    for idx in np.ndindex(q_star.shape):

        q = q_star[idx]
        r = r_star[idx]

        coefficients = [
            2.0,
            -6.0,
            3.0 + r,
            -r + q**2 + 1.0
        ]

        roots = np.roots(coefficients)

        # Keep real roots in the physically admissible interval.
        real_roots = roots.real[
            (np.abs(roots.imag) < 1e-10)
            & (roots.real >= -1e-10)
            & (roots.real <= 1.0 + 1e-10)
        ]

        if len(real_roots) == 0:
            raise ValueError(
                f"No admissible EQMOM root for q*={q}, r*={r}"
            )

        if len(real_roots) > 1:
            raise ValueError(
                f"Multiple admissible EQMOM roots for q*={q}, r*={r}"
            )

        a2[idx] = np.clip(real_roots[0], 0.0, 1.0)

    return a2


def eqmom_flux(q_star, r_star):
    """
    EQMOM closing flux.
    """

    q_star, r_star = np.broadcast_arrays(q_star, r_star)

    a2 = eqmom_a2(q_star, r_star)

    with np.errstate(divide='ignore', invalid='ignore'):

        flux = (
            q_star**3 / (1.0 - a2)**2
            + (10.0 - 8.0*(1.0 - a2))*q_star
        )

    return flux


# ---------------------------------------------------------
# 3. Interpolative-based maximum entropy closure
# ---------------------------------------------------------

def interpolative_beta(q_star, r_star):
    """
    Calculate the parabolic mapping parameter beta.
    """

    beta = (
        3.0 - r_star
        + np.sqrt((3.0 - r_star)**2 + 8.0*q_star**2)
    ) / 4.0

    return beta


def interpolative_flux(q_star, r_star):
    """
    Interpolative-based maximum-entropy closing flux.
    """

    beta = interpolative_beta(q_star, r_star)

    with np.errstate(divide='ignore', invalid='ignore'):

        flux = (
            q_star**3 / beta**2
            + (10.0 - 8.0*beta**2)*q_star
        )

    return flux


# ---------------------------------------------------------
# 4. Main plotting function
# ---------------------------------------------------------

def plot_closing_fluxes(
    q_min,
    q_max,
    n_points=500,
    closures=None,
    ax=None
):
    """
    Plot closing fluxes s*(q*) along the curve:

        r* = 3/2 + q*^2

    Different closures are distinguished by colour.
    """

    if closures is None:
        closures = ['Grad', 'EQMOM', 'Interpolative']

    closure_functions = {
        'Grad': grad_flux,
        'EQMOM': eqmom_flux,
        'Interpolative': interpolative_flux
    }

    closure_colours = {
        'Grad': 'tab:blue',
        'EQMOM': 'tab:orange',
        'Interpolative': 'tab:green'
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    else:
        fig = ax.figure

    # Generate q* values
    q = np.linspace(q_min, q_max, n_points)

    # First curve: r* = 3/2 + q*^2
    r = 1.5 + q**2

    # Plot each closure
    for closure in closures:

        if closure not in closure_functions:
            raise ValueError(f"Unknown closure: {closure}")

        flux_function = closure_functions[closure]

        if closure == 'Grad':
            s = flux_function(q)
        else:
            s = flux_function(q, r)

        ax.plot(
            q,
            s,
            color=closure_colours[closure],
            linewidth=2,
            label=closure
        )

    ax.set_xlabel(r'$q_*$', fontsize=13)
    ax.set_ylabel(r'$s_*$', fontsize=13)

    ax.set_xlim(q_min, q_max)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()

    return fig, ax




if __name__ == '__main__':
    fig, ax = plot_closing_fluxes(
        q_min=-2,
        q_max=2
    )

    plt.show()