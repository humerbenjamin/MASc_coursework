"""
Finite-difference verification for calculate_jacobians.py

Place this file in the SAME directory as calculate_jacobians.py, then run:

    python test_calculate_jacobians.py

The tests compare the analytical Jacobians against central finite differences
of the forward semi-discrete operator, the complete SSP-RK2 step, and the
discrete residual.

Important:
- The source term used here is S(u, x; k), so it is recalculated at BOTH
  RK2 stages. The imported ssp_rk2 implementation previously used a single
  precomputed S array at both stages; that is not equivalent when S depends
  on u. Therefore rk2_step() below explicitly performs the same two RK2
  stages while recalculating S at each stage.
- The first and last state values are fixed boundary values. State-Jacobian
  finite differences therefore perturb only interior state components.
"""

import os
import sys
import numpy as np

# Make the sibling solver directory importable.
HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.abspath(
    os.path.join(HERE, "..", "viscous_burgers_solver")
)
if SOLVER_DIR not in sys.path:
    sys.path.insert(0, SOLVER_DIR)

import calculate_jacobians as jac
from semi_discrete_solver import L


# ---------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------

N = 9
X = np.linspace(0.0, 0.1, N)

NU = 1.5e-5
DT = 2.0e-4

# Keep all states positive. This keeps the Godunov flux on a smooth branch
# and avoids testing at shock/rarefaction switching points.
BCS = [1.05, 0.85]
U = np.linspace(BCS[0], BCS[1], N)
U[1:-1] += np.array([0.035, -0.018, 0.022, -0.027, 0.014, -0.009, 0.006])

# Source parameters: [a, mu, sigma].
K = np.array([0.25, 0.045, 0.018], dtype=float)

# Numerical-difference step multipliers.
STATE_STEP = 1.0e-6
PARAMETER_STEP = 1.0e-6


# ---------------------------------------------------------------------
# Source function and its analytical derivatives
# ---------------------------------------------------------------------

def source_term(u, x, parameters):
    a, mu, sigma = parameters
    return (
        a * u / (np.sqrt(2.0 * np.pi) * sigma)
        * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    )


def source_derivative_u(u, x, parameters):
    a, mu, sigma = parameters
    return (
        a / (np.sqrt(2.0 * np.pi) * sigma)
        * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    )


def source_derivative_k(u, x, parameters, parameter_index):
    a, mu, sigma = parameters
    source = source_term(u, x, parameters)

    if parameter_index == 0:
        return (
            u / (np.sqrt(2.0 * np.pi) * sigma)
            * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
        )
    elif parameter_index == 1:
        return source * (x - mu) / sigma**2
    elif parameter_index == 2:
        return source * (
            -1.0 / sigma + (x - mu) ** 2 / sigma**3
        )
    else:
        raise IndexError("Unknown source parameter index")


# ---------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------

def rhs(u, parameters):
    """
    Evaluate the semi-discrete RHS.

    The boundary values are fixed, so their time derivatives are zero.
    """
    S = source_term(u, X, parameters)
    dudt = L(u, X, S, 0.0, BCs=BCS, nu=NU)

    # Enforce fixed boundary conditions
    dudt[0] = 0
    dudt[-1] = 0

    return dudt


def rk2_step(u, parameters):
    """
    One SSP-RK2 step, recalculating the state-dependent source at each stage.

    Returns both the intermediate state and the final state.
    """
    u_tilde = u + DT * rhs(u, parameters)
    u_next = 0.5 * u + 0.5 * (
        u_tilde + DT * rhs(u_tilde, parameters)
    )
    return u_tilde, u_next


def residual(u, u_next, parameters):
    """
    Discrete residual:
        R = (u_next - M(u, parameters)) / DT
    """
    _, model_next = rk2_step(u, parameters)
    return (u_next - model_next) / DT


# ---------------------------------------------------------------------
# Finite-difference utilities
# ---------------------------------------------------------------------

def finite_difference_jacobian(function, vector, columns=None, step_scale=1.0e-6):
    """
    Central finite-difference Jacobian of a vector-valued function.

    If columns is supplied, only those input components are perturbed.
    """
    vector = np.asarray(vector, dtype=float)
    base_output = np.asarray(function(vector), dtype=float).ravel()

    if columns is None:
        columns = range(vector.size)

    J = np.zeros((base_output.size, vector.size), dtype=float)

    for j in columns:
        h = step_scale * max(1.0, abs(vector[j]))
        plus = vector.copy()
        minus = vector.copy()
        plus[j] += h
        minus[j] -= h

        f_plus = np.asarray(function(plus), dtype=float).ravel()
        f_minus = np.asarray(function(minus), dtype=float).ravel()

        J[:, j] = (f_plus - f_minus) / (2.0 * h)

    return J


def finite_difference_parameter_jacobian(function, parameters):
    parameters = np.asarray(parameters, dtype=float)
    base_output = np.asarray(function(parameters), dtype=float).ravel()
    J = np.zeros((base_output.size, parameters.size), dtype=float)

    for j in range(parameters.size):
        h = PARAMETER_STEP * max(abs(parameters[j]), 1.0e-2)
        plus = parameters.copy()
        minus = parameters.copy()
        plus[j] += h
        minus[j] -= h

        f_plus = np.asarray(function(plus), dtype=float).ravel()
        f_minus = np.asarray(function(minus), dtype=float).ravel()

        J[:, j] = (f_plus - f_minus) / (2.0 * h)

    return J


def compare_jacobians(name, analytical, numerical, rtol=2.0e-5, atol=2.0e-7):
    analytical = np.asarray(analytical, dtype=float)
    numerical = np.asarray(numerical, dtype=float)

    difference = analytical - numerical
    max_abs = np.max(np.abs(difference))
    denominator = max(
        np.linalg.norm(analytical),
        np.linalg.norm(numerical),
        1.0e-14
    )
    relative_l2 = np.linalg.norm(difference) / denominator

    passed = np.allclose(
        analytical, numerical, rtol=rtol, atol=atol
    )

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Maximum absolute error : {max_abs:.6e}")
    print(f"Relative L2 error      : {relative_l2:.6e}")
    print(f"Tolerance              : rtol={rtol:.1e}, atol={atol:.1e}")
    print("RESULT                 : " + ("PASS" if passed else "FAIL"))

    return passed


# ---------------------------------------------------------------------
# Analytical Jacobians
# ---------------------------------------------------------------------

def analytical_dL_dU(u):
    return jac.dL_dU(
        u, X, NU, X[1] - X[0],
        source_derivative_u, K
    ).toarray()


def analytical_dM_dU(u, u_tilde):
    return jac.calc_dM_dU(
        u, u_tilde, X, NU, X[1] - X[0], DT,
        source_derivative_u, K
    ).toarray()


# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------

def main():
    dx = X[1] - X[0]
    interior_columns = range(1, N - 1)

    print("=" * 72)
    print("JACOBIAN FINITE-DIFFERENCE VERIFICATION")
    print("=" * 72)
    print(f"Number of grid points : {N}")
    print(f"dx                   : {dx:.6e}")
    print(f"dt                   : {DT:.6e}")
    print(f"nu                   : {NU:.6e}")
    print(f"Source parameters    : {K}")
    print("State perturbations  : interior points only (fixed endpoints)")
    print("=" * 72)

    # Base forward trajectory for this one timestep.
    u_tilde, u_next = rk2_step(U, K)

    # 1. Semi-discrete Jacobian dL/dU.
    J_L_analytical = analytical_dL_dU(U)
    J_L_numerical = finite_difference_jacobian(
        lambda v: rhs(v, K),
        U,
        columns=interior_columns,
        step_scale=STATE_STEP
    )
    pass_JL = compare_jacobians(
        "1. Semi-discrete Jacobian: dL/dU",
        J_L_analytical[:, 1:-1],
        J_L_numerical[:, 1:-1]
    )

    # 2. Full RK2 model Jacobian dM/dU.
    A_analytical = analytical_dM_dU(U, u_tilde)
    A_numerical = finite_difference_jacobian(
        lambda v: rk2_step(v, K)[1],
        U,
        columns=interior_columns,
        step_scale=STATE_STEP
    )
    pass_A = compare_jacobians(
        "2. Full RK2 model Jacobian: dM/dU",
        A_analytical[:, 1:-1],
        A_numerical[:, 1:-1]
    )

    # 3. Residual derivative with respect to U^n.
    R_U_analytical = jac.calc_dR_dU(
        jac.sp.sparse.csc_matrix(A_analytical), U, DT
    ).toarray()
    R_U_numerical = finite_difference_jacobian(
        lambda v: residual(v, u_next, K),
        U,
        columns=interior_columns,
        step_scale=STATE_STEP
    )
    pass_RU = compare_jacobians(
        "3. Residual Jacobian: dR/dU^n (holding U^(n+1) fixed)",
        R_U_analytical[:, 1:-1],
        R_U_numerical[:, 1:-1]
    )

    # 4. Residual derivative with respect to U^(n+1).
    R_Unext_analytical = jac.calc_dR_dUnp1(U, DT).toarray()
    R_Unext_numerical = finite_difference_jacobian(
        lambda v: residual(U, v, K),
        u_next,
        step_scale=STATE_STEP
    )
    pass_RUnext = compare_jacobians(
        "4. Residual Jacobian: dR/dU^(n+1)",
        R_Unext_analytical,
        R_Unext_numerical
    )

    # 5. Residual derivative with respect to source parameters.
    R_k_analytical = jac.calc_dtR_dk(
        U, u_tilde, X, DT, NU, dx,
        source_derivative_k,
        source_derivative_u,
        K
    )
    R_k_analytical = np.asarray(R_k_analytical)

    R_k_numerical = finite_difference_parameter_jacobian(
        lambda parameters: residual(U, u_next, parameters),
        K
    )
    pass_Rk = compare_jacobians(
        "5. Residual Jacobian: dR/dk (holding both state vectors fixed)",
        R_k_analytical,
        R_k_numerical
    )

    # Overall result.
    results = [pass_JL, pass_A, pass_RU, pass_RUnext, pass_Rk]

    print("\n" + "=" * 72)
    print("OVERALL RESULT")
    print("=" * 72)
    print(f"Passed: {sum(results)} / {len(results)}")
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("ONE OR MORE TESTS FAILED")
        print(
            "Inspect the individual errors above. In particular, confirm "
            "that the imported L function uses the same boundary and source "
            "conventions as the Jacobian implementation."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
