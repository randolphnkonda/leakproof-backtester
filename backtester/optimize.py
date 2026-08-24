"""Convex portfolio optimisation.

Two long-only, fully invested quadratic programs:

    minimum variance        minimise w'Sw   subject to sum(w) = 1, 0 <= w <= w_max
    maximum decorrelation   minimise w'Cw   subject to sum(w) = 1, 0 <= w <= w_max

where S is the covariance matrix and C the corresponding correlation matrix. Maximum
decorrelation weights the correlation structure only, ignoring individual variances.

Solvers are selected at call time. cvxpy is used when installed; otherwise scipy's
SLSQP solves the same program. Callers request weights, not a specific solver.
"""
from __future__ import annotations

import numpy as np


def _has_cvxpy() -> bool:
    try:
        import cvxpy  # noqa: F401
        return True
    except Exception:
        return False


def active_backend(backend: str = "auto") -> str:
    if backend == "auto":
        return "cvxpy" if _has_cvxpy() else "scipy"
    return backend


def _psd_project(M: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Symmetrise and add a ridge term to ensure positive definiteness."""
    M = 0.5 * (M + M.T)
    return M + ridge * np.eye(M.shape[0])


def _solve_cvxpy(Sigma: np.ndarray, max_weight: float) -> np.ndarray:
    import cvxpy as cp
    k = Sigma.shape[0]
    w = cp.Variable(k)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), constraints)
    prob.solve()
    if w.value is None:
        raise RuntimeError("cvxpy failed to converge")
    return np.asarray(w.value).flatten()


def _solve_scipy(Sigma: np.ndarray, max_weight: float) -> np.ndarray:
    from scipy.optimize import minimize
    k = Sigma.shape[0]
    w0 = np.ones(k) / k
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, max_weight)] * k
    res = minimize(lambda w: float(w @ Sigma @ w), w0, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"scipy SLSQP failed: {res.message}")
    return res.x


def solve_min_variance(
    Sigma: np.ndarray, max_weight: float = 1.0, backend: str = "auto"
) -> np.ndarray:
    """Solve for long-only, fully invested minimum-variance weights.

    Args:
        Sigma: Covariance matrix of shape (k, k).
        max_weight: Per-security weight cap.
        backend: "auto", "cvxpy", or "scipy".

    Returns:
        Weight vector summing to one.
    """
    k = Sigma.shape[0]
    if k == 1:
        return np.array([1.0])
    # A cap below 1/k is infeasible given the budget constraint.
    max_weight = max(max_weight, 1.0 / k)
    Sigma = _psd_project(np.asarray(Sigma, dtype=float))
    solver = active_backend(backend)
    w = _solve_cvxpy(Sigma, max_weight) if solver == "cvxpy" else _solve_scipy(Sigma, max_weight)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(k) / k


def solve_max_decorrelation(
    Sigma: np.ndarray, max_weight: float = 1.0, backend: str = "auto"
) -> np.ndarray:
    """Solve for maximum-decorrelation weights."""
    Sigma = np.asarray(Sigma, dtype=float)
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-16, None))
    C = Sigma / np.outer(d, d)
    return solve_min_variance(C, max_weight=max_weight, backend=backend)
