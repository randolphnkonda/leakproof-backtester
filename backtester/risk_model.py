"""Covariance estimation.

A sample covariance over k securities and T observations has k(k+1)/2 free
parameters. When T is not substantially larger than k the estimate is dominated by
noise, and minimum-variance optimisation inverts the matrix, amplifying that noise
into unstable weights.

Ledoit-Wolf shrinkage pulls the sample estimate toward a scaled identity target at
an analytically optimal intensity, accepting bias in exchange for a large reduction
in variance. See run_optimizer_demo.py for a quantitative comparison.
"""
from __future__ import annotations

import numpy as np


def sample_cov(R: np.ndarray) -> np.ndarray:
    """Return the sample covariance of a (T, k) return matrix."""
    return np.cov(R, rowvar=False, ddof=1)


def ledoit_wolf_cov(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the Ledoit-Wolf covariance estimate and its shrinkage intensity."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(R)
    return lw.covariance_, float(lw.shrinkage_)


def estimate_covariance(R: np.ndarray, method: str = "ledoit_wolf") -> np.ndarray:
    """Estimate a covariance matrix from simple returns.

    Args:
        R: Return matrix of shape (T, k).
        method: "ledoit_wolf" or "sample".

    Returns:
        Covariance matrix of shape (k, k). Falls back to a diagonal variance matrix
        when there is too little data to estimate cross-sectional structure.

    Raises:
        ValueError: If the method is not recognised.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] < 3 or R.shape[1] < 1:
        v = np.var(R, axis=0, ddof=1) if R.ndim == 2 and R.shape[0] > 1 else np.array([1.0])
        return np.diag(np.clip(v, 1e-12, None))
    if method == "sample":
        return sample_cov(R)
    if method == "ledoit_wolf":
        return ledoit_wolf_cov(R)[0]
    raise ValueError(f"unknown covariance method: {method}")
