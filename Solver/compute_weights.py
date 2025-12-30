import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from scipy.linalg import solve


def compute_weights(C_xx, C_yy, C_xy, D_xy, f=1, gamma=None):
    """
    Python version of MATLAB function compute_weights.m
    Computes weight vectors w_x, w_y and optimal lambda3 for given covariance-like matrices.
    Usage:
      [w_x, w_y, lambda3] = compute_weights(C_xx, C_yy, C_xy, D_xy, f, gamma)

    Parameters
    ----------
    C_xx, C_yy, C_xy, D_xy : (n x n) np.ndarray
        Covariance and cross-covariance (real-valued expected).
    f : int, optional (default=1)
        Number of leading solutions to consider; the f-th is used (0-based index f-1 internally).
    gamma : float or None, optional
        If provided, applies ridge regularization: C_xx += gamma*I and C_yy += gamma*I.

    Returns
    -------
    w_x : (n,) np.ndarray
    w_y : (n,) np.ndarray
    lambda3 : float
    """

    C_xx = np.array(C_xx, dtype=float, copy=True)
    C_yy = np.array(C_yy, dtype=float, copy=True)
    C_xy = np.array(C_xy, dtype=float, copy=True)
    D_xy = np.array(D_xy, dtype=float, copy=True)

    n = C_xx.shape[0]
    if not (C_xx.shape == C_yy.shape and C_xx.shape == C_yy.T.shape and C_xy.shape == D_xy.shape):
        raise ValueError("Covariance matrices are not of correct size.")

    if f < 1 or f >= n:
        raise ValueError("f must satisfy 1 <= f < n.")

    # Regularization (mirrors: if nargin > 5)
    if gamma is not None:
        C_xx = C_xx + gamma * np.eye(n)
        C_yy = C_yy + gamma * np.eye(n)

    # Objective for lambda3 (mirrors foo2 with k=f, default 'LM')
    def foo2(lbd3, which="LM"):
        # M = inv(C_xx) * (C_xy + lbd3*D_xy) * inv(C_yy) * (C_xy + lbd3*D_xy)'
        A = C_xy + lbd3 * D_xy
        # Use linear solves instead of inverses
        # compute inv(C_yy) * A^T  as solve(C_yy, A.T)
        Y = solve(C_yy, A.T, assume_a='sym')  # shape (n, n)
        # compute M = inv(C_xx) * (A @ Y^T) since (inv(C_yy) * A^T)^T = A @ inv(C_yy)^T = A @ inv(C_yy) for SPD
        M = solve(C_xx, A @ Y.T, assume_a='sym')

        # eigen-solve: top-k by requested criterion
        # ARPACK returns unsorted; we will sort explicitly.
        vals, vecs = eigs(M, k=f, which=which)
        # Sorting rule:
        if which in ("LR", "SR"):
            order = np.argsort(-vals.real)  # descending by real part
        elif which in ("LM", "SM"):
            order = np.argsort(-np.abs(vals))  # descending by magnitude
        elif which in ("LI", "SI"):
            order = np.argsort(-np.abs(vals.imag))  # descending by imag magnitude
        else:
            order = np.arange(len(vals))
        vals = vals[order]
        vecs = vecs[:, order]

        w_x = vecs[:, f - 1].real  # take the f-th vector in the ordered set; use real part
        # Normalize with respect to C_xx
        denom = np.sqrt(w_x @ (C_xx @ w_x))
        if denom == 0:
            raise FloatingPointError("Normalization failed: w_x' C_xx w_x == 0.")
        w_x = w_x / denom

        # w_y = inv(C_yy) * (A') * w_x   (no 1/lbd here; this mirrors foo2 for the objective)
        w_y = solve(C_yy, A.T @ w_x, assume_a='sym')

        # Objective: abs(w_x' * D_xy * w_y)
        return abs(w_x @ (D_xy @ w_y))

    # Minimize foo2 over lambda3 starting at 0 (mirrors fminsearch with Nelder–Mead)
    res = minimize(lambda l: foo2(l[0], which="LM"), x0=np.array([0.0]),
                   method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    lambda3 = float(res.x[0])

    # With optimal lambda3, assemble final M and solve for the f-th eigenpair using 'LR' (largest real part)
    A = C_xy + lambda3 * D_xy
    Y = solve(C_yy, A.T, assume_a='sym')
    M = solve(C_xx, A @ Y.T, assume_a='sym')

    vals, vecs = eigs(M, k=f, which="LR")
    # Sort by real part descending and pick the f-th
    order = np.argsort(-vals.real)
    vals = vals[order]
    vecs = vecs[:, order]

    w_x = vecs[:, f - 1].real
    denom = np.sqrt(w_x @ (C_xx @ w_x))
    if denom == 0:
        raise FloatingPointError("Normalization failed: w_x' C_xx w_x == 0.")
    w_x = w_x / denom

    # lbd = sqrt(D(f,f))
    lam = float(vals[f - 1].real)
    if lam < 0 and abs(lam) < 1e-10:
        lam = 0.0  # guard tiny negative due to numerical noise
    if lam < 0:
        # If genuinely negative, the square root is not real; proceed with real part magnitude.
        lbd = np.sqrt(abs(lam))
    else:
        lbd = np.sqrt(lam)

    # w_y = -inv(C_yy) * A' / lbd * w_x
    w_y = -solve(C_yy, A.T @ (w_x / lbd), assume_a='sym')

    # Sign correction to maximize correlation
    if (w_x @ (C_xy @ w_y)) < 0:
        w_y = -w_y

    return w_x, w_y, lambda3