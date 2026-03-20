import numpy as np


def compute_weights_sparse(C_xx, C_yy, C_xy, D_xy, *, f=1, mu=float('inf'), step_size=0, sparsity=float('nan'), theta_x=0, theta_y=0, gamma=0, max_iter=10000, tol=0.001):
    """COMPUTE_WEIGHTS_SPARSE  Sparse CRM (legacy interface).

    [w_x, w_y] = compute_weights_sparse(C_xx, C_yy, C_xy, D_xy)
    [w_x, w_y] = compute_weights_sparse(..., f=2, gamma=0.01, mu=Inf, ...)

    Prefer compute_weights_sparse_init_rand for new code (supports random
    init, automatic sparsity selection, Cholesky path).

    mu defaults to lambda3 from the dense init (Inf -> auto).
    step_size defaults to 1/sigma_max(C_xy + mu*D_xy) (0 -> auto).

    Parameters
    ----------
    C_xx : ndarray, shape (p, p)
        SPD covariance matrix for X.
    C_yy : ndarray, shape (q, q)
        SPD covariance matrix for Y.
    C_xy : ndarray, shape (p, q)
        Cross-covariance matrix.
    D_xy : ndarray, shape (p, q)
        Confound matrix (to be minimized).
    f : int, optional
        Number of canonical components (default: 1).
    mu : float, optional
        Confound penalty weight (default: inf -> auto, use lambda3).
    step_size : float, optional
        Gradient step size (default: 0 -> auto, 1/L).
    sparsity : float, optional
        Fraction of zero entries (0-1). E.g. 0.9 = 90% zeros.
        NaN (default) -> auto-select via choose_sparsity().
    theta_x : float, optional
        L1 constraint for w_x (0 -> use sparsity or auto).
    theta_y : float, optional
        L1 constraint for w_y (0 -> use sparsity or auto).
    gamma : float, optional
        Ridge regularization strength added to C_xx, C_yy (default: 0).
    max_iter : int, optional
        Maximum number of gradient iterations (default: 10000).
    tol : float, optional
        Stopping tolerance based on change in w_x, w_y (default: 0.001).

    Returns
    -------
    w_x : ndarray
        Sparse canonical weight vector for X.
    w_y : ndarray
        Sparse canonical weight vector for Y.
    """
    raise NotImplementedError("TODO: port from MATLAB")
