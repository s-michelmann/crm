import numpy as np


def compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy, *, f=1, gamma=0, chlsky=True, k=0):
    """COMPUTE_WEIGHTS_INIT_RAND

    Compute dense canonical weight vectors (w_x, w_y) for CRM using
    either a standard solver or a Cholesky-based solver, with optional
    random initialization of the confound parameter lambda3.

    This function implements the core dense CRM optimization:
        - ridge-regularizes C_xx and C_yy (gamma)
        - optionally uses Cholesky factors for numerical stability
        - searches over lambda3 using fminsearch
        - solves the generalized eigenproblem for the top component
        - enforces a consistent sign convention

    It is the base solver used by:
        compute_weights_multi_rand
        compute_weights_sparse_init_rand
        compute_weights_sparse_multi_rand

    Syntax
    ------
    w_x, w_y, lambda3 = compute_weights_init_rand(C_xx, C_yy, C_xy, D_xy)
    w_x, w_y, lambda3 = compute_weights_init_rand(..., f=2, gamma=0.01, chlsky=True, k=3)

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
        Number of canonical component (default: 1, i.e., first).
    gamma : float, optional
        Ridge regularization strength added to C_xx and C_yy (default: 0).
    chlsky : bool, optional
        If True, use Cholesky-based solver; if False, use standard solver.
    k : int, optional
        Random seed index for initializing lambda3.
        k = 0 -> deterministic start (lambda3 = 0).
        k > 0 -> random start in a heuristic range.

    Returns
    -------
    w_x : ndarray
        Canonical weight vector for X.
    w_y : ndarray
        Canonical weight vector for Y.
    lambda3 : float
        Confound parameter minimizing |w_x' * D_xy * w_y|.

    Notes
    -----
    - If the signal-to-confound ratio is large (ratio > 10), the function
      defaults to classical CCA (lambda3 = 0).
    - Otherwise, lambda3 is optimized via fminsearch using the objective:
          | w_x' * D_xy * w_y |
    - The Cholesky path avoids explicit matrix inversion and is more
      numerically stable for ill-conditioned covariance matrices.
    - The returned vectors satisfy:
          w_x' * C_xx * w_x = 1
          w_y' * C_yy * w_y = 1
    - A sign convention ensures w_x' * C_xy * w_y >= 0.

    See Also
    --------
    compute_weights_multi_rand
    compute_weights_sparse_init_rand
    compute_weights_sparse_multi_rand
    choose_sparsity
    """
    raise NotImplementedError("TODO: port from MATLAB")
