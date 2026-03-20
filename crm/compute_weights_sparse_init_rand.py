import numpy as np


def compute_weights_sparse_init_rand(C_xx, C_yy, C_xy, D_xy, *, f=1, mu=float('inf'), step_size=0, sparsity=float('nan'), theta_x=0, theta_y=0, gamma=0, chlsky=True, k=0, max_iter=10000, tol=1e-6):
    """COMPUTE_WEIGHTS_SPARSE_INIT_RAND

    Compute sparse CRM weight vectors (w_x, w_y) using a
    gradient-based update rule with optional automatic sparsity selection.

    This function:
    1. Initializes (w_x, w_y) using compute_weights_init_rand
    2. Chooses L1-norm constraints (theta_x, theta_y) if not provided
    3. Iteratively updates w_x, w_y using proximal gradient ascent on
       the combined objective  w_x' * (C_xy + mu * D_xy) * w_y
    4. Applies soft-thresholding to enforce L1 constraints
    5. Renormalizes w_x, w_y in the C_xx / C_yy metric

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
        Confound penalty weight (default: inf -> auto).
        inf uses lambda3 from the dense init, matching the
        trade-off found by fminsearch in compute_weights_init_rand.
    step_size : float, optional
        Gradient step size (default: 0 -> auto).
        0 computes 1 / sigma_max(C_xy + mu * D_xy).
    sparsity : float, optional
        Fraction of zero entries (0-1). E.g. 0.9 = 90% zeros.
        Overridden by explicit theta_x/theta_y if provided.
        NaN (default) -> auto-select via choose_sparsity().
    theta_x : float, optional
        L1 constraint for w_x (0 -> use sparsity or auto).
    theta_y : float, optional
        L1 constraint for w_y (0 -> use sparsity or auto).
    gamma : float, optional
        Ridge regularization added to C_xx, C_yy (default: 0).
    chlsky : bool, optional
        Whether to use Cholesky-based solver for init (default: True).
    k : int, optional
        Random seed index for initialization (default: 0).
    max_iter : int, optional
        Maximum number of gradient iterations (default: 10000).
    tol : float, optional
        Stopping tolerance based on change in w_x, w_y (default: 1e-6).

    Sparsity priority: explicit theta > sparsity ratio > auto.

    Returns
    -------
    w_x : ndarray
        Sparse canonical vector for X.
    w_y : ndarray
        Sparse canonical vector for Y.

    Notes
    -----
    The gradient uses the same combined matrix as the dense solver:
        grad = (C_xy + mu * D_xy) * w
    With mu = lambda3 (the default), both solvers optimize the same
    objective -- dense via eigendecomposition, sparse via proximal gradient.

    Sparsity is determined by (in order of priority):
      1. Explicit theta_x / theta_y (if nonzero)
      2. sparsity ratio (if not NaN)
      3. Auto via choose_sparsity() with a warning -- uses the universal
         threshold heuristic k = min(10%, 2/sqrt(p)) * p non-zero entries
         (Donoho & Johnstone, 1994, "Ideal spatial adaptation by wavelet
         shrinkage", Biometrika 81(3); applied to CCA in Witten et al.,
         2009, "A penalized matrix decomposition", Biostatistics 10(3)).

    Soft-thresholding is applied using apply_threshold(), which finds
      the shrinkage delta such that ||w||_1 <= theta (pre-normalization).

    After each update, w_x and w_y are normalized so that:
        w_x' * (C_xx + gamma*I) * w_x = 1
        w_y' * (C_yy + gamma*I) * w_y = 1

    The algorithm uses alternating proximal gradient (block coordinate
      ascent): w_x is fully updated before its new value is used to
      compute the gradient for w_y. This follows Witten et al. (2009).

    This function computes *one* sparse solution. For multiple random
      starts, use compute_weights_sparse_multi_rand().

    See Also
    --------
    compute_weights_init_rand
    compute_weights_sparse_multi_rand
    choose_sparsity
    """
    raise NotImplementedError("TODO: port from MATLAB")
