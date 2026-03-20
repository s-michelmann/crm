import numpy as np


def compute_weights_sparse_multi_rand(C_xx, C_yy, C_xy, D_xy, *, f=1, mu=float('inf'), step_size=0, sparsity=float('nan'), theta_x=0, theta_y=0, gamma=0, chlsky=True, max_iter=10000, tol=1e-6, n_init=10, rank_ratio_thresh=float('nan')):
    """COMPUTE_WEIGHTS_SPARSE_MULTI_RAND

    Run multiple random initializations of the sparse CCA/CRM solver
    (compute_weights_sparse_init_rand) and return the best sparse
    canonical vectors based on the achieved cross-covariance correlation.

    This function is a wrapper that:
        1. Calls compute_weights_sparse_init_rand() n_init times
        2. Uses different random seeds (k = 0,1,2,...)
        3. Applies user-specified or automatically chosen sparsity levels
        4. Stores all sparse solutions (w_x, w_y)
        5. Computes the canonical correlation wx' * C_xy * wy
        6. Selects and returns the best sparse solution

    It is useful when the sparse optimization landscape is non-convex
    and different random starts may converge to different local optima.

    Syntax
    ------
    w_x_best, w_y_best, Wxs, Wys, corrs = \\
        compute_weights_sparse_multi_rand(C_xx, C_yy, C_xy, D_xy)

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
        Confound penalty weight (inf -> auto from lambda3).
    step_size : float, optional
        Gradient step size (0 -> auto from spectral norm).
    sparsity : float, optional
        Fraction of zero entries (0-1). NaN (default) -> auto.
    theta_x : float, optional
        L1 constraint for w_x (0 -> auto-select).
    theta_y : float, optional
        L1 constraint for w_y (0 -> auto-select).
    gamma : float, optional
        Ridge regularization strength (default: 0).
    chlsky : bool, optional
        Use Cholesky-based initialization (default: True).
    max_iter : int, optional
        Maximum number of gradient iterations (default: 10000).
    tol : float, optional
        Convergence tolerance (default: 1e-6).
    n_init : int, optional
        Number of random initializations (default: 10).
    rank_ratio_thresh : float, optional
        Controls how the best init is selected (NaN -> auto = 2).
        See compute_weights_multi_rand for full description.

    Returns
    -------
    w_x_best : ndarray
        Best sparse canonical vector for X.
    w_y_best : ndarray
        Best sparse canonical vector for Y.
    Wxs : ndarray, shape (p, n_init)
        Matrix of all sparse w_x solutions.
    Wys : ndarray, shape (q, n_init)
        Matrix of all sparse w_y solutions.
    corrs : ndarray, shape (n_init,)
        Vector of correlations wx' * C_xy * wy.

    Notes
    -----
    - Each initialization uses seed (k-1), matching the dense solver's
      behavior and ensuring reproducibility.
    - Sparsity is enforced via soft-thresholding inside
      compute_weights_sparse_init_rand.
    - If theta_x/theta_y are zero, sparsity levels are chosen
      automatically using choose_sparsity(), following Witten et al.
    - The "best" solution is selected adaptively based on the rank
      ratio of C_xy and D_xy (see rank_ratio_thresh). When the signal
      subspace is large relative to the confound, confound-first
      selection is used. When data is limited and subspaces overlap,
      diff-based selection (signal - |confound|) is used instead.
    - All solutions are returned so the user can inspect variability
      across random starts or perform stability analysis.

    See Also
    --------
    compute_weights_sparse_init_rand
    compute_weights_init_rand
    compute_weights_multi_rand
    choose_sparsity
    """
    raise NotImplementedError("TODO: port from MATLAB")
