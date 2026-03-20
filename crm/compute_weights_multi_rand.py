import numpy as np


def compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy, *, f=1, gamma=0, chlsky=True, n_init=10, rank_ratio_thresh=float('nan')):
    """COMPUTE_WEIGHTS_MULTI_RAND

    Run multiple random initializations of the dense CRM solver
    (compute_weights_init_rand) and return the best solution based on
    the achieved cross-covariance correlation.

    This function is a wrapper that:
        1. Calls compute_weights_init_rand() n_init times
        2. Uses different random seeds (k = 0,1,2,...)
        3. Stores all solutions (w_x, w_y, lambda3)
        4. Computes the canonical correlation wx' * C_xy * wy
        5. Selects and returns the best solution

    It is useful when the optimization landscape is non-convex and
    different random starts may converge to different local optima.

    Syntax
    ------
    w_x_best, w_y_best, lambda3_best, Wxs, Wys, lambdas, corrs = \\
        compute_weights_multi_rand(C_xx, C_yy, C_xy, D_xy)

    w_x_best, w_y_best, lambda3_best, Wxs, Wys, lambdas, corrs = \\
        compute_weights_multi_rand(..., f=2, gamma=0.01, chlsky=True, n_init=20)

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
    gamma : float, optional
        Ridge regularization strength passed to
        compute_weights_init_rand (default: 0).
    chlsky : bool, optional
        If True, use Cholesky-based solver for initialization;
        if False, use standard solver.
    n_init : int, optional
        Number of random initializations (default: 10).
    rank_ratio_thresh : float, optional
        Controls how the best init is selected. The solver computes
        rank(C_xy) / rank(D_xy). When this ratio exceeds the threshold,
        the signal subspace is much larger than the confound subspace,
        so strict confound-first selection is used (low confound, then
        max signal). Below the threshold, the subspaces nearly overlap,
        so diff-based selection is used instead (max signal - |confound|),
        tolerating some confound for higher signal.

        NaN (default): auto-compute threshold as 2, i.e. switch to
        diff-based selection (max signal - |confound|) when
        rank(C_xy) < 2 * rank(D_xy), meaning the excess rank (signal
        directions beyond the confound) is less than the confound rank
        itself. The difference score avoids ratio blow-up when confound
        fluctuates near zero.

    Returns
    -------
    w_x_best : ndarray
        Best canonical weight vector for X.
    w_y_best : ndarray
        Best canonical weight vector for Y.
    lambda3_best : float
        Best lambda3 value returned by the solver.
    Wxs : ndarray, shape (p, n_init)
        Matrix of all w_x solutions.
    Wys : ndarray, shape (q, n_init)
        Matrix of all w_y solutions.
    lambdas : ndarray, shape (n_init,)
        Vector of all lambda3 values.
    corrs : ndarray, shape (n_init,)
        Vector of correlations wx' * C_xy * wy.

    Notes
    -----
    - Each initialization uses seed (k-1).
    - The "best" solution is selected adaptively based on the rank
      ratio of C_xy and D_xy (see rank_ratio_thresh). When the signal
      subspace is large relative to the confound, confound-first
      selection is used. When data is limited and subspaces overlap,
      diff-based selection (signal - |confound|) is used instead.
    - All solutions are returned for inspection of variability
      across random starts.

    See Also
    --------
    compute_weights_init_rand
    compute_weights_sparse_init_rand
    compute_weights_sparse_multi_rand
    """
    raise NotImplementedError("TODO: port from MATLAB")
