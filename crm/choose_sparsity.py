import numpy as np


def choose_sparsity(C_xx, C_yy, percent_zeros_x, percent_zeros_y):
    """CHOOSE_SPARSITY  Compute L1-norm targets for desired sparsity.

    percent_zeros_x, percent_zeros_y are between 0 and 1.
    Example: 0.5 means 50% zeros (50% non-zero).

    This function sets theta_x and theta_y so that the L1 norm
    after shrinkage corresponds to the desired number of non-zero
    coefficients, assuming w is normalized such that w' C w = 1.

    Parameters
    ----------
    C_xx : ndarray, shape (p, p)
        SPD covariance matrix for X.
    C_yy : ndarray, shape (q, q)
        SPD covariance matrix for Y.
    percent_zeros_x : float
        Desired fraction of zero entries in w_x (0 to 1).
    percent_zeros_y : float
        Desired fraction of zero entries in w_y (0 to 1).

    Returns
    -------
    theta_x : float
        L1-norm target for w_x.
    theta_y : float
        L1-norm target for w_y.
    """
    raise NotImplementedError("TODO: port from MATLAB")
