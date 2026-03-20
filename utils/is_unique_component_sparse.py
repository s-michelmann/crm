import numpy as np


def is_unique_component_sparse(
    w_new: np.ndarray,
    W_existing: np.ndarray | None,
    tol: float | None = None,
    rnz: float = 0.15,
) -> bool:
    """Determines whether w_new is unique relative to the columns of W_existing,
    using a strict elementwise test that is invariant to global sign.

    A vector is considered a duplicate if it matches an existing vector
    up to sign, i.e. norm(w_new - w_old) < tol OR norm(w_new + w_old) < tol.

    Parameters
    ----------
    w_new : np.ndarray
        New weight vector to test for uniqueness.
    W_existing : np.ndarray or None
        Matrix whose columns are existing weight vectors. Pass None or an
        empty array if there are no existing vectors yet.
    tol : float, optional
        Distance tolerance for declaring a duplicate. Default is
        sqrt(2 / (rnz * len(w_new))).
    rnz : float, optional
        Target ratio of non-zero entries (medium sparsity). Default is 0.15.

    Returns
    -------
    bool
        True if w_new is unique (no duplicate found), False otherwise.
    """
    raise NotImplementedError("TODO: port from MATLAB")
