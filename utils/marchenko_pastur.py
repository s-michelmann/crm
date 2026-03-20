import numpy as np


def marchenko_pastur(n, p, eigenvalues, sigma2=None):
    """Marchenko-Pastur threshold for eigenvalue significance.

    Computes the upper edge of the Marchenko-Pastur distribution, which
    gives the maximum eigenvalue expected from a random matrix with i.i.d.
    entries of variance sigma^2.  Eigenvalues exceeding this threshold are
    considered signal (not noise).

    Parameters
    ----------
    n : int
        Number of samples (observations).
    p : int
        Number of features (variables).
    eigenvalues : array_like
        Vector of eigenvalues to test.
    sigma2 : float, optional
        Noise variance.  Default: median of *eigenvalues* (Marchenko-Pastur
        median approximation).

    Returns
    -------
    threshold : float
        Upper edge lambda_+ of the MP distribution.
    is_significant : np.ndarray of bool
        True where eigenvalues > threshold.

    Notes
    -----
    gamma = p / n
    lambda_+ = sigma^2 * (1 + sqrt(gamma))^2
    lambda_- = sigma^2 * (1 - sqrt(gamma))^2   (not returned)

    References
    ----------
    Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets
    of random matrices", Mathematics of the USSR-Sbornik 1(4), 457-483.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if sigma2 is None:
        sigma2 = float(np.median(eigenvalues))

    gamma = p / n
    threshold = sigma2 * (1 + np.sqrt(gamma)) ** 2
    is_significant = eigenvalues > threshold

    return threshold, is_significant
