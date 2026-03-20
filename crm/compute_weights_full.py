import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve, eig


def compute_weights_full(C_xx, C_yy, C_xy, D_xy):
    """
    Python version of:
      [r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy] = compute_weights_full(C_xx, C_yy, C_xy, D_xy)

    Returns
    -------
    r_wx : (n, k) ndarray
    r_wy : (n, k) ndarray
    r_lam : (k,) ndarray
    wxcxywy : (k,) ndarray
    wxdxywy : (k,) ndarray
    wxcxxwx : (k,) ndarray
    wycyywy : (k,) ndarray
    """
    C_xx = np.array(C_xx, dtype=float, copy=True)
    C_yy = np.array(C_yy, dtype=float, copy=True)
    C_xy = np.array(C_xy, dtype=float, copy=True)
    D_xy = np.array(D_xy, dtype=float, copy=True)

    n = C_xx.shape[0]
    if not (C_xx.shape == C_yy.shape == C_xy.shape == D_xy.shape == (n, n)):
        raise ValueError("All inputs must be square matrices of the same size.")

    list_lbd3s = []
    list_wx = []
    list_wy = []
    list_constraint1 = []  # w_x' D_xy w_y
    list_constraint2 = []  # w_x' C_xx w_x
    list_constraint3 = []  # w_y' C_yy w_y
    list_correlation = []  # w_x' C_xy w_y

    ratio = norm(C_xy, "fro") / norm(D_xy, "fro") if norm(D_xy, "fro") != 0 else np.inf

    if ratio > 10:
        # Normal CCA branch
        # M = inv(C_xx) * C_xy * inv(C_yy) * C_xy'
        # Construct via linear solves
        Y = solve(C_yy, C_xy.T, assume_a='sym')           # inv(C_yy) * C_xy'
        M = solve(C_xx, C_xy @ Y, assume_a='sym')         # inv(C_xx) * (C_xy * inv(C_yy) * C_xy')
        eigvals, W = eig(M)                                # dense eig (possibly complex)
        # Order is unspecified; keep MATLAB's “as returned” behavior (loop all columns)
        for i in range(n):
            w_x = W[:, i].real
            denom = np.sqrt(w_x @ (C_xx @ w_x))
            if denom == 0:
                continue
            w_x = w_x / denom

            lam_i = eigvals[i]
            # Robust sqrt for possibly slightly negative/complex values
            if abs(lam_i.imag) < 1e-10:
                val = max(lam_i.real, 0.0)
                s = np.sqrt(val)
            else:
                s = np.sqrt(abs(lam_i))  # fall back to magnitude

            w_y = solve(C_yy, C_xy.T @ w_x, assume_a='sym') / (s if s != 0 else 1.0)

            list_lbd3s.append(0.0)
            list_wx.append(w_x)
            list_wy.append(w_y)
            list_constraint1.append(float(w_x @ (D_xy @ w_y)))
            list_constraint2.append(float(w_x @ (C_xx @ w_x)))
            list_constraint3.append(float(w_y @ (C_yy @ w_y)))
            list_correlation.append(float(w_x @ (C_xy @ w_y)))
    else:
        # Scan through a range of lambda3 to get candidate solutions
        maxrange = 10.0 * ratio
        L = 20001  # matches MATLAB: drange = maxrange/10000, range from -max to +max
        lbd3_range = np.linspace(-maxrange, maxrange, L)
        step = lbd3_range[1] - lbd3_range[0]

        constraint = np.zeros((n, L), dtype=float)

        for l_idx, lbd3 in enumerate(lbd3_range):
            A = C_xy + lbd3 * D_xy
            Y = solve(C_yy, A.T, assume_a='sym')          # inv(C_yy) * A'
            M = solve(C_xx, A @ Y, assume_a='sym')        # inv(C_xx) * A * inv(C_yy) * A'

            eigvals, W = eig(M)

            for i in range(n):
                w_x = W[:, i].real
                denom = np.sqrt(w_x @ (C_xx @ w_x))
                if denom == 0:
                    continue
                w_x = w_x / denom

                lam_i = eigvals[i]
                if abs(lam_i.imag) < 1e-10:
                    val = max(lam_i.real, 0.0)
                    s = np.sqrt(val)
                else:
                    s = np.sqrt(abs(lam_i))

                w_y = solve(C_yy, A.T @ w_x, assume_a='sym') / (s if s != 0 else 1.0)

                cval = float(w_x @ (D_xy @ w_y))
                constraint[i, l_idx] = cval

                if l_idx > 0:
                    prev = constraint[i, l_idx - 1]
                    flip = (np.sign(cval) * np.sign(prev) < 0)
                    dif = abs(abs(cval) - prev) / step if step != 0 else np.inf
                    if flip and (dif < 10) and (abs(cval) < 0.1):
                        list_lbd3s.append(float(lbd3))
                        list_wx.append(w_x)
                        list_wy.append(w_y)
                        list_constraint1.append(float(w_x @ (D_xy @ w_y)))   # ~0
                        list_constraint2.append(float(w_x @ (C_xx @ w_x)))    # ~1
                        list_constraint3.append(float(w_y @ (C_yy @ w_y)))    # ~1
                        list_correlation.append(float(w_x @ (C_xy @ w_y)))    # max-ish

    # If nothing was collected, return empties in consistent shapes
    if len(list_lbd3s) == 0:
        return (np.zeros((n, 0)), np.zeros((n, 0)),
                np.zeros((0,)), np.zeros((0,)),
                np.zeros((0,)), np.zeros((0,)), np.zeros((0,)))

    # Sort by correlation descending
    idx = np.argsort(-np.array(list_correlation))
    r_lam = np.array(list_lbd3s)[idx]
    r_wx = np.column_stack([list_wx[j] for j in idx])
    r_wy = np.column_stack([list_wy[j] for j in idx])
    wxcxywy = np.array(list_correlation)[idx]
    wxdxywy = np.array(list_constraint1)[idx]
    wxcxxwx = np.array(list_constraint2)[idx]
    wycyywy = np.array(list_constraint3)[idx]  # corrected to use constraint3

    return r_wx, r_wy, r_lam, wxcxywy, wxdxywy, wxcxxwx, wycyywy
