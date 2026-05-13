"""
Numba adaptive RK45 (DOPRI5) integrator for the eccentric EOB dynamics.
Zero Python dispatch per step. Proper mixed atol+rtol error scaling.
"""

import math
import numpy as np
from numba import njit


# Dormand-Prince RK45 (DOPRI5) Butcher tableau
_a21 = 1.0 / 5.0
_a31 = 3.0 / 40.0;    _a32 = 9.0 / 40.0
_a41 = 44.0 / 45.0;   _a42 = -56.0 / 15.0;   _a43 = 32.0 / 9.0
_a51 = 19372.0/6561.0; _a52 = -25360.0/2187.0; _a53 = 64448.0/6561.0; _a54 = -212.0/729.0
_a61 = 9017.0/3168.0;  _a62 = -355.0/33.0;     _a63 = 46732.0/5247.0; _a64 = 49.0/176.0; _a65 = -5103.0/18656.0
_a71 = 35.0/384.0;     _a73 = 500.0/1113.0;    _a74 = 125.0/192.0;    _a75 = -2187.0/6784.0; _a76 = 11.0/84.0

# Error coefficients (b5 - b4)
_e1 = 71.0/57600.0;    _e3 = -71.0/16695.0;    _e4 = 71.0/1920.0
_e5 = -17253.0/339200.0; _e6 = 22.0/525.0;      _e7 = -1.0/40.0

_c2 = 1.0/5.0; _c3 = 3.0/10.0; _c4 = 4.0/5.0; _c5 = 8.0/9.0


@njit(cache=True, fastmath=True)
def integrate_adaptive(rhs_func, t0, y0, t_end, params,
                       rtol=1e-8, atol=1e-9, h_init=1.0,
                       max_steps=100000, r_stop=2.5):
    """
    Adaptive RK45 (DOPRI5) integration with proper error scaling.

    Error norm: err_i = |delta_i| / (atol + rtol * max(|y_i|, |y_new_i|))
    Step accepted when max(err_i) <= 1.

    Parameters
    ----------
    rhs_func : callable(t, y, params) -> array
    t0 : float
    y0 : 1D array
    t_end : float
    params : 1D array (packed parameters)
    rtol, atol : float
    h_init : float
    max_steps : int
    r_stop : float — stop when r < r_stop

    Returns
    -------
    t_arr : 1D array of times
    y_arr : 2D array (n_steps+1, n_vars)
    """
    n = len(y0)
    t_out = np.empty(max_steps + 1)
    y_out = np.empty((max_steps + 1, n))

    t_out[0] = t0
    for i in range(n):
        y_out[0, i] = y0[i]

    t = t0
    y = y0.copy()
    h = h_init
    n_steps = 0

    safety = 0.9
    h_min = 1e-8
    h_max = 500.0
    fac_max = 5.0
    fac_min = 0.2

    # Scratch arrays (pre-allocate to avoid allocation per step)
    ytmp = np.empty(n)
    k1 = np.empty(n)
    k2 = np.empty(n)
    k3 = np.empty(n)
    k4 = np.empty(n)
    k5 = np.empty(n)
    k6 = np.empty(n)
    k7 = np.empty(n)
    y_new = np.empty(n)

    # First RHS evaluation
    k1_vals = rhs_func(t, y, params)
    for i in range(n):
        k1[i] = k1_vals[i]

    while t < t_end and n_steps < max_steps:
        if t + h > t_end:
            h = t_end - t
        if h < h_min:
            h = h_min

        # --- DOPRI5 stages ---
        for i in range(n):
            ytmp[i] = y[i] + h * _a21 * k1[i]
        k2_v = rhs_func(t + _c2 * h, ytmp, params)
        for i in range(n):
            k2[i] = k2_v[i]

        for i in range(n):
            ytmp[i] = y[i] + h * (_a31 * k1[i] + _a32 * k2[i])
        k3_v = rhs_func(t + _c3 * h, ytmp, params)
        for i in range(n):
            k3[i] = k3_v[i]

        for i in range(n):
            ytmp[i] = y[i] + h * (_a41 * k1[i] + _a42 * k2[i] + _a43 * k3[i])
        k4_v = rhs_func(t + _c4 * h, ytmp, params)
        for i in range(n):
            k4[i] = k4_v[i]

        for i in range(n):
            ytmp[i] = y[i] + h * (_a51 * k1[i] + _a52 * k2[i] + _a53 * k3[i] + _a54 * k4[i])
        k5_v = rhs_func(t + _c5 * h, ytmp, params)
        for i in range(n):
            k5[i] = k5_v[i]

        for i in range(n):
            ytmp[i] = y[i] + h * (_a61 * k1[i] + _a62 * k2[i] + _a63 * k3[i] + _a64 * k4[i] + _a65 * k5[i])
        k6_v = rhs_func(t + h, ytmp, params)
        for i in range(n):
            k6[i] = k6_v[i]

        # 5th order solution
        for i in range(n):
            y_new[i] = y[i] + h * (_a71 * k1[i] + _a73 * k3[i] + _a74 * k4[i] + _a75 * k5[i] + _a76 * k6[i])

        k7_v = rhs_func(t + h, y_new, params)
        for i in range(n):
            k7[i] = k7_v[i]

        # --- Error estimate (mixed atol + rtol scaling) ---
        err_max = 0.0
        for i in range(n):
            sc = atol + rtol * max(abs(y[i]), abs(y_new[i]))
            err_i = abs(h * (_e1 * k1[i] + _e3 * k3[i] + _e4 * k4[i] + _e5 * k5[i] + _e6 * k6[i] + _e7 * k7[i])) / sc
            if err_i > err_max:
                err_max = err_i

        # --- Step acceptance ---
        if err_max <= 1.0:
            # Accept
            t = t + h
            for i in range(n):
                y[i] = y_new[i]
            n_steps += 1
            t_out[n_steps] = t
            for i in range(n):
                y_out[n_steps, i] = y[i]

            # FSAL: k1 for next step = k7 of this step
            for i in range(n):
                k1[i] = k7[i]

            # Termination checks
            if y[0] < r_stop:
                break
            if y[4] < -0.01 or y[4] > 0.999:
                break

        # --- Step size adjustment ---
        if err_max > 1e-30:
            h_new = h * min(fac_max, max(fac_min, safety * (1.0 / err_max) ** 0.2))
        else:
            h_new = h * fac_max

        h = max(h_min, min(h_max, h_new))

    return t_out[:n_steps + 1], y_out[:n_steps + 1]
