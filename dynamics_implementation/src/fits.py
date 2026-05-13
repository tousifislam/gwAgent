"""
Calibration fits for the EOB Hamiltonian and GSF amplitude corrections.

Translated from:
  pyseobnr/eob/fits/fits_Hamiltonian.py  (a6_NS, dSO)
  pyseobnr/eob/fits/GSF_fits.py          (GSF_amplitude_fits)
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def a6_NS(nu):
    """Non-spinning calibration coefficient a6 at 5PN in the A potential."""
    return (41.7877875e0 + (-3021.93382) * nu + 33414.4394 * nu**2
            + (-169019.140) * nu**3 + 329523.262 * nu**4)


@njit(cache=True, fastmath=True)
def dSO(nu, ap, am):
    """Spin-orbit calibration coefficient dSO at 4.5PN."""
    return (
        -7.71251231383957 * am**3
        - 17.2294679794015 * am**2 * ap
        - 238.430383378296 * am**2 * nu
        + 69.5461667822545 * am**2
        - 10.5225438990315 * am * ap**2
        + 362.767393298729 * am * ap * nu
        - 85.8036338010274 * am * ap
        - 1254.66845939312 * am * nu**2
        + 472.431937787377 * am * nu
        - 39.742317057316 * am
        - 7.58458103577458 * ap**3
        - 42.7601129678844 * ap**2 * nu
        + 18.1783435552183 * ap**2
        - 201.905934468847 * ap * nu**2
        - 90.5790079104259 * ap * nu
        + 49.6299175121658 * ap
        + 478.546231305475 * nu**3
        + 679.521769948995 * nu**2
        - 177.334831768076 * nu
        - 37.6897780220529
    )


@njit(cache=True, fastmath=True)
def GSF_amplitude_fits_numba(nu):
    """
    GSF fit coefficients for waveform mode amplitudes.

    Returns a flat array of 21 coefficients (nu * raw_coeffs).
    Index mapping:
      0: h22_v8,  1: h22_v10,  2: h33_v8,  3: h33_v10,
      4: h21_v6,  5: h21_v8,   6: h21_v10,
      7: h44_v6,  8: h44_v8,   9: h44_v10,
      10: h55_v4, 11: h55_v6,  12: h55_v8,
      13: h32_v6, 14: h32_v8,  15: h32_v10, 16: h32_vlog10,
      17: h43_v4, 18: h43_v6,  19: h43_v8,  20: h43_vlog8,
    """
    raw = np.empty(21)
    raw[0] = 21.2
    raw[1] = -411.0
    raw[2] = 12.0
    raw[3] = -215.0
    raw[4] = 1.65
    raw[5] = 26.5
    raw[6] = 80.0
    raw[7] = -3.56
    raw[8] = 15.6
    raw[9] = -216.0
    raw[10] = -2.61
    raw[11] = 1.25
    raw[12] = -35.7
    raw[13] = 0.333
    raw[14] = -6.5
    raw[15] = 98.0 - (1312549797426453052.0 / 176264081083715625.0) / nu
    raw[16] = (18778864.0 / 12629925.0) / nu
    raw[17] = -0.654
    raw[18] = -3.69
    raw[19] = 18.5 - (2465107182496333.0 / 460490801971200.0) / nu
    raw[20] = (174381.0 / 67760.0) / nu
    for i in range(21):
        raw[i] *= nu
    return raw
