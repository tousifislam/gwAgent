"""
Numba-JIT GW flux computation and RR force for eccentric EOB dynamics.

Assembles the flux from:
  - waveform_modes.py  (Newtonian prefixes, tail, rho_lm)
  - ecc_corrections.py (eccentric corrections to modes and RR force)

Uses 16 modes: (2,2), (2,1), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (4,4),
               (5,2), (5,3), (5,4), (5,5), (6,6), (7,7), (8,8).

Translated from pySEOBNR waveform_ecc.pyx:
  - compute_flux_ecc  (lines 383-500)
  - RR_force_ecc      (lines 318-375)
"""

import math
import numpy as np
from numba import njit

from waveform_modes import (
    compute_newtonian_prefixes_abs,
    newtonian_multipole_abs,
    compute_tail_16mode,
    compute_tail_8mode,
    compute_tail_6mode,
    compute_tail_3mode,
    compute_rho_coeffs_16mode,
    compute_rho_coeffs_8mode,
    compute_rho_coeffs_6mode,
    compute_rho_coeffs_3mode,
    compute_rholm_single_22,
    compute_rholm_single_21,
    compute_rholm_single_31,
    compute_rholm_single_33,
    compute_rholm_single_32,
    compute_rholm_single_41,
    compute_rholm_single_42,
    compute_rholm_single_43,
    compute_rholm_single_44,
    compute_rholm_single_52,
    compute_rholm_single_53,
    compute_rholm_single_54,
    compute_rholm_single_55,
    compute_rholm_single_66,
    compute_rholm_single_77,
    compute_rholm_single_88,
    YLMS_22,
    YLMS_11,
    YLMS_13,
    YLMS_24,
    YLMS_33,
    YLMS_35,
    YLMS_44,
    YLMS_55,
    YLMS_66,
    YLMS_77,
    YLMS_88,
)
from ecc_corrections import (
    compute_ecc_mode_corrections_default,
)

PI = math.pi


# ============================================================================
# Precompute all static waveform quantities (16-mode version)
# ============================================================================
@njit(cache=True, fastmath=True)
def precompute_waveform_statics(m_1, m_2, nu, delta, a, chiS, chiA,
                                extra_PN_terms):
    """
    Precompute all waveform quantities that depend only on intrinsic parameters.

    Parameters
    ----------
    m_1, m_2 : float (component masses, m_1 + m_2 = 1)
    nu : float (symmetric mass ratio)
    delta : float (mass difference ratio)
    a : float (Kerr parameter)
    chiS, chiA : float (symmetric/antisymmetric spin)
    extra_PN_terms : bool

    Returns
    -------
    16 Newtonian multipole absolute prefixes, rho and f coefficients for
    each mode, f_vh_33_6 : float
    """
    # Newtonian prefixes
    (prefix_22, prefix_21, prefix_31, prefix_33, prefix_32,
     prefix_41, prefix_42, prefix_43, prefix_44,
     prefix_52, prefix_53, prefix_54, prefix_55,
     prefix_66, prefix_77, prefix_88) = compute_newtonian_prefixes_abs(m_1, m_2)

    # rho coefficients
    result = compute_rho_coeffs_16mode(nu, delta, a, chiS, chiA, extra_PN_terms)
    (rho_22, rho_log_22,
     rho_21, rho_log_21, f_21,
     rho_31, rho_log_31, f_31,
     rho_33, rho_log_33, f_33, f_vh_33_6,
     rho_32, rho_log_32,
     rho_41, rho_log_41, f_41,
     rho_42, rho_log_42,
     rho_43, rho_log_43, f_43,
     rho_44, rho_log_44,
     rho_52, rho_log_52,
     rho_53, rho_log_53,
     rho_54, rho_log_54,
     rho_55, rho_log_55,
     rho_66, rho_log_66,
     rho_77, rho_log_77,
     rho_88, rho_log_88) = result

    return (prefix_22, prefix_21, prefix_31, prefix_33, prefix_32,
            prefix_41, prefix_42, prefix_43, prefix_44,
            prefix_52, prefix_53, prefix_54, prefix_55,
            prefix_66, prefix_77, prefix_88,
            rho_22, rho_log_22,
            rho_21, rho_log_21, f_21,
            rho_31, rho_log_31, f_31,
            rho_33, rho_log_33, f_33, f_vh_33_6,
            rho_32, rho_log_32,
            rho_41, rho_log_41, f_41,
            rho_42, rho_log_42,
            rho_43, rho_log_43, f_43,
            rho_44, rho_log_44,
            rho_52, rho_log_52,
            rho_53, rho_log_53,
            rho_54, rho_log_54,
            rho_55, rho_log_55,
            rho_66, rho_log_66,
            rho_77, rho_log_77,
            rho_88, rho_log_88)


# ============================================================================
# Full flux computation using 16 modes
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux(omega, e, z, x, H_times_nu, nu, pphi,
                 prefix_22, prefix_21, prefix_31, prefix_33,
                 prefix_32, prefix_41, prefix_42, prefix_43, prefix_44,
                 prefix_52, prefix_53, prefix_54, prefix_55,
                 prefix_66, prefix_77, prefix_88,
                 rho_22, rho_log_22,
                 rho_21, rho_log_21, f_21,
                 rho_31, rho_log_31, f_31,
                 rho_33, rho_log_33, f_33, f_vh_33_6,
                 rho_32, rho_log_32,
                 rho_41, rho_log_41, f_41,
                 rho_42, rho_log_42,
                 rho_43, rho_log_43, f_43,
                 rho_44, rho_log_44,
                 rho_52, rho_log_52,
                 rho_53, rho_log_53,
                 rho_54, rho_log_54,
                 rho_55, rho_log_55,
                 rho_66, rho_log_66,
                 rho_77, rho_log_77,
                 rho_88, rho_log_88,
                 h22_ecc_re, h22_ecc_im,
                 h21_ecc_re, h21_ecc_im,
                 h31_ecc_re, h31_ecc_im,
                 h32_ecc_re, h32_ecc_im,
                 h33_ecc_re, h33_ecc_im,
                 h41_ecc_re, h41_ecc_im,
                 h42_ecc_re, h42_ecc_im,
                 h43_ecc_re, h43_ecc_im,
                 h44_ecc_re, h44_ecc_im,
                 h52_ecc_re, h52_ecc_im,
                 h53_ecc_re, h53_ecc_im,
                 h54_ecc_re, h54_ecc_im,
                 h55_ecc_re, h55_ecc_im,
                 h66_ecc_re, h66_ecc_im,
                 h77_ecc_re, h77_ecc_im,
                 h88_ecc_re, h88_ecc_im):
    """
    Compute GW flux using all 16 modes.

    Parameters
    ----------
    omega : float - instantaneous orbital frequency
    e, z, x : float - Keplerian parameters
    H_times_nu : float - Hamiltonian * nu
    nu : float - symmetric mass ratio
    pphi : float - orbital angular momentum
    prefix_* : float - Newtonian prefixes
    rho_*, f_* : arrays - PN coefficients
    h*_ecc_re, h*_ecc_im : float - eccentric corrections (complex)

    Returns
    -------
    flux : float - GW energy flux
    """
    if x < 0.0:
        return 0.0

    v = math.sqrt(x)
    vh3 = H_times_nu * x ** 1.5
    if vh3 <= 0.0:
        vh3 = 1e-30
    vh = vh3 ** (1.0 / 3.0)
    omega_avg = x ** 1.5
    omega2 = omega * omega
    v2 = x  # v^2 = x

    # Compute tail factors for all 16 modes
    (T_22, T_21, T_31, T_33, T_32, T_41, T_42, T_43, T_44,
     T_52, T_53, T_54, T_55, T_66, T_77, T_88) = compute_tail_16mode(
        omega_avg, H_times_nu)

    # Source terms
    source1 = (H_times_nu * H_times_nu - 1.0) / (2.0 * nu) + 1.0  # H_eff (even eps)
    source2 = v * pphi  # (odd eps)

    flux = 0.0

    # ---- (2,2) mode: l=2, m=2, epsilon=0 (even) ----
    hNewton_22 = prefix_22 * v2 ** (1.0) * YLMS_22  # (l+eps)/2 = 1
    Slm_22 = source1
    rholmPwrl_22 = compute_rholm_single_22(v, vh, nu, rho_22, rho_log_22)
    hlm_QC_22 = hNewton_22 * Slm_22 * T_22 * rholmPwrl_22
    hlm_re = hlm_QC_22 * h22_ecc_re
    hlm_im = hlm_QC_22 * h22_ecc_im
    flux += 4.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 4

    # ---- (2,1) mode: l=2, m=1, epsilon=1 (odd) ----
    hNewton_21 = prefix_21 * v2 ** (1.5) * YLMS_11  # (l+eps)/2 = 1.5
    Slm_21 = source2
    rholmPwrl_21 = compute_rholm_single_21(v, vh, nu, rho_21, rho_log_21, f_21)
    hlm_QC_21 = hNewton_21 * Slm_21 * T_21 * rholmPwrl_21
    hlm_re = hlm_QC_21 * h21_ecc_re
    hlm_im = hlm_QC_21 * h21_ecc_im
    flux += 1.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 1

    # ---- (3,1) mode: l=3, m=1, epsilon=0 (even -> source1) ----
    hNewton_31 = prefix_31 * v2 ** (1.5) * YLMS_13  # (l+eps)/2 = 3/2 = 1.5
    Slm_31 = source1
    rholmPwrl_31 = compute_rholm_single_31(v, vh, nu, rho_31, rho_log_31, f_31)
    hlm_QC_31 = hNewton_31 * Slm_31 * T_31 * rholmPwrl_31
    hlm_re = hlm_QC_31 * h31_ecc_re
    hlm_im = hlm_QC_31 * h31_ecc_im
    flux += 1.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 1

    # ---- (3,3) mode: l=3, m=3, epsilon=0 (even) ----
    hNewton_33 = prefix_33 * v2 ** (1.5) * YLMS_33  # (l+eps)/2 = 1.5
    Slm_33 = source1
    rholmPwrl_33_re, rholmPwrl_33_im = compute_rholm_single_33(
        v, vh, nu, rho_33, rho_log_33, f_33, f_vh_33_6
    )
    hlm_QC_re = hNewton_33 * Slm_33 * T_33 * rholmPwrl_33_re
    hlm_QC_im = hNewton_33 * Slm_33 * T_33 * rholmPwrl_33_im
    hlm_re = hlm_QC_re * h33_ecc_re - hlm_QC_im * h33_ecc_im
    hlm_im = hlm_QC_re * h33_ecc_im + hlm_QC_im * h33_ecc_re
    flux += 9.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 9

    # ---- (3,2) mode: l=3, m=2, epsilon=1 (odd -> source2) ----
    hNewton_32 = prefix_32 * v2 ** (2.0) * YLMS_22  # (l+eps)/2 = (3+1)/2 = 2; ylms[2][2]
    Slm_32 = source2
    rholmPwrl_32 = compute_rholm_single_32(v, vh, nu, rho_32, rho_log_32)
    hlm_QC_32 = hNewton_32 * Slm_32 * T_32 * rholmPwrl_32
    hlm_re = hlm_QC_32 * h32_ecc_re
    hlm_im = hlm_QC_32 * h32_ecc_im
    flux += 4.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 4

    # ---- (4,1) mode: l=4, m=1, epsilon=1 (odd -> source2) ----
    hNewton_41 = prefix_41 * v2 ** (2.5) * YLMS_13  # (l+eps)/2 = (4+1)/2 = 2.5; ylms[1][3]
    Slm_41 = source2
    rholmPwrl_41 = compute_rholm_single_41(v, vh, nu, rho_41, rho_log_41, f_41)
    hlm_QC_41 = hNewton_41 * Slm_41 * T_41 * rholmPwrl_41
    hlm_re = hlm_QC_41 * h41_ecc_re
    hlm_im = hlm_QC_41 * h41_ecc_im
    flux += 1.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 1

    # ---- (4,2) mode: l=4, m=2, epsilon=0 (even -> source1) ----
    hNewton_42 = prefix_42 * v2 ** (2.0) * YLMS_24  # (l+eps)/2 = (4+0)/2 = 2; ylms[2][4]
    Slm_42 = source1
    rholmPwrl_42 = compute_rholm_single_42(v, vh, nu, rho_42, rho_log_42)
    hlm_QC_42 = hNewton_42 * Slm_42 * T_42 * rholmPwrl_42
    hlm_re = hlm_QC_42 * h42_ecc_re
    hlm_im = hlm_QC_42 * h42_ecc_im
    flux += 4.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 4

    # ---- (4,3) mode: l=4, m=3, epsilon=1 (odd -> source2) ----
    hNewton_43 = prefix_43 * v2 ** (2.5) * YLMS_33  # (l+eps)/2 = (4+1)/2 = 2.5; ylms[3][3]
    Slm_43 = source2
    rholmPwrl_43 = compute_rholm_single_43(v, vh, nu, rho_43, rho_log_43, f_43)
    hlm_QC_43 = hNewton_43 * Slm_43 * T_43 * rholmPwrl_43
    hlm_re = hlm_QC_43 * h43_ecc_re
    hlm_im = hlm_QC_43 * h43_ecc_im
    flux += 9.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 9

    # ---- (4,4) mode: l=4, m=4, epsilon=0 (even -> source1) ----
    hNewton_44 = prefix_44 * v2 ** (2.0) * YLMS_44  # (l+eps)/2 = (4+0)/2 = 2; ylms[4][4]
    Slm_44 = source1
    rholmPwrl_44 = compute_rholm_single_44(v, vh, nu, rho_44, rho_log_44)
    hlm_QC_44 = hNewton_44 * Slm_44 * T_44 * rholmPwrl_44
    hlm_re = hlm_QC_44 * h44_ecc_re
    hlm_im = hlm_QC_44 * h44_ecc_im
    flux += 16.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 16

    # ---- (5,2) mode: l=5, m=2, epsilon=1 (odd -> source2) ----
    hNewton_52 = prefix_52 * v2 ** (3.0) * YLMS_24  # (l+eps)/2 = (5+1)/2 = 3; ylms[2][4]
    Slm_52 = source2
    rholmPwrl_52 = compute_rholm_single_52(v, vh, nu, rho_52, rho_log_52)
    hlm_QC_52 = hNewton_52 * Slm_52 * T_52 * rholmPwrl_52
    hlm_re = hlm_QC_52 * h52_ecc_re
    hlm_im = hlm_QC_52 * h52_ecc_im
    flux += 4.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 4

    # ---- (5,3) mode: l=5, m=3, epsilon=0 (even -> source1) ----
    hNewton_53 = prefix_53 * v2 ** (2.5) * YLMS_35  # (l+eps)/2 = (5+0)/2 = 2.5; ylms[3][5]
    Slm_53 = source1
    rholmPwrl_53 = compute_rholm_single_53(v, vh, nu, rho_53, rho_log_53)
    hlm_QC_53 = hNewton_53 * Slm_53 * T_53 * rholmPwrl_53
    hlm_re = hlm_QC_53 * h53_ecc_re
    hlm_im = hlm_QC_53 * h53_ecc_im
    flux += 9.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 9

    # ---- (5,4) mode: l=5, m=4, epsilon=1 (odd -> source2) ----
    hNewton_54 = prefix_54 * v2 ** (3.0) * YLMS_44  # (l+eps)/2 = (5+1)/2 = 3; ylms[4][4]
    Slm_54 = source2
    rholmPwrl_54 = compute_rholm_single_54(v, vh, nu, rho_54, rho_log_54)
    hlm_QC_54 = hNewton_54 * Slm_54 * T_54 * rholmPwrl_54
    hlm_re = hlm_QC_54 * h54_ecc_re
    hlm_im = hlm_QC_54 * h54_ecc_im
    flux += 16.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 16

    # ---- (5,5) mode: l=5, m=5, epsilon=0 (even -> source1) ----
    hNewton_55 = prefix_55 * v2 ** (2.5) * YLMS_55  # (l+eps)/2 = (5+0)/2 = 2.5
    Slm_55 = source1
    rholmPwrl_55 = compute_rholm_single_55(v, vh, nu, rho_55, rho_log_55)
    hlm_QC_55 = hNewton_55 * Slm_55 * T_55 * rholmPwrl_55
    hlm_re = hlm_QC_55 * h55_ecc_re
    hlm_im = hlm_QC_55 * h55_ecc_im
    flux += 25.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 25

    # ---- (6,6) mode: l=6, m=6, epsilon=0 (even -> source1) ----
    hNewton_66 = prefix_66 * v2 ** (3.0) * YLMS_66  # (l+eps)/2 = (6+0)/2 = 3
    Slm_66 = source1
    rholmPwrl_66 = compute_rholm_single_66(v, vh, nu, rho_66, rho_log_66)
    hlm_QC_66 = hNewton_66 * Slm_66 * T_66 * rholmPwrl_66
    hlm_re = hlm_QC_66 * h66_ecc_re
    hlm_im = hlm_QC_66 * h66_ecc_im
    flux += 36.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 36

    # ---- (7,7) mode: l=7, m=7, epsilon=0 (even -> source1) ----
    hNewton_77 = prefix_77 * v2 ** (3.5) * YLMS_77  # (l+eps)/2 = (7+0)/2 = 3.5
    Slm_77 = source1
    rholmPwrl_77 = compute_rholm_single_77(v, vh, nu, rho_77, rho_log_77)
    hlm_QC_77 = hNewton_77 * Slm_77 * T_77 * rholmPwrl_77
    hlm_re = hlm_QC_77 * h77_ecc_re
    hlm_im = hlm_QC_77 * h77_ecc_im
    flux += 49.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 49

    # ---- (8,8) mode: l=8, m=8, epsilon=0 (even -> source1) ----
    hNewton_88 = prefix_88 * v2 ** (4.0) * YLMS_88  # (l+eps)/2 = (8+0)/2 = 4
    Slm_88 = source1
    rholmPwrl_88 = compute_rholm_single_88(v, vh, nu, rho_88, rho_log_88)
    hlm_QC_88 = hNewton_88 * Slm_88 * T_88 * rholmPwrl_88
    hlm_re = hlm_QC_88 * h88_ecc_re
    hlm_im = hlm_QC_88 * h88_ecc_im
    flux += 64.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)  # m^2 = 64

    return flux / (8.0 * PI)


# ============================================================================
# Backward-compatible 8-mode flux wrapper
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux_8mode(omega, e, z, x, H_times_nu, nu, pphi,
                       prefix_22, prefix_21, prefix_33,
                       prefix_32, prefix_43, prefix_44,
                       prefix_55, prefix_66,
                       rho_22, rho_log_22,
                       rho_21, rho_log_21, f_21,
                       rho_33, rho_log_33, f_33, f_vh_33_6,
                       rho_32, rho_log_32,
                       rho_43, rho_log_43, f_43,
                       rho_44, rho_log_44,
                       rho_55, rho_log_55,
                       rho_66, rho_log_66,
                       h22_ecc_re, h22_ecc_im,
                       h21_ecc_re, h21_ecc_im,
                       h33_ecc_re, h33_ecc_im,
                       h32_ecc_re, h32_ecc_im,
                       h43_ecc_re, h43_ecc_im,
                       h44_ecc_re, h44_ecc_im,
                       h55_ecc_re, h55_ecc_im,
                       h66_ecc_re, h66_ecc_im):
    """
    Backward-compatible 8-mode flux. Calls compute_flux with trivial corrections
    for the 8 new modes and zero prefixes.
    """
    rho_dummy = np.empty(11)
    rlog_dummy = np.empty(11)
    f_dummy = np.empty(11)
    for _i in range(11):
        rho_dummy[_i] = 0.0; rlog_dummy[_i] = 0.0; f_dummy[_i] = 0.0
    return compute_flux(
        omega, e, z, x, H_times_nu, nu, pphi,
        prefix_22, prefix_21,
        0.0,  # prefix_31
        prefix_33, prefix_32,
        0.0, 0.0,  # prefix_41, prefix_42
        prefix_43, prefix_44,
        0.0, 0.0, 0.0,  # prefix_52, prefix_53, prefix_54
        prefix_55, prefix_66,
        0.0, 0.0,  # prefix_77, prefix_88
        rho_22, rho_log_22,
        rho_21, rho_log_21, f_21,
        rho_dummy, rlog_dummy, f_dummy,  # rho_31, rho_log_31, f_31
        rho_33, rho_log_33, f_33, f_vh_33_6,
        rho_32, rho_log_32,
        rho_dummy, rlog_dummy, f_dummy,  # rho_41, rho_log_41, f_41
        rho_dummy, rlog_dummy,  # rho_42, rho_log_42
        rho_43, rho_log_43, f_43,
        rho_44, rho_log_44,
        rho_dummy, rlog_dummy,  # rho_52, rho_log_52
        rho_dummy, rlog_dummy,  # rho_53, rho_log_53
        rho_dummy, rlog_dummy,  # rho_54, rho_log_54
        rho_55, rho_log_55,
        rho_66, rho_log_66,
        rho_dummy, rlog_dummy,  # rho_77, rho_log_77
        rho_dummy, rlog_dummy,  # rho_88, rho_log_88
        h22_ecc_re, h22_ecc_im,
        h21_ecc_re, h21_ecc_im,
        1.0, 0.0,  # h31_ecc = 1+0i
        h32_ecc_re, h32_ecc_im,
        h33_ecc_re, h33_ecc_im,
        1.0, 0.0,  # h41_ecc = 1+0i
        1.0, 0.0,  # h42_ecc = 1+0i
        h43_ecc_re, h43_ecc_im,
        h44_ecc_re, h44_ecc_im,
        1.0, 0.0,  # h52_ecc = 1+0i
        1.0, 0.0,  # h53_ecc = 1+0i
        1.0, 0.0,  # h54_ecc = 1+0i
        h55_ecc_re, h55_ecc_im,
        h66_ecc_re, h66_ecc_im,
        1.0, 0.0,  # h77_ecc = 1+0i
        1.0, 0.0,  # h88_ecc = 1+0i
    )


# ============================================================================
# Backward-compatible 6-mode flux wrapper
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux_6mode(omega, e, z, x, H_times_nu, nu, pphi,
                       prefix_22, prefix_21, prefix_33,
                       prefix_32, prefix_43, prefix_44,
                       rho_22, rho_log_22,
                       rho_21, rho_log_21, f_21,
                       rho_33, rho_log_33, f_33, f_vh_33_6,
                       rho_32, rho_log_32,
                       rho_43, rho_log_43, f_43,
                       rho_44, rho_log_44,
                       h22_ecc_re, h22_ecc_im,
                       h21_ecc_re, h21_ecc_im,
                       h33_ecc_re, h33_ecc_im,
                       h32_ecc_re, h32_ecc_im,
                       h43_ecc_re, h43_ecc_im,
                       h44_ecc_re, h44_ecc_im):
    """
    Backward-compatible 6-mode flux wrapper.
    Calls compute_flux_8mode with trivial (1+0i) corrections for (5,5) and (6,6)
    and zero rho coefficients.
    """
    rho_55_dummy = np.empty(11)
    rlog_55_dummy = np.empty(11)
    rho_66_dummy = np.empty(11)
    rlog_66_dummy = np.empty(11)
    for _i in range(11):
        rho_55_dummy[_i] = 0.0; rlog_55_dummy[_i] = 0.0
        rho_66_dummy[_i] = 0.0; rlog_66_dummy[_i] = 0.0
    return compute_flux_8mode(
        omega, e, z, x, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_33,
        prefix_32, prefix_43, prefix_44,
        0.0, 0.0,  # prefix_55, prefix_66 = 0 => no contribution
        rho_22, rho_log_22,
        rho_21, rho_log_21, f_21,
        rho_33, rho_log_33, f_33, f_vh_33_6,
        rho_32, rho_log_32,
        rho_43, rho_log_43, f_43,
        rho_44, rho_log_44,
        rho_55_dummy, rlog_55_dummy,
        rho_66_dummy, rlog_66_dummy,
        h22_ecc_re, h22_ecc_im,
        h21_ecc_re, h21_ecc_im,
        h33_ecc_re, h33_ecc_im,
        h32_ecc_re, h32_ecc_im,
        h43_ecc_re, h43_ecc_im,
        h44_ecc_re, h44_ecc_im,
        1.0, 0.0,
        1.0, 0.0,
    )


# ============================================================================
# Backward-compatible 3-mode flux
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux_3mode(omega, e, z, x, H_times_nu, nu, pphi,
                       prefix_22, prefix_21, prefix_33,
                       rho_22, rho_log_22,
                       rho_21, rho_log_21, f_21,
                       rho_33, rho_log_33, f_33, f_vh_33_6,
                       h22_ecc_re, h22_ecc_im,
                       h21_ecc_re, h21_ecc_im,
                       h33_ecc_re, h33_ecc_im):
    """
    Compute GW flux using (2,2) + (2,1) + (3,3) modes only.
    Backward-compatible wrapper.
    """
    # Call 6-mode with trivial (1+0i) corrections for the 3 new modes
    # and zero prefixes/coefficients - but actually easier to just replicate
    # the old logic. We use the 6-mode function with dummy values.
    if x < 0.0:
        return 0.0

    v = math.sqrt(x)
    vh3 = H_times_nu * x ** 1.5
    if vh3 <= 0.0:
        vh3 = 1e-30
    vh = vh3 ** (1.0 / 3.0)
    omega_avg = x ** 1.5
    omega2 = omega * omega
    v2 = x

    T_22, T_21, T_33 = compute_tail_3mode(omega_avg, H_times_nu)

    source1 = (H_times_nu * H_times_nu - 1.0) / (2.0 * nu) + 1.0
    source2 = v * pphi

    flux = 0.0

    # (2,2)
    hNewton_22 = newtonian_multipole_abs(v2, 2, 2, prefix_22)
    rholmPwrl_22 = compute_rholm_single_22(v, vh, nu, rho_22, rho_log_22)
    hlm_QC_22 = hNewton_22 * source1 * T_22 * rholmPwrl_22
    hlm_re = hlm_QC_22 * h22_ecc_re
    hlm_im = hlm_QC_22 * h22_ecc_im
    flux += 4.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)

    # (2,1)
    hNewton_21 = newtonian_multipole_abs(v2, 2, 1, prefix_21)
    rholmPwrl_21 = compute_rholm_single_21(v, vh, nu, rho_21, rho_log_21, f_21)
    hlm_QC_21 = hNewton_21 * source2 * T_21 * rholmPwrl_21
    hlm_re = hlm_QC_21 * h21_ecc_re
    hlm_im = hlm_QC_21 * h21_ecc_im
    flux += 1.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)

    # (3,3)
    hNewton_33 = newtonian_multipole_abs(v2, 3, 3, prefix_33)
    rholmPwrl_33_re, rholmPwrl_33_im = compute_rholm_single_33(
        v, vh, nu, rho_33, rho_log_33, f_33, f_vh_33_6)
    hlm_QC_re = hNewton_33 * source1 * T_33 * rholmPwrl_33_re
    hlm_QC_im = hNewton_33 * source1 * T_33 * rholmPwrl_33_im
    hlm_re = hlm_QC_re * h33_ecc_re - hlm_QC_im * h33_ecc_im
    hlm_im = hlm_QC_re * h33_ecc_im + hlm_QC_im * h33_ecc_re
    flux += 9.0 * omega2 * (hlm_re * hlm_re + hlm_im * hlm_im)

    return flux / (8.0 * PI)


# ============================================================================
# RR force
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_rr_force(pr, pphi, flux, omega, Fphi_corr, Fr_corr, nu):
    """
    Compute the radiation reaction force in polar coordinates.
    """
    flux_norm = flux / nu
    f_over_om = flux_norm / omega

    Fr = -pr / pphi * f_over_om * Fr_corr
    Fphi = -f_over_om * Fphi_corr

    return Fr, Fphi


# ============================================================================
# Combined RR force computation (8-mode version)
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_rr_force_full(r, phi, pr, pphi, omega, e, z, x,
                          H_times_nu, nu,
                          prefix_22, prefix_21, prefix_33,
                          prefix_32, prefix_43, prefix_44,
                          prefix_55, prefix_66,
                          rho_22, rho_log_22,
                          rho_21, rho_log_21, f_21,
                          rho_33, rho_log_33, f_33, f_vh_33_6,
                          rho_32, rho_log_32,
                          rho_43, rho_log_43, f_43,
                          rho_44, rho_log_44,
                          rho_55, rho_log_55,
                          rho_66, rho_log_66,
                          h22_ecc_re, h22_ecc_im,
                          h21_ecc_re, h21_ecc_im,
                          h33_ecc_re, h33_ecc_im,
                          h32_ecc_re, h32_ecc_im,
                          h43_ecc_re, h43_ecc_im,
                          h44_ecc_re, h44_ecc_im,
                          h55_ecc_re, h55_ecc_im,
                          h66_ecc_re, h66_ecc_im,
                          Fphi_corr, Fr_corr):
    """
    Compute the full RR force: flux + force decomposition.

    This is the main function called from the ODE right-hand side.
    """
    flux = compute_flux_8mode(
        omega, e, z, x, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_33,
        prefix_32, prefix_43, prefix_44,
        prefix_55, prefix_66,
        rho_22, rho_log_22,
        rho_21, rho_log_21, f_21,
        rho_33, rho_log_33, f_33, f_vh_33_6,
        rho_32, rho_log_32,
        rho_43, rho_log_43, f_43,
        rho_44, rho_log_44,
        rho_55, rho_log_55,
        rho_66, rho_log_66,
        h22_ecc_re, h22_ecc_im,
        h21_ecc_re, h21_ecc_im,
        h33_ecc_re, h33_ecc_im,
        h32_ecc_re, h32_ecc_im,
        h43_ecc_re, h43_ecc_im,
        h44_ecc_re, h44_ecc_im,
        h55_ecc_re, h55_ecc_im,
        h66_ecc_re, h66_ecc_im,
    )

    Fr, Fphi = compute_rr_force(pr, pphi, flux, omega, Fphi_corr, Fr_corr, nu)

    return Fr, Fphi


# ============================================================================
# Circular flux (8-mode, for testing)
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux_8mode_circular(omega, x, H_times_nu, nu, pphi,
                                prefix_22, prefix_21, prefix_33,
                                prefix_32, prefix_43, prefix_44,
                                prefix_55, prefix_66,
                                rho_22, rho_log_22,
                                rho_21, rho_log_21, f_21,
                                rho_33, rho_log_33, f_33, f_vh_33_6,
                                rho_32, rho_log_32,
                                rho_43, rho_log_43, f_43,
                                rho_44, rho_log_44,
                                rho_55, rho_log_55,
                                rho_66, rho_log_66):
    """
    Compute GW flux for quasi-circular orbits (e=0), 8-mode version.
    Eccentric corrections are all (1, 0).
    """
    return compute_flux_8mode(
        omega, 0.0, 0.0, x, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_33,
        prefix_32, prefix_43, prefix_44,
        prefix_55, prefix_66,
        rho_22, rho_log_22,
        rho_21, rho_log_21, f_21,
        rho_33, rho_log_33, f_33, f_vh_33_6,
        rho_32, rho_log_32,
        rho_43, rho_log_43, f_43,
        rho_44, rho_log_44,
        rho_55, rho_log_55,
        rho_66, rho_log_66,
        1.0, 0.0,  # h22_ecc = 1+0i
        1.0, 0.0,  # h21_ecc = 1+0i
        1.0, 0.0,  # h33_ecc = 1+0i
        1.0, 0.0,  # h32_ecc = 1+0i
        1.0, 0.0,  # h43_ecc = 1+0i
        1.0, 0.0,  # h44_ecc = 1+0i
        1.0, 0.0,  # h55_ecc = 1+0i
        1.0, 0.0,  # h66_ecc = 1+0i
    )


# ============================================================================
# Backward-compatible 6-mode circular flux
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux_6mode_circular(omega, x, H_times_nu, nu, pphi,
                                prefix_22, prefix_21, prefix_33,
                                prefix_32, prefix_43, prefix_44,
                                rho_22, rho_log_22,
                                rho_21, rho_log_21, f_21,
                                rho_33, rho_log_33, f_33, f_vh_33_6,
                                rho_32, rho_log_32,
                                rho_43, rho_log_43, f_43,
                                rho_44, rho_log_44):
    """
    Compute GW flux for quasi-circular orbits (e=0), 6-mode version.
    """
    return compute_flux_6mode(
        omega, 0.0, 0.0, x, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_33,
        prefix_32, prefix_43, prefix_44,
        rho_22, rho_log_22,
        rho_21, rho_log_21, f_21,
        rho_33, rho_log_33, f_33, f_vh_33_6,
        rho_32, rho_log_32,
        rho_43, rho_log_43, f_43,
        rho_44, rho_log_44,
        1.0, 0.0,  # h22_ecc = 1+0i
        1.0, 0.0,  # h21_ecc = 1+0i
        1.0, 0.0,  # h33_ecc = 1+0i
        1.0, 0.0,  # h32_ecc = 1+0i
        1.0, 0.0,  # h43_ecc = 1+0i
        1.0, 0.0,  # h44_ecc = 1+0i
    )


# ============================================================================
# Backward-compatible 3-mode circular flux
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_flux_3mode_circular(omega, x, H_times_nu, nu, pphi,
                                prefix_22, prefix_21, prefix_33,
                                rho_22, rho_log_22,
                                rho_21, rho_log_21, f_21,
                                rho_33, rho_log_33, f_33, f_vh_33_6):
    """
    Compute GW flux for quasi-circular orbits (e=0), 3-mode version.
    """
    return compute_flux_3mode(
        omega, 0.0, 0.0, x, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_33,
        rho_22, rho_log_22,
        rho_21, rho_log_21, f_21,
        rho_33, rho_log_33, f_33, f_vh_33_6,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    )
