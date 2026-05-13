"""
Numba-JIT circular waveform infrastructure for the 16-mode eccentric EOB flux.

Translated from pySEOBNR waveform.pyx.
Modes: (2,2), (2,1), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (4,4),
       (5,2), (5,3), (5,4), (5,5), (6,6), (7,7), (8,8).

Functions:
  - compute_newtonian_prefixes_abs: Newtonian multipole |prefixes| for 16 modes
  - compute_tail_16mode: Tail factors |T_lm| for 16 modes
  - compute_rho_coeffs_16mode: rho_lm PN coefficients for 16 modes
  - compute_rholm_single_*: rho_lm^l resummed factor for each mode
  - newtonian_multipole_abs: |h_Newton| for one mode
"""

import math
import numpy as np
from numba import njit

# ============================================================================
# Constants
# ============================================================================
EULER_GAMMA = 0.5772156649015329
PI = math.pi
PN_LIMIT = 11  # PN_limit from eob_parameters.h

# Lookup table: |Y_l^{-m}(pi/2, 0)| (spin-weighted spherical harmonics)
# ylms[m][l] -- entries for all modes
YLMS_22 = 0.3862742020231896   # ylms[2][2]
YLMS_11 = 0.3454941494713355   # ylms[1][1]  (for l=2, m=1, epsilon=1 => l-eps=1)
YLMS_33 = 0.4172238236327842   # ylms[3][3]
# (3,2): l=3, m=2, eps=1, l-eps=2 => ylms[2][2]
YLMS_22_for_32 = 0.3862742020231896   # ylms[2][2]
# (4,3): l=4, m=3, eps=1, l-eps=3 => ylms[3][3]
YLMS_33_for_43 = 0.4172238236327842   # ylms[3][3]
# (4,4): l=4, m=4, eps=0, l-eps=4 => ylms[4][4]
YLMS_44 = 0.4425326924449826   # ylms[4][4]
# (5,5): l=5, m=5, eps=0, l-eps=5 => ylms[5][5]
YLMS_55 = 0.3641169726079959   # |Y_{-2,5,5}(pi/2, 0)|
# (6,6): l=6, m=6, eps=0, l-eps=6 => ylms[6][6]
YLMS_66 = 0.3017325350840579   # |Y_{-2,6,6}(pi/2, 0)|
# New modes for 16-mode extension:
# (3,1): l=3, m=1, eps=0, l-eps=3 => ylms[1][3]
YLMS_13 = 0.3231801841141506   # ylms[1][3]
# (4,1): l=4, m=1, eps=1, l-eps=3 => ylms[1][3]
# (reuses YLMS_13)
# (4,2): l=4, m=2, eps=0, l-eps=4 => ylms[2][4]
YLMS_24 = 0.33452327177864466  # ylms[2][4]
# (5,2): l=5, m=2, eps=1, l-eps=4 => ylms[2][4]
# (reuses YLMS_24)
# (5,3): l=5, m=3, eps=0, l-eps=5 => ylms[3][5]
YLMS_35 = 0.34594371914684025  # ylms[3][5]
# (5,4): l=5, m=4, eps=1, l-eps=4 => ylms[4][4]
# (reuses YLMS_44)
# (7,7): l=7, m=7, eps=0, l-eps=7 => ylms[7][7]
YLMS_77 = 0.5000395635705508   # ylms[7][7]
# (8,8): l=8, m=8, eps=0, l-eps=8 => ylms[8][8]
YLMS_88 = 0.5154289843972844   # ylms[8][8]

# Factorial lookup table (n=0..20)
FACTORIAL_TABLE = np.array([
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
    362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0,
    87178291200.0, 1307674368000.0, 20922789888000.0,
    355687428096000.0, 6402373705728000.0, 121645100408832000.0,
    2432902008176640000.0,
], dtype=np.float64)

# Double factorial (2l+1)!! for l=2,3,4,5,6,7,8
# (2*2+1)!! = 5!! = 15
# (2*3+1)!! = 7!! = 105
# (2*4+1)!! = 9!! = 945
# (2*5+1)!! = 11!! = 10395
# (2*6+1)!! = 13!! = 135135
# (2*7+1)!! = 15!! = 2027025
# (2*8+1)!! = 17!! = 34459425
DOUBLE_FACTORIAL_5 = 15.0
DOUBLE_FACTORIAL_7 = 105.0
DOUBLE_FACTORIAL_9 = 945.0
DOUBLE_FACTORIAL_11 = 10395.0
DOUBLE_FACTORIAL_13 = 135135.0
DOUBLE_FACTORIAL_15 = 2027025.0
DOUBLE_FACTORIAL_17 = 34459425.0


# ============================================================================
# Newtonian multipole prefixes (absolute value)
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_newtonian_prefixes_abs(m_1, m_2):
    """
    Compute |prefix| for 16 modes.

    From waveform.pyx calculate_multipole_prefix, taking abs value.

    Parameters
    ----------
    m_1, m_2 : float  (m_1 + m_2 = 1)

    Returns
    -------
    prefix_22, prefix_21, prefix_31, prefix_33, prefix_32,
    prefix_41, prefix_42, prefix_43, prefix_44,
    prefix_52, prefix_53, prefix_54, prefix_55,
    prefix_66, prefix_77, prefix_88 : float
        Absolute values of the Newtonian multipole prefixes.
    """
    totalMass = m_1 + m_2
    x1 = m_1 / totalMass
    x2 = m_2 / totalMass
    eta = m_1 * m_2 / (totalMass * totalMass)

    dm = m_1 - m_2

    # ---- (2,2) mode: l=2, m=2, epsilon=0 ----
    # sign = +1 (m even)
    # c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2 + x1 = 1
    c_22 = x2 + x1  # = 1.0
    # n for epsilon=0:
    # (i*2)^2 = -4, |.| = 4
    # 8*pi/15 * sqrt(3*4/(2*1)) = 8*pi/15 * sqrt(6)
    n_22_abs = 4.0 * 8.0 * PI / DOUBLE_FACTORIAL_5 * math.sqrt(
        (3.0 * 4.0) / (2.0 * 1.0)
    )
    prefix_22 = n_22_abs * eta * abs(c_22)

    # ---- (2,1) mode: l=2, m=1, epsilon=1 ----
    # sign = -1 (m odd)
    # For epsilon=1: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^2 - x1^2
    if abs(dm) > 1e-14:
        c_21 = x2 * x2 - x1 * x1
    else:
        c_21 = -1.0

    # n for epsilon=1:
    # -(i*1)^2 = 1, * i => magnitude = 1
    # 16*pi/15 * sqrt(5*4*(4-1)/(3*3*2*1)) = 16*pi/15 * sqrt(20/18)
    n_21_abs = 1.0 * 16.0 * PI / DOUBLE_FACTORIAL_5 * math.sqrt(
        (5.0 * 4.0 * (4.0 - 1.0)) / (3.0 * 3.0 * 2.0 * 1.0)
    )
    prefix_21 = n_21_abs * eta * abs(c_21)

    # ---- (3,3) mode: l=3, m=3, epsilon=0 ----
    # sign = -1 (m odd)
    # c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^2 - x1^2
    if abs(dm) > 1e-14:
        c_33 = x2 * x2 - x1 * x1
    else:
        c_33 = -1.0

    # n for epsilon=0:
    # (i*3)^3 = i^3 * 27 = -27i, |.| = 27
    # 8*pi/105 * sqrt(4*5/(3*2))
    n_33_abs = 27.0 * 8.0 * PI / DOUBLE_FACTORIAL_7 * math.sqrt(
        (4.0 * 5.0) / (3.0 * 2.0)
    )
    prefix_33 = n_33_abs * eta * abs(c_33)

    # ---- (3,2) mode: l=3, m=2, epsilon=1 ----
    # sign = +1 (m even)
    # For epsilon=1: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^3 + x1^3
    if abs(dm) > 1e-14:
        c_32 = x2 * x2 * x2 + x1 * x1 * x1
    else:
        # Equal mass: sign=+1, l=3, eps=1 => c = x2^3 + x1^3
        # For equal mass x1=x2=0.5: c = 0.125+0.125 = 0.25
        # But the formula works for equal mass too (no dm division)
        c_32 = x2 * x2 * x2 + x1 * x1 * x1

    # n for epsilon=1:
    # -(i*2)^3 = -(-8i) = 8i, |.| = 8
    # * i => |n| = 8
    # 16*pi/105 * sqrt(7*5*(9-4)/(5*4*3*2)) = 16*pi/105 * sqrt(7*5*5/(5*4*3*2))
    # = 16*pi/105 * sqrt(175/120)
    # Simplified: (2l+1)=7, (l+2)=5, (l^2-m^2)=(9-4)=5, (2l-1)=5, (l+1)=4, l=3, (l-1)=2
    n_32_abs = 8.0 * 16.0 * PI / DOUBLE_FACTORIAL_7 * math.sqrt(
        (7.0 * 5.0 * 5.0) / (5.0 * 4.0 * 3.0 * 2.0)
    )
    prefix_32 = n_32_abs * eta * abs(c_32)

    # ---- (4,3) mode: l=4, m=3, epsilon=1 ----
    # sign = -1 (m odd)
    # For epsilon=1: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^4 - x1^4
    if abs(dm) > 1e-14:
        c_43 = x2 ** 4 - x1 ** 4
    else:
        # Equal mass: l=4, odd m => c = -0.5
        c_43 = -0.5

    # n for epsilon=1:
    # -(i*3)^4 = -(81) = -81, * i => |-81i| = 81
    # 16*pi/945 * sqrt(9*6*(16-9)/(7*5*4*3)) = 16*pi/945 * sqrt(9*6*7/(7*5*4*3))
    # = 16*pi/945 * sqrt(378/420) = 16*pi/945 * sqrt(9/10)
    # (2l+1)=9, (l+2)=6, (l^2-m^2)=(16-9)=7, (2l-1)=7, (l+1)=5, l=4, (l-1)=3
    n_43_abs = 81.0 * 16.0 * PI / DOUBLE_FACTORIAL_9 * math.sqrt(
        (9.0 * 6.0 * 7.0) / (7.0 * 5.0 * 4.0 * 3.0)
    )
    prefix_43 = n_43_abs * eta * abs(c_43)

    # ---- (4,4) mode: l=4, m=4, epsilon=0 ----
    # sign = +1 (m even)
    # For epsilon=0: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^3 + x1^3
    if abs(dm) > 1e-14:
        c_44 = x2 * x2 * x2 + x1 * x1 * x1
    else:
        c_44 = x2 * x2 * x2 + x1 * x1 * x1

    # n for epsilon=0:
    # (i*4)^4 = 256, |.| = 256
    # 8*pi/945 * sqrt(5*6/(4*3)) = 8*pi/945 * sqrt(30/12) = 8*pi/945 * sqrt(5/2)
    # (l+1)=5, (l+2)=6, l=4, (l-1)=3
    n_44_abs = 256.0 * 8.0 * PI / DOUBLE_FACTORIAL_9 * math.sqrt(
        (5.0 * 6.0) / (4.0 * 3.0)
    )
    prefix_44 = n_44_abs * eta * abs(c_44)

    # ---- (5,5) mode: l=5, m=5, epsilon=0 ----
    # sign = -1 (m odd)
    # For epsilon=0: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^4 - x1^4
    # c_55 = x2^4 - x1^4 (vanishes at equal mass by symmetry)
    c_55 = x2 ** 4 - x1 ** 4

    # n for epsilon=0:
    # (i*5)^5 = i^5 * 3125 = 3125i, |.| = 3125
    # 8*pi/10395 * sqrt((l+1)*(l+2)/(l*(l-1))) = 8*pi/10395 * sqrt(6*7/(5*4))
    n_55_abs = 3125.0 * 8.0 * PI / DOUBLE_FACTORIAL_11 * math.sqrt(
        (6.0 * 7.0) / (5.0 * 4.0)
    )
    prefix_55 = n_55_abs * eta * abs(c_55)

    # ---- (6,6) mode: l=6, m=6, epsilon=0 ----
    # sign = +1 (m even)
    # For epsilon=0: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^5 + x1^5
    if abs(dm) > 1e-14:
        c_66 = x2 ** 5 + x1 ** 5
    else:
        c_66 = x2 ** 5 + x1 ** 5

    # n for epsilon=0:
    # (i*6)^6 = i^6 * 46656 = -46656, |.| = 46656
    # 8*pi/135135 * sqrt((l+1)*(l+2)/(l*(l-1))) = 8*pi/135135 * sqrt(7*8/(6*5))
    n_66_abs = 46656.0 * 8.0 * PI / DOUBLE_FACTORIAL_13 * math.sqrt(
        (7.0 * 8.0) / (6.0 * 5.0)
    )
    prefix_66 = n_66_abs * eta * abs(c_66)

    # ---- (3,1) mode: l=3, m=1, epsilon=0 ----
    # sign = -1 (m odd)
    # For epsilon=0: c = x2^(l-1) + sign * x1^(l-1) = x2^2 - x1^2
    if abs(dm) > 1e-14:
        c_31 = x2 * x2 - x1 * x1
    else:
        c_31 = -1.0

    # n for epsilon=0:
    # (i*1)^3 = -i, |.| = 1
    # 8*pi/105 * sqrt((l+1)*(l+2)/(l*(l-1))) = 8*pi/105 * sqrt(4*5/(3*2))
    n_31_abs = 1.0 * 8.0 * PI / DOUBLE_FACTORIAL_7 * math.sqrt(
        (4.0 * 5.0) / (3.0 * 2.0)
    )
    prefix_31 = n_31_abs * eta * abs(c_31)

    # ---- (4,1) mode: l=4, m=1, epsilon=1 ----
    # sign = -1 (m odd)
    # For epsilon=1: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^4 - x1^4
    if abs(dm) > 1e-14:
        c_41 = x2 ** 4 - x1 ** 4
    else:
        c_41 = -0.5

    # n for epsilon=1:
    # -(i*1)^4 = -1, * i => |-i| = 1
    # 16*pi/(2l+1)!! * sqrt((2l+1)*(l+2)*(l^2-m^2)/((2l-1)*(l+1)*l*(l-1)))
    # l=4, m=1: 16*pi/945 * sqrt(9*6*(16-1)/(7*5*4*3))
    # = 16*pi/945 * sqrt(9*6*15/(7*5*4*3)) = 16*pi/945 * sqrt(810/420)
    n_41_abs = 1.0 * 16.0 * PI / DOUBLE_FACTORIAL_9 * math.sqrt(
        (9.0 * 6.0 * 15.0) / (7.0 * 5.0 * 4.0 * 3.0)
    )
    prefix_41 = n_41_abs * eta * abs(c_41)

    # ---- (4,2) mode: l=4, m=2, epsilon=0 ----
    # sign = +1 (m even)
    # For epsilon=0: c = x2^(l-1) + sign * x1^(l-1) = x2^3 + x1^3
    c_42 = x2 * x2 * x2 + x1 * x1 * x1

    # n for epsilon=0:
    # (i*2)^4 = 16, |.| = 16
    # 8*pi/945 * sqrt(5*6/(4*3))
    n_42_abs = 16.0 * 8.0 * PI / DOUBLE_FACTORIAL_9 * math.sqrt(
        (5.0 * 6.0) / (4.0 * 3.0)
    )
    prefix_42 = n_42_abs * eta * abs(c_42)

    # ---- (5,2) mode: l=5, m=2, epsilon=1 ----
    # sign = +1 (m even)
    # For epsilon=1: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^5 + x1^5
    c_52 = x2 ** 5 + x1 ** 5

    # n for epsilon=1:
    # -(i*2)^5 = -32i, * i => |32i*i|=32... let's compute carefully:
    # n = -(i*m)^l * i = -(i*2)^5 * i = -(32i) * i = 32, |n| = 32
    # Actually: (i*m)^l = (2i)^5 = 32*i^5 = 32*i. Then n = -(32i) * 16*pi/... * i * sqrt(...)
    # But we just need |n|. |-(i*m)^l| = m^l = 2^5 = 32
    # 16*pi/(2l+1)!! * sqrt((2l+1)*(l+2)*(l^2-m^2)/((2l-1)*(l+1)*l*(l-1)))
    # l=5, m=2: 16*pi/10395 * sqrt(11*7*(25-4)/(9*6*5*4))
    # = 16*pi/10395 * sqrt(11*7*21/(9*6*5*4)) = 16*pi/10395 * sqrt(1617/1080)
    n_52_abs = 32.0 * 16.0 * PI / DOUBLE_FACTORIAL_11 * math.sqrt(
        (11.0 * 7.0 * 21.0) / (9.0 * 6.0 * 5.0 * 4.0)
    )
    prefix_52 = n_52_abs * eta * abs(c_52)

    # ---- (5,3) mode: l=5, m=3, epsilon=0 ----
    # sign = -1 (m odd)
    # For epsilon=0: c = x2^(l-1) + sign * x1^(l-1) = x2^4 - x1^4
    c_53 = x2 ** 4 - x1 ** 4
    if abs(dm) < 1e-14:
        c_53 = -0.5  # equal mass: vanishes for odd m, but l=5 odd => c0=-0.5

    # n for epsilon=0:
    # (i*3)^5 = 243*i^5 = 243*i, |.| = 243
    # 8*pi/10395 * sqrt(6*7/(5*4))
    n_53_abs = 243.0 * 8.0 * PI / DOUBLE_FACTORIAL_11 * math.sqrt(
        (6.0 * 7.0) / (5.0 * 4.0)
    )
    prefix_53 = n_53_abs * eta * abs(c_53)

    # ---- (5,4) mode: l=5, m=4, epsilon=1 ----
    # sign = +1 (m even)
    # For epsilon=1: c = x2^(l+eps-1) + sign * x1^(l+eps-1) = x2^5 + x1^5
    c_54 = x2 ** 5 + x1 ** 5

    # n for epsilon=1:
    # -(i*4)^5 = -(1024*i^5) = -(1024*i) = -1024i, * i => 1024, |.| = 1024
    # 16*pi/10395 * sqrt(11*7*(25-16)/(9*6*5*4))
    # = 16*pi/10395 * sqrt(11*7*9/(9*6*5*4)) = 16*pi/10395 * sqrt(693/1080)
    n_54_abs = 1024.0 * 16.0 * PI / DOUBLE_FACTORIAL_11 * math.sqrt(
        (11.0 * 7.0 * 9.0) / (9.0 * 6.0 * 5.0 * 4.0)
    )
    prefix_54 = n_54_abs * eta * abs(c_54)

    # ---- (7,7) mode: l=7, m=7, epsilon=0 ----
    # sign = -1 (m odd)
    # For epsilon=0: c = x2^(l-1) + sign * x1^(l-1) = x2^6 - x1^6
    c_77 = x2 ** 6 - x1 ** 6
    if abs(dm) < 1e-14:
        c_77 = 0.0  # equal mass: odd m, l>=6 => c=0

    # n for epsilon=0:
    # (i*7)^7 = 7^7 * i^7 = 823543 * (-i), |.| = 823543
    # 8*pi/2027025 * sqrt(8*9/(7*6))
    n_77_abs = 823543.0 * 8.0 * PI / DOUBLE_FACTORIAL_15 * math.sqrt(
        (8.0 * 9.0) / (7.0 * 6.0)
    )
    prefix_77 = n_77_abs * eta * abs(c_77)

    # ---- (8,8) mode: l=8, m=8, epsilon=0 ----
    # sign = +1 (m even)
    # For epsilon=0: c = x2^(l-1) + sign * x1^(l-1) = x2^7 + x1^7
    c_88 = x2 ** 7 + x1 ** 7

    # n for epsilon=0:
    # (i*8)^8 = 8^8 * i^8 = 16777216 * 1, |.| = 16777216
    # 8*pi/34459425 * sqrt(9*10/(8*7))
    n_88_abs = 16777216.0 * 8.0 * PI / DOUBLE_FACTORIAL_17 * math.sqrt(
        (9.0 * 10.0) / (8.0 * 7.0)
    )
    prefix_88 = n_88_abs * eta * abs(c_88)

    return (prefix_22, prefix_21, prefix_31, prefix_33, prefix_32,
            prefix_41, prefix_42, prefix_43, prefix_44,
            prefix_52, prefix_53, prefix_54, prefix_55,
            prefix_66, prefix_77, prefix_88)


@njit(cache=True, fastmath=True)
def newtonian_multipole_abs(v2, l, m, prefix_abs):
    """
    Compute |h_Newton| for mode (l,m) for the flux.

    From EOBFluxCalculateNewtonianMultipoleAbs in waveform.pyx:
      multipole = prefix_abs * v2^((l+epsilon)/2) * ylms[m][l-epsilon]

    Parameters
    ----------
    v2 : float  (= v^2 = x)
    l, m : int
    prefix_abs : float

    Returns
    -------
    float: |h_Newton|
    """
    epsilon = (l + m) % 2

    # Select |Y_l^{-m}(pi/2, 0)| for l-epsilon
    # ylms[m][l-epsilon] from the pySEOBNR lookup table
    l_eff = l - epsilon
    if l_eff == 1 and m == 1:
        y = YLMS_11
    elif l_eff == 2 and m == 2:
        y = YLMS_22  # used by (2,2) and (3,2)
    elif l_eff == 3 and m == 1:
        y = YLMS_13  # used by (3,1) and (4,1)
    elif l_eff == 3 and m == 3:
        y = YLMS_33  # used by (3,3) and (4,3)
    elif l_eff == 4 and m == 2:
        y = YLMS_24  # used by (4,2) and (5,2)
    elif l_eff == 4 and m == 4:
        y = YLMS_44  # used by (4,4) and (5,4)
    elif l_eff == 5 and m == 3:
        y = YLMS_35  # used by (5,3)
    elif l_eff == 5 and m == 5:
        y = YLMS_55
    elif l_eff == 6 and m == 6:
        y = YLMS_66
    elif l_eff == 7 and m == 7:
        y = YLMS_77
    elif l_eff == 8 and m == 8:
        y = YLMS_88
    else:
        y = 0.0

    multipole = prefix_abs * v2 ** ((l + epsilon) * 0.5) * y
    return multipole


# ============================================================================
# Tail factor |T_lm|
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_tail_16mode(omega_avg, H):
    """
    Compute |T_lm| for all 16 modes.

    From waveform.pyx compute_tail.

    Parameters
    ----------
    omega_avg : float
    H : float  (= nu * H_real)

    Returns
    -------
    T_22, T_21, T_31, T_33, T_32, T_41, T_42, T_43, T_44,
    T_52, T_53, T_54, T_55, T_66, T_77, T_88 : float
    """
    # m=1: used by (2,1), (3,1), (4,1)
    k1 = 1.0 * omega_avg
    hathatk1 = H * k1
    hathatksq4_1 = 4.0 * hathatk1 * hathatk1
    hathatk4pi_1 = 4.0 * PI * hathatk1

    if abs(hathatk4pi_1) < 1e-15:
        tlmprefac_1 = 1.0
    else:
        tlmprefac_1 = math.sqrt(hathatk4pi_1 / (1.0 - math.exp(-hathatk4pi_1)))

    # T_21: l=2, m=1 => product j=1..2
    prod_21 = 1.0
    for j in range(1, 3):  # j=1,2
        prod_21 *= (hathatksq4_1 + j * j)
    T_21 = tlmprefac_1 * math.sqrt(prod_21) / FACTORIAL_TABLE[2]  # 2! = 2

    # T_31: l=3, m=1 => product j=1..3
    prod_31 = 1.0
    for j in range(1, 4):  # j=1,2,3
        prod_31 *= (hathatksq4_1 + j * j)
    T_31 = tlmprefac_1 * math.sqrt(prod_31) / FACTORIAL_TABLE[3]  # 3! = 6

    # T_41: l=4, m=1 => product j=1..4
    prod_41 = 1.0
    for j in range(1, 5):  # j=1,2,3,4
        prod_41 *= (hathatksq4_1 + j * j)
    T_41 = tlmprefac_1 * math.sqrt(prod_41) / FACTORIAL_TABLE[4]  # 4! = 24

    # m=2: used by (2,2), (3,2), (4,2), (5,2)
    k2 = 2.0 * omega_avg
    hathatk2 = H * k2
    hathatksq4_2 = 4.0 * hathatk2 * hathatk2
    hathatk4pi_2 = 4.0 * PI * hathatk2

    if abs(hathatk4pi_2) < 1e-15:
        tlmprefac_2 = 1.0
    else:
        tlmprefac_2 = math.sqrt(hathatk4pi_2 / (1.0 - math.exp(-hathatk4pi_2)))

    # T_22: l=2, m=2 => product j=1..2
    prod_22 = 1.0
    for j in range(1, 3):
        prod_22 *= (hathatksq4_2 + j * j)
    T_22 = tlmprefac_2 * math.sqrt(prod_22) / FACTORIAL_TABLE[2]

    # T_32: l=3, m=2 => product j=1..3
    prod_32 = 1.0
    for j in range(1, 4):  # j=1,2,3
        prod_32 *= (hathatksq4_2 + j * j)
    T_32 = tlmprefac_2 * math.sqrt(prod_32) / FACTORIAL_TABLE[3]  # 3! = 6

    # T_42: l=4, m=2 => product j=1..4
    prod_42 = 1.0
    for j in range(1, 5):  # j=1,2,3,4
        prod_42 *= (hathatksq4_2 + j * j)
    T_42 = tlmprefac_2 * math.sqrt(prod_42) / FACTORIAL_TABLE[4]  # 4! = 24

    # T_52: l=5, m=2 => product j=1..5
    prod_52 = 1.0
    for j in range(1, 6):  # j=1,2,3,4,5
        prod_52 *= (hathatksq4_2 + j * j)
    T_52 = tlmprefac_2 * math.sqrt(prod_52) / FACTORIAL_TABLE[5]  # 5! = 120

    # m=3: used by (3,3), (4,3), (5,3)
    k3 = 3.0 * omega_avg
    hathatk3 = H * k3
    hathatksq4_3 = 4.0 * hathatk3 * hathatk3
    hathatk4pi_3 = 4.0 * PI * hathatk3

    if abs(hathatk4pi_3) < 1e-15:
        tlmprefac_3 = 1.0
    else:
        tlmprefac_3 = math.sqrt(hathatk4pi_3 / (1.0 - math.exp(-hathatk4pi_3)))

    # T_33: l=3, m=3 => product j=1..3
    prod_33 = 1.0
    for j in range(1, 4):
        prod_33 *= (hathatksq4_3 + j * j)
    T_33 = tlmprefac_3 * math.sqrt(prod_33) / FACTORIAL_TABLE[3]  # 3! = 6

    # T_43: l=4, m=3 => product j=1..4
    prod_43 = 1.0
    for j in range(1, 5):  # j=1,2,3,4
        prod_43 *= (hathatksq4_3 + j * j)
    T_43 = tlmprefac_3 * math.sqrt(prod_43) / FACTORIAL_TABLE[4]  # 4! = 24

    # T_53: l=5, m=3 => product j=1..5
    prod_53 = 1.0
    for j in range(1, 6):  # j=1,2,3,4,5
        prod_53 *= (hathatksq4_3 + j * j)
    T_53 = tlmprefac_3 * math.sqrt(prod_53) / FACTORIAL_TABLE[5]  # 5! = 120

    # m=4: used by (4,4), (5,4)
    k4 = 4.0 * omega_avg
    hathatk4 = H * k4
    hathatksq4_4 = 4.0 * hathatk4 * hathatk4
    hathatk4pi_4 = 4.0 * PI * hathatk4

    if abs(hathatk4pi_4) < 1e-15:
        tlmprefac_4 = 1.0
    else:
        tlmprefac_4 = math.sqrt(hathatk4pi_4 / (1.0 - math.exp(-hathatk4pi_4)))

    # T_44: l=4, m=4 => product j=1..4
    prod_44 = 1.0
    for j in range(1, 5):  # j=1,2,3,4
        prod_44 *= (hathatksq4_4 + j * j)
    T_44 = tlmprefac_4 * math.sqrt(prod_44) / FACTORIAL_TABLE[4]  # 4! = 24

    # T_54: l=5, m=4 => product j=1..5
    prod_54 = 1.0
    for j in range(1, 6):  # j=1,2,3,4,5
        prod_54 *= (hathatksq4_4 + j * j)
    T_54 = tlmprefac_4 * math.sqrt(prod_54) / FACTORIAL_TABLE[5]  # 5! = 120

    # m=5: used by (5,5)
    k5 = 5.0 * omega_avg
    hathatk5 = H * k5
    hathatksq4_5 = 4.0 * hathatk5 * hathatk5
    hathatk4pi_5 = 4.0 * PI * hathatk5

    if abs(hathatk4pi_5) < 1e-15:
        tlmprefac_5 = 1.0
    else:
        tlmprefac_5 = math.sqrt(hathatk4pi_5 / (1.0 - math.exp(-hathatk4pi_5)))

    # T_55: l=5, m=5 => product j=1..5
    prod_55 = 1.0
    for j in range(1, 6):  # j=1,2,3,4,5
        prod_55 *= (hathatksq4_5 + j * j)
    T_55 = tlmprefac_5 * math.sqrt(prod_55) / FACTORIAL_TABLE[5]  # 5! = 120

    # m=6: used by (6,6)
    k6 = 6.0 * omega_avg
    hathatk6 = H * k6
    hathatksq4_6 = 4.0 * hathatk6 * hathatk6
    hathatk4pi_6 = 4.0 * PI * hathatk6

    if abs(hathatk4pi_6) < 1e-15:
        tlmprefac_6 = 1.0
    else:
        tlmprefac_6 = math.sqrt(hathatk4pi_6 / (1.0 - math.exp(-hathatk4pi_6)))

    # T_66: l=6, m=6 => product j=1..6
    prod_66 = 1.0
    for j in range(1, 7):  # j=1,2,3,4,5,6
        prod_66 *= (hathatksq4_6 + j * j)
    T_66 = tlmprefac_6 * math.sqrt(prod_66) / FACTORIAL_TABLE[6]  # 6! = 720

    # m=7: used by (7,7)
    k7 = 7.0 * omega_avg
    hathatk7 = H * k7
    hathatksq4_7 = 4.0 * hathatk7 * hathatk7
    hathatk4pi_7 = 4.0 * PI * hathatk7

    if abs(hathatk4pi_7) < 1e-15:
        tlmprefac_7 = 1.0
    else:
        tlmprefac_7 = math.sqrt(hathatk4pi_7 / (1.0 - math.exp(-hathatk4pi_7)))

    # T_77: l=7, m=7 => product j=1..7
    prod_77 = 1.0
    for j in range(1, 8):  # j=1,2,3,4,5,6,7
        prod_77 *= (hathatksq4_7 + j * j)
    T_77 = tlmprefac_7 * math.sqrt(prod_77) / FACTORIAL_TABLE[7]  # 7! = 5040

    # m=8: used by (8,8)
    k8 = 8.0 * omega_avg
    hathatk8 = H * k8
    hathatksq4_8 = 4.0 * hathatk8 * hathatk8
    hathatk4pi_8 = 4.0 * PI * hathatk8

    if abs(hathatk4pi_8) < 1e-15:
        tlmprefac_8 = 1.0
    else:
        tlmprefac_8 = math.sqrt(hathatk4pi_8 / (1.0 - math.exp(-hathatk4pi_8)))

    # T_88: l=8, m=8 => product j=1..8
    prod_88 = 1.0
    for j in range(1, 9):  # j=1,2,3,4,5,6,7,8
        prod_88 *= (hathatksq4_8 + j * j)
    T_88 = tlmprefac_8 * math.sqrt(prod_88) / FACTORIAL_TABLE[8]  # 8! = 40320

    return (T_22, T_21, T_31, T_33, T_32, T_41, T_42, T_43, T_44,
            T_52, T_53, T_54, T_55, T_66, T_77, T_88)


# Backward-compatible wrappers
@njit(cache=True, fastmath=True)
def compute_tail_8mode(omega_avg, H):
    """Backward-compatible 8-mode tail wrapper."""
    result = compute_tail_16mode(omega_avg, H)
    T_22 = result[0]; T_21 = result[1]; T_33 = result[3]; T_32 = result[4]
    T_43 = result[7]; T_44 = result[8]; T_55 = result[12]; T_66 = result[13]
    return T_22, T_21, T_33, T_32, T_43, T_44, T_55, T_66


@njit(cache=True, fastmath=True)
def compute_tail_6mode(omega_avg, H):
    """Backward-compatible 6-mode tail wrapper."""
    result = compute_tail_16mode(omega_avg, H)
    T_22 = result[0]; T_21 = result[1]; T_33 = result[3]; T_32 = result[4]
    T_43 = result[7]; T_44 = result[8]
    return T_22, T_21, T_33, T_32, T_43, T_44


@njit(cache=True, fastmath=True)
def compute_tail_3mode(omega_avg, H):
    """Backward-compatible 3-mode tail wrapper."""
    result = compute_tail_16mode(omega_avg, H)
    T_22 = result[0]; T_21 = result[1]; T_33 = result[3]
    return T_22, T_21, T_33


# ============================================================================
# rho coefficients for 6 modes
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_rho_coeffs_16mode(nu, dm, a, chiS, chiA, extra_PN_terms):
    """
    Compute rho_lm PN coefficients for all 16 modes.

    Returns arrays indexed [0..PN_LIMIT-1] for each mode, plus f_coeffs
    for the odd-m modes.

    From waveform.pyx compute_rho_coeffs.

    Parameters
    ----------
    nu : float
    dm : float  (delta = (m1-m2)/(m1+m2))
    a : float   (Kerr parameter)
    chiS, chiA : float
    extra_PN_terms : bool

    Returns
    -------
    rho_22, rho_log_22 : arrays
    rho_21, rho_log_21, f_21 : arrays
    rho_31, rho_log_31, f_31 : arrays
    rho_33, rho_log_33, f_33 : arrays
    f_vh_33_6 : float
    rho_32, rho_log_32 : arrays
    rho_41, rho_log_41, f_41 : arrays
    rho_42, rho_log_42 : arrays
    rho_43, rho_log_43, f_43 : arrays
    rho_44, rho_log_44 : arrays
    rho_52, rho_log_52 : arrays
    rho_53, rho_log_53 : arrays
    rho_54, rho_log_54 : arrays
    rho_55, rho_log_55 : arrays
    rho_66, rho_log_66 : arrays
    rho_77, rho_log_77 : arrays
    rho_88, rho_log_88 : arrays
    """
    nu2 = nu * nu
    nu3 = nu * nu2
    nu4 = nu2 * nu2

    a2 = a * a
    a3 = a * a2
    dm2 = dm * dm
    atemp = a  # save before zeroing

    chiA2 = chiA * chiA
    chiS2 = chiS * chiS
    chiA3 = chiA2 * chiA
    chiS3 = chiS2 * chiS

    m1Plus3nu = -1.0 + 3.0 * nu
    m1Plus3nu2 = m1Plus3nu * m1Plus3nu
    m1Plus3nu3 = m1Plus3nu * m1Plus3nu2

    # Output arrays
    rho_22 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_22 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_21 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_21 = np.zeros(PN_LIMIT, dtype=np.float64)
    f_21 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_31 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_31 = np.zeros(PN_LIMIT, dtype=np.float64)
    f_31 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_33 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_33 = np.zeros(PN_LIMIT, dtype=np.float64)
    f_33 = np.zeros(PN_LIMIT, dtype=np.float64)
    f_vh_33_6 = 0.0
    rho_32 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_32 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_41 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_41 = np.zeros(PN_LIMIT, dtype=np.float64)
    f_41 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_42 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_42 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_43 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_43 = np.zeros(PN_LIMIT, dtype=np.float64)
    f_43 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_44 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_44 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_52 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_52 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_53 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_53 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_54 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_54 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_55 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_55 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_66 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_66 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_77 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_77 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_88 = np.zeros(PN_LIMIT, dtype=np.float64)
    rho_log_88 = np.zeros(PN_LIMIT, dtype=np.float64)

    # ---- (2,2) mode ----
    rho_22[2] = -43.0 / 42.0 + (55.0 * nu) / 84.0
    rho_22[3] = (-2.0 * (chiS + chiA * dm - chiS * nu)) / 3.0

    rho_22[4] = (
        -20555.0 / 10584.0
        + 0.5 * (chiS + chiA * dm) * (chiS + chiA * dm)
        - (33025.0 * nu) / 21168.0
        + (19583.0 * nu2) / 42336.0
    )

    rho_22[5] = (
        (-34.0 / 21.0 + 49.0 * nu / 18.0 + 209.0 * nu2 / 126.0) * chiS
        + (-34.0 / 21.0 - 19.0 * nu / 42.0) * dm * chiA
    )

    rho_22[6] = (
        1556919113.0 / 122245200.0
        + (89.0 * a2) / 252.0
        - (48993925.0 * nu) / 9779616.0
        - (6292061.0 * nu2) / 3259872.0
        + (10620745.0 * nu3) / 39118464.0
        + (41.0 * nu * PI * PI) / 192.0
    )
    rho_log_22[6] = -428.0 / 105.0

    rho_22[7] = (
        a3 / 3.0
        + chiA * dm * (18733.0 / 15876.0 + (50140.0 * nu) / 3969.0
                       + (97865.0 * nu2) / 63504.0)
        + chiS * (
            18733.0 / 15876.0
            + (74749.0 * nu) / 5292.0
            - (245717.0 * nu2) / 63504.0
            + (50803.0 * nu3) / 63504.0
        )
    )

    rho_22[8] = (
        -387216563023.0 / 160190110080.0
        + (18353.0 * a2) / 21168.0 - a2 * a2 / 8.0
    )
    rho_log_22[8] = 9202.0 / 2205.0

    rho_22[10] = -16094530514677.0 / 533967033600.0
    rho_log_22[10] = 439877.0 / 55566.0

    # extra_PN_terms for (2,2)
    if extra_PN_terms:
        rho_22[6] -= (89.0 * atemp * atemp) / 252.0
        rho_22[6] += (
            ((178.0 - 457.0 * nu - 972.0 * nu2) * chiA2) / 504.0
            + (dm * (178.0 - 781.0 * nu) * chiA * chiS) / 252.0
            + ((178.0 - 1817.0 * nu + 560.0 * nu2) * chiS2) / 504.0
        )
        rho_22[7] -= atemp ** 3 / 3.0
        rho_22[7] += (
            ((dm - 4.0 * dm * nu) * chiA3) / 3.0
            + (1.0 - 3.0 * nu - 4.0 * nu2) * chiA2 * chiS
            + (dm + 2.0 * dm * nu) * chiA * chiS2
            + (1.0 / 3.0 + nu) * chiS3
        )

    # ---- (2,1) mode ----
    # Zero test-spin terms
    a_21 = 0.0
    a2_21 = 0.0
    a3_21 = 0.0

    if dm2 > 1e-28:
        rho_21[1] = 0.0
        rho_21[2] = -59.0 / 56.0 + (23.0 * nu) / 84.0
        rho_21[3] = 0.0
        rho_21[4] = (
            -47009.0 / 56448.0
            - (865.0 * a2_21) / 1792.0
            - (405.0 * a2_21 * a2_21) / 2048.0
            - (10993.0 * nu) / 14112.0
            + (617.0 * nu2) / 4704.0
        )
        rho_21[5] = (
            (-98635.0 * a_21) / 75264.0
            + (2031.0 * a_21 * a2_21) / 7168.0
            - (1701.0 * a2_21 * a3_21) / 8192.0
        )
        rho_21[6] = (
            7613184941.0 / 2607897600.0
            + (9032393.0 * a2_21) / 1806336.0
            + (3897.0 * a2_21 * a2_21) / 16384.0
            - (15309.0 * a3_21 * a3_21) / 65536.0
        )
        rho_log_21[6] = -107.0 / 105.0
        rho_21[7] = (
            (-3859374457.0 * a_21) / 1159065600.0
            - (55169.0 * a3_21) / 16384.0
            + (18603.0 * a2_21 * a3_21) / 65536.0
            - (72171.0 * a2_21 * a2_21 * a3_21) / 262144.0
        )
        rho_log_21[7] = 107.0 * a_21 / 140.0
        rho_21[8] = -1168617463883.0 / 911303737344.0
        rho_log_21[8] = 6313.0 / 5880.0
        rho_21[10] = -63735873771463.0 / 16569158860800.0
        rho_log_21[10] = 5029963.0 / 5927040.0

        f_21[1] = (-3.0 * (chiS + chiA / dm)) / 2.0
        f_21[3] = (
            (chiS * dm * (427.0 + 79.0 * nu)
             + chiA * (147.0 + 280.0 * dm2 + 1251.0 * nu))
            / 84.0 / dm
        )
        f_21[4] = (
            (-3.0 - 2.0 * nu) * chiA2
            + (-3.0 + nu / 2.0) * chiS2
            + (-6.0 + 21.0 * nu / 2.0) * chiS * chiA / dm
        )
        f_21[5] = (
            (3.0 / 4.0 - 3.0 * nu) * chiA3 / dm
            + (
                -81.0 / 16.0
                + 1709.0 * nu / 1008.0
                + 613.0 * nu2 / 1008.0
                + (9.0 / 4.0 - 3.0 * nu) * chiA2
            ) * chiS
            + 3.0 / 4.0 * chiS3
            + (
                -81.0 / 16.0
                - 703.0 * nu2 / 112.0
                + 8797.0 * nu / 1008.0
                + (9.0 / 4.0 - 6.0 * nu) * chiS2
            ) * chiA / dm
        )
        f_21[6] = (
            ((16652.0 - 9287.0 * nu + 720.0 * nu2) * chiA2) / 1008.0
            + ((16652.0 - 39264.0 * nu + 9487.0 * nu2) * chiA * chiS) / (504.0 * dm)
            + ((16652.0 - 2633.0 * nu + 1946.0 * nu2) * chiS2) / 1008.0
        )
    else:
        # Equal mass
        f_21[1] = -3.0 * chiA / 2.0
        f_21[3] = (
            chiS * dm * (427.0 + 79.0 * nu)
            + chiA * (147.0 + 280.0 * dm2 + 1251.0 * nu)
        ) / 84.0
        f_21[4] = (-6.0 + 21.0 * nu / 2.0) * chiS * chiA
        f_21[5] = (
            (3.0 / 4.0 - 3.0 * nu) * chiA3
            + (
                -81.0 / 16.0
                - 703.0 * nu2 / 112.0
                + 8797.0 * nu / 1008.0
                + (9.0 / 4.0 - 6.0 * nu) * chiS2
            ) * chiA
        )
        f_21[6] = ((16652.0 - 39264.0 * nu + 9487.0 * nu2) * chiA * chiS) / 504.0

    # ---- (3,3) mode ----
    a_33 = 0.0
    a2_33 = 0.0
    a3_33 = 0.0

    if dm2 > 1e-28:
        rho_33[2] = -7.0 / 6.0 + (2.0 * nu) / 3.0
        rho_33[3] = 0.0
        rho_33[4] = (
            -6719.0 / 3960.0
            + a2_33 / 2.0
            - (1861.0 * nu) / 990.0
            + (149.0 * nu2) / 330.0
        )
        rho_33[5] = (-4.0 * a_33) / 3.0
        rho_33[6] = (
            3203101567.0 / 227026800.0
            + (5.0 * a2_33) / 36.0
            + (-129509.0 / 25740.0 + 41.0 / 192.0 * PI * PI) * nu
            - 274621.0 / 154440.0 * nu2
            + 12011.0 / 46332.0 * nu3
        )
        rho_log_33[6] = -26.0 / 7.0
        rho_33[7] = (5297.0 * a_33) / 2970.0 + a_33 * a2_33 / 3.0
        rho_33[8] = -57566572157.0 / 8562153600.0
        rho_log_33[8] = 13.0 / 3.0
        rho_33[10] = -903823148417327.0 / 30566888352000.0
        rho_log_33[10] = 87347.0 / 13860.0

        f_33[3] = (
            chiS * dm * (-4.0 + 5.0 * nu) + chiA * (-4.0 + 19.0 * nu)
        ) / (2.0 * dm)
        f_33[4] = (
            3.0 / 2.0 * chiS2 * dm
            + (3.0 - 12.0 * nu) * chiA * chiS
            + dm * (3.0 / 2.0 - 6.0 * nu) * chiA2
        ) / dm
        f_33[5] = (
            dm * (241.0 / 30.0 * nu2 + 11.0 / 20.0 * nu + 2.0 / 3.0) * chiS
            + (407.0 / 30.0 * nu2 - 593.0 / 60.0 * nu + 2.0 / 3.0) * chiA
        ) / dm
        f_33[6] = (
            dm * (6.0 * nu2 - 27.0 / 2.0 * nu - 7.0 / 4.0) * chiS2
            + (44.0 * nu2 - 1.0 * nu - 7.0 / 2.0) * chiA * chiS
            + dm * (-12.0 * nu2 + 11.0 / 2.0 * nu - 7.0 / 4.0) * chiA2
        ) / dm
        f_vh_33_6 = (
            dm * (593.0 / 108.0 * nu - 81.0 / 20.0) * chiS
            + (7339.0 / 540.0 * nu - 81.0 / 20.0) * chiA
        ) / dm
    else:
        # Equal mass
        f_33[3] = chiA * 3.0 / 8.0
        f_33[4] = (3.0 - 12.0 * nu) * chiA * chiS
        f_33[5] = (407.0 / 30.0 * nu2 - 593.0 / 60.0 * nu + 2.0 / 3.0) * chiA
        f_33[6] = (44.0 * nu2 - 1.0 * nu - 7.0 / 2.0) * chiA * chiS
        f_vh_33_6 = (7339.0 / 540.0 * nu - 81.0 / 20.0) * chiA

    # ---- (3,2) mode ----
    # Note: test-spin terms are already zeroed (a=0 since (2,1) block)
    # But (3,2) in pySEOBNR uses the original 'a' for some terms.
    # Looking at the source: a2 is used (which was zeroed), but also a2/3 in rho[4].
    # The pySEOBNR code zeroes a before (2,1) and keeps it zeroed for all subsequent modes.
    # So a=0, a2=0, a3=0 for (3,2) rho_coeffs.
    a_32 = 0.0
    a2_32 = 0.0

    rho_32[1] = (4.0 * chiS * nu) / (-3.0 * m1Plus3nu)
    rho_32[2] = (328.0 - 1115.0 * nu + 320.0 * nu2) / (270.0 * m1Plus3nu)

    rho_32[3] = (
        2.0
        * (
            45.0 * a_32 * m1Plus3nu3
            - a_32
            * nu
            * (328.0 - 2099.0 * nu + 5.0 * (733.0 + 20.0 * a2_32) * nu2 - 960.0 * nu3)
        )
    ) / (405.0 * m1Plus3nu3)

    rho_32[4] = a2_32 / 3.0 + (
        -1444528.0
        + 8050045.0 * nu
        - 4725605.0 * nu2
        - 20338960.0 * nu3
        + 3085640.0 * nu4
    ) / (1603800.0 * m1Plus3nu2)
    rho_32[5] = (-2788.0 * a_32) / 1215.0
    rho_32[6] = 5849948554.0 / 940355325.0 + (488.0 * a2_32) / 405.0
    rho_log_32[6] = -104.0 / 63.0
    rho_32[8] = -10607269449358.0 / 3072140846775.0
    rho_log_32[8] = 17056.0 / 8505.0

    # ---- (4,3) mode ----
    a_43 = 0.0
    a2_43 = 0.0
    a3_43 = 0.0

    if dm2 > 1e-28:
        rho_43[1] = 0.0
        rho_43[2] = (222.0 - 547.0 * nu + 160.0 * nu2) / (
            176.0 * (-1.0 + 2.0 * nu)
        )
        rho_43[4] = -6894273.0 / 7047040.0 + (3.0 * a2_43) / 8.0
        rho_43[5] = (-12113.0 * a_43) / 6160.0
        rho_43[6] = 1664224207351.0 / 195343948800.0
        rho_log_43[6] = -1571.0 / 770.0
        f_43[1] = (5.0 * (chiA - chiS * dm) * nu) / (
            2.0 * dm * (-1.0 + 2.0 * nu)
        )
    else:
        # Equal mass
        f_43[1] = -5.0 * chiA / 4.0

    # ---- (4,4) mode ----
    # a is still 0 (zeroed test-spin), but (4,4) uses original a2 for some terms
    # In pySEOBNR the a was zeroed before (2,1) and stays zeroed.
    a_44 = 0.0
    a2_44 = 0.0

    rho_44[2] = (1614.0 - 5870.0 * nu + 2625.0 * nu2) / (1320.0 * m1Plus3nu)
    rho_44[3] = (
        chiA * (10.0 - 39.0 * nu) * dm
        + chiS * (10.0 - 41.0 * nu + 42.0 * nu2)
    ) / (15.0 * m1Plus3nu)

    rho_44[4] = (
        (
            -511573572.0
            + 2338945704.0 * nu
            - 313857376.0 * nu2
            - 6733146000.0 * nu3
            + 1252563795.0 * nu4
        )
        / (317116800.0 * m1Plus3nu2)
        + chiS2 / 2.0
        + dm * chiS * chiA
        + dm2 * chiA2 / 2.0
    )
    rho_44[5] = chiA * dm * (
        -8280.0 + 42716.0 * nu - 57990.0 * nu2 + 8955.0 * nu3
    ) / (6600.0 * m1Plus3nu2) + chiS * (
        -8280.0
        + 66284.0 * nu
        - 176418.0 * nu2
        + 128085.0 * nu3
        + 88650.0 * nu4
    ) / (
        6600.0 * m1Plus3nu2
    )
    rho_44[6] = 16600939332793.0 / 1098809712000.0 + (217.0 * a2_44) / 3960.0
    rho_log_44[6] = -12568.0 / 3465.0
    rho_44[8] = -172066910136202271.0 / 19426955708160000.0
    rho_log_44[8] = 845198.0 / 190575.0
    rho_44[10] = -17154485653213713419357.0 / 568432724020761600000.0
    rho_log_44[10] = 22324502267.0 / 3815311500.0

    # ---- (5,5) mode ----
    # From pySEOBNR: only nonzero when dm2 > 0 (unequal masses)
    # a is already zeroed (test-spin terms zeroed before (2,1) block)
    a_55 = 0.0
    a2_55 = 0.0

    if dm2 > 1e-28:
        rho_55[2] = (487.0 - 1298.0 * nu + 512.0 * nu2) / (
            390.0 * (-1.0 + 2.0 * nu)
        )
        rho_55[3] = (-2.0 * a_55) / 3.0
        rho_55[4] = -3353747.0 / 2129400.0 + a2_55 / 2.0
        rho_55[5] = -241.0 * a_55 / 195.0

        # Higher-order terms from Eq.A9 in PRD 98, 084028
        rho_55[6] = 190606537999247.0 / 11957879934000.0
        rho_log_55[6] = -1546.0 / 429.0
        rho_55[8] = -1213641959949291437.0 / 118143853747920000.0
        rho_log_55[8] = 376451.0 / 83655.0
        rho_55[10] = -150082616449726042201261.0 / 4837990810977324000000.0
        rho_log_55[10] = 2592446431.0 / 456756300.0

    # ---- (6,6) mode ----
    # Even m => no f_coeffs. a is zeroed.
    a_66 = 0.0
    a2_66 = 0.0

    rho_66[2] = (-106.0 + 602.0 * nu - 861.0 * nu2 + 273.0 * nu3) / (
        84.0 * (1.0 - 5.0 * nu + 5.0 * nu2)
    )
    rho_66[3] = (-2.0 * a_66) / 3.0
    rho_66[4] = -1025435.0 / 659736.0 + a2_66 / 2.0

    # ---- (3,1) mode ----
    # a is zeroed (same as all modes after (2,2))
    a_31 = 0.0
    a2_31 = 0.0

    if dm2 > 1e-28:
        rho_31[2] = -13.0 / 18.0 - (2.0 * nu) / 9.0
        rho_31[3] = 0.0
        rho_31[4] = (
            101.0 / 7128.0
            - (5.0 * a2_31) / 6.0
            - (1685.0 * nu) / 1782.0
            - (829.0 * nu2) / 1782.0
        )
        rho_31[5] = (4.0 * a_31) / 9.0
        rho_31[6] = 11706720301.0 / 6129723600.0 - (49.0 * a2_31) / 108.0
        rho_log_31[6] = -26.0 / 63.0
        rho_31[7] = (-2579.0 * a_31) / 5346.0 + a_31 * a2_31 / 9.0
        rho_31[8] = 2606097992581.0 / 4854741091200.0
        rho_log_31[8] = 169.0 / 567.0

        f_31[3] = (
            chiA * (-4.0 + 11.0 * nu) + chiS * dm * (-4.0 + 13.0 * nu)
        ) / (2.0 * dm)
    else:
        f_31[3] = -chiA * 5.0 / 8.0

    # ---- (4,1) mode ----
    a_41 = 0.0
    a2_41 = 0.0

    if dm2 > 1e-28:
        rho_41[1] = 0.0
        rho_41[2] = (602.0 - 1385.0 * nu + 288.0 * nu2) / (
            528.0 * (-1.0 + 2.0 * nu)
        )
        rho_41[4] = -7775491.0 / 21141120.0 + (3.0 * a2_41) / 8.0
        rho_41[5] = (-20033.0 * a_41) / 55440.0 - (5.0 * a_41 * a2_41) / 6.0
        rho_41[6] = 1227423222031.0 / 1758095539200.0
        rho_log_41[6] = -1571.0 / 6930.0

        f_41[1] = (5.0 * (chiA - chiS * dm) * nu) / (
            2.0 * dm * (-1.0 + 2.0 * nu)
        )
    else:
        f_41[1] = -5.0 * chiA / 4.0

    # ---- (4,2) mode ----
    # Even m => no f_coeffs.
    a_42 = 0.0
    a2_42 = 0.0

    rho_42[2] = (1146.0 - 3530.0 * nu + 285.0 * nu2) / (1320.0 * m1Plus3nu)
    rho_42[3] = (
        chiA * (10.0 - 21.0 * nu) * dm
        + chiS * (10.0 - 59.0 * nu + 78.0 * nu2)
    ) / (15.0 * m1Plus3nu)
    rho_42[4] = a2_42 / 2.0 + (
        -114859044.0
        + 295834536.0 * nu
        + 1204388696.0 * nu2
        - 3047981160.0 * nu3
        - 379526805.0 * nu4
    ) / (317116800.0 * m1Plus3nu2)
    rho_42[5] = (-7.0 * a_42) / 110.0
    rho_42[6] = 848238724511.0 / 219761942400.0 + (2323.0 * a2_42) / 3960.0
    rho_log_42[6] = -3142.0 / 3465.0

    # ---- (5,2) mode ----
    # Even m => no f_coeffs. a is zeroed.
    a_52 = 0.0
    a2_52 = 0.0

    rho_52[2] = (
        -15828.0 + 84679.0 * nu - 104930.0 * nu2 + 21980.0 * nu3
    ) / (13650.0 * (1.0 - 5.0 * nu + 5.0 * nu2))
    rho_52[3] = (-2.0 * a_52) / 15.0
    rho_52[4] = -7187914.0 / 15526875.0 + (2.0 * a2_52) / 5.0

    # ---- (5,3) mode ----
    # Odd m but no f_coeffs in pySEOBNR. a is zeroed.
    a_53 = 0.0
    a2_53 = 0.0

    if dm2 > 1e-28:
        rho_53[2] = (375.0 - 850.0 * nu + 176.0 * nu2) / (
            390.0 * (-1.0 + 2.0 * nu)
        )
        rho_53[3] = (-2.0 * a_53) / 3.0
        rho_53[4] = -410833.0 / 709800.0 + a2_53 / 2.0
        rho_53[5] = -103.0 * a_53 / 325.0

    # ---- (5,4) mode ----
    # Even m => no f_coeffs. a is zeroed.
    a_54 = 0.0
    a2_54 = 0.0

    rho_54[2] = (
        -17448.0 + 96019.0 * nu - 127610.0 * nu2 + 33320.0 * nu3
    ) / (13650.0 * (1.0 - 5.0 * nu + 5.0 * nu2))
    rho_54[3] = (-2.0 * a_54) / 15.0
    rho_54[4] = -16213384.0 / 15526875.0 + (2.0 * a2_54) / 5.0

    # ---- (7,7) mode ----
    # Odd m but no f_coeffs in pySEOBNR. a is zeroed.
    a_77 = 0.0

    if dm2 > 1e-28:
        rho_77[2] = (-906.0 + 4246.0 * nu - 4963.0 * nu2 + 1380.0 * nu3) / (
            714.0 * (dm2 + 3.0 * nu2)
        )
        rho_77[3] = -2.0 * a_77 / 3.0

    # ---- (8,8) mode ----
    # Even m => no f_coeffs. a is zeroed.

    rho_88[2] = (
        3482.0 - 26778.0 * nu + 64659.0 * nu2
        - 53445.0 * nu3 + 12243.0 * nu4
    ) / (2736.0 * (-1.0 + 7.0 * nu - 14.0 * nu2 + 7.0 * nu3))

    return (rho_22, rho_log_22,
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


# Backward-compatible wrappers
@njit(cache=True, fastmath=True)
def compute_rho_coeffs_8mode(nu, dm, a, chiS, chiA, extra_PN_terms):
    """Backward-compatible 8-mode rho coefficients wrapper."""
    result = compute_rho_coeffs_16mode(nu, dm, a, chiS, chiA, extra_PN_terms)
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
    return (rho_22, rho_log_22,
            rho_21, rho_log_21, f_21,
            rho_33, rho_log_33, f_33, f_vh_33_6,
            rho_32, rho_log_32,
            rho_43, rho_log_43, f_43,
            rho_44, rho_log_44,
            rho_55, rho_log_55,
            rho_66, rho_log_66)


@njit(cache=True, fastmath=True)
def compute_rho_coeffs_6mode(nu, dm, a, chiS, chiA, extra_PN_terms):
    """Backward-compatible 6-mode rho coefficients wrapper."""
    result = compute_rho_coeffs_8mode(nu, dm, a, chiS, chiA, extra_PN_terms)
    return (result[0], result[1],    # rho_22, rho_log_22
            result[2], result[3], result[4],  # rho_21, rho_log_21, f_21
            result[5], result[6], result[7], result[8],  # rho_33, rho_log_33, f_33, f_vh_33_6
            result[9], result[10],   # rho_32, rho_log_32
            result[11], result[12], result[13],  # rho_43, rho_log_43, f_43
            result[14], result[15])  # rho_44, rho_log_44


@njit(cache=True, fastmath=True)
def compute_rho_coeffs_3mode(nu, dm, a, chiS, chiA, extra_PN_terms):
    """Backward-compatible 3-mode rho coefficients wrapper."""
    result = compute_rho_coeffs_8mode(nu, dm, a, chiS, chiA, extra_PN_terms)
    return (result[0], result[1],    # rho_22, rho_log_22
            result[2], result[3], result[4],  # rho_21, rho_log_21, f_21
            result[5], result[6], result[7], result[8])  # rho_33, rho_log_33, f_33, f_vh_33_6


# ============================================================================
# compute_rholm_single: rho_lm^l for each mode
# ============================================================================
@njit(cache=True, fastmath=True)
def compute_rholm_single_22(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_22^2 (the resummed PN factor) for the (2,2) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 2
    m = 2
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    # Build powers of v
    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_21(v, vh, nu, rho_coeffs, rho_log_coeffs, f_coeffs):
    """
    Compute rho_21^2 + f_21 for the (2,1) mode.
    Odd m => has f_coeffs. For equal mass (nu=0.25), rho_final = f_final.
    """
    l = 2
    m = 1
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    f = 0.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]
        f += f_coeffs[j] * vs[j]

    rho_l = rho ** l
    if abs(nu - 0.25) < 1e-14:
        return f
    else:
        return rho_l + f


@njit(cache=True, fastmath=True)
def compute_rholm_single_33(v, vh, nu, rho_coeffs, rho_log_coeffs,
                            f_coeffs, f_vh_33_6):
    """
    Compute rho_33^3 + f_33 for the (3,3) mode.
    Odd m => has f_coeffs.
    Also has the complex f_vh term: f_final += i * vh^6 * f_vh_33_6.
    For flux we need |rho_final|^2.

    Returns (real_part, imag_part) of the rho_final.
    """
    l = 3
    m = 3
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    f = 0.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]
        f += f_coeffs[j] * vs[j]

    # Complex extra term: i * vh^6 * f_vh_33_6
    vh3 = vh ** 3
    vh6 = vh3 * vh3
    extra_imag = vh6 * f_vh_33_6

    rho_l = rho ** l
    if abs(nu - 0.25) < 1e-14:
        real_part = f
    else:
        real_part = rho_l + f
    imag_part = extra_imag

    return real_part, imag_part


@njit(cache=True, fastmath=True)
def compute_rholm_single_32(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_32^3 for the (3,2) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 3
    m = 2
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_43(v, vh, nu, rho_coeffs, rho_log_coeffs, f_coeffs):
    """
    Compute rho_43^4 + f_43 for the (4,3) mode.
    Odd m => has f_coeffs. No f_vh term (only (3,3) has that).

    Returns a real number.
    """
    l = 4
    m = 3
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    f = 0.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]
        f += f_coeffs[j] * vs[j]

    rho_l = rho ** l
    if abs(nu - 0.25) < 1e-14:
        return f
    else:
        return rho_l + f


@njit(cache=True, fastmath=True)
def compute_rholm_single_44(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_44^4 for the (4,4) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 4
    m = 4
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_55(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_55^5 for the (5,5) mode.
    Odd m, but treated as rho-only (no f_coeffs or f_vh terms).

    Returns a real number (rho^l).
    """
    l = 5
    m = 5
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_66(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_66^6 for the (6,6) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 6
    m = 6
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_31(v, vh, nu, rho_coeffs, rho_log_coeffs, f_coeffs):
    """
    Compute rho_31^3 + f_31 for the (3,1) mode.
    Odd m => has f_coeffs.

    Returns a real number.
    """
    l = 3
    m = 1
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    f = 0.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]
        f += f_coeffs[j] * vs[j]

    rho_l = rho ** l
    if abs(nu - 0.25) < 1e-14:
        return f
    else:
        return rho_l + f


@njit(cache=True, fastmath=True)
def compute_rholm_single_41(v, vh, nu, rho_coeffs, rho_log_coeffs, f_coeffs):
    """
    Compute rho_41^4 + f_41 for the (4,1) mode.
    Odd m => has f_coeffs.

    Returns a real number.
    """
    l = 4
    m = 1
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    f = 0.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]
        f += f_coeffs[j] * vs[j]

    rho_l = rho ** l
    if abs(nu - 0.25) < 1e-14:
        return f
    else:
        return rho_l + f


@njit(cache=True, fastmath=True)
def compute_rholm_single_42(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_42^4 for the (4,2) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 4
    m = 2
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_52(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_52^5 for the (5,2) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 5
    m = 2
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_53(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_53^5 for the (5,3) mode.
    Odd m but no f_coeffs in pySEOBNR for this mode.

    Returns a real number (rho^l).
    """
    l = 5
    m = 3
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_54(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_54^5 for the (5,4) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 5
    m = 4
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_77(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_77^7 for the (7,7) mode.
    Odd m but no f_coeffs in pySEOBNR for this mode.

    Returns a real number (rho^l).
    """
    l = 7
    m = 7
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l


@njit(cache=True, fastmath=True)
def compute_rholm_single_88(v, vh, nu, rho_coeffs, rho_log_coeffs):
    """
    Compute rho_88^8 for the (8,8) mode.
    Even m => no f_coeffs.

    Returns a real number (rho^l).
    """
    l = 8
    m = 8
    eulerlogxabs = EULER_GAMMA + math.log(2.0 * m * v)

    vs = np.empty(PN_LIMIT, dtype=np.float64)
    vs[0] = 1.0
    vs[1] = v
    for i in range(2, PN_LIMIT):
        vs[i] = v * vs[i - 1]

    rho = 1.0
    for j in range(1, PN_LIMIT):
        rho += (rho_coeffs[j] + rho_log_coeffs[j] * eulerlogxabs) * vs[j]

    return rho ** l
