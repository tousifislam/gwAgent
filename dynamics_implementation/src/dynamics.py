"""
Top-level dynamics: assembles the full 6-variable EOB RHS and integrates.

Uses split real/imag arrays for complex mode coefficients to stay in float64 world.
Uses 16 modes: (2,2), (2,1), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (4,4),
               (5,2), (5,3), (5,4), (5,5), (6,6), (7,7), (8,8).
"""

import numpy as np
from numba import njit

from hamiltonian import evaluate_H, ham_and_derivs
from fits import a6_NS, dSO, GSF_amplitude_fits_numba
from evolution_equations import initialize_keplerian_coeffs, compute_edot_zdot_xavg
from flux import precompute_waveform_statics, compute_flux
from ecc_corrections import initialize_rr_force_coeffs, compute_rr_force_corrections
from ecc_mode_corrections import initialize_ecc_mode_coeffs, compute_ecc_mode_corrections
from integrator import integrate_adaptive


@njit(cache=True, fastmath=True)
def rhs_ecc(t, y, params):
    """
    Full 6-variable eccentric EOB RHS.
    y = [r, phi, pr, pphi, e, zeta]
    params = packed float64 array containing all coefficients
    """
    r = y[0]; phi = y[1]; pr = y[2]; pphi = y[3]
    e = y[4]; z = y[5]

    # Unpack scalar parameters (first 24 floats)
    chi_1 = params[0]; chi_2 = params[1]
    m_1 = params[2]; m_2 = params[3]; nu = params[4]
    a6v = params[5]; dsov = params[6]
    prefix_22 = params[7]; prefix_21 = params[8]
    prefix_31 = params[9]; prefix_33 = params[10]; prefix_32 = params[11]
    prefix_41 = params[12]; prefix_42 = params[13]
    prefix_43 = params[14]; prefix_44 = params[15]
    prefix_52 = params[16]; prefix_53 = params[17]
    prefix_54 = params[18]; prefix_55 = params[19]
    prefix_66 = params[20]; prefix_77 = params[21]; prefix_88 = params[22]
    f_vh_33_6 = params[23]

    # Unpack array sizes (next 39 ints encoded as floats)
    base = 24
    n_kep = int(params[base]); n_rr = int(params[base+1])
    n_rho22 = int(params[base+2]); n_rlog22 = int(params[base+3])
    n_rho21 = int(params[base+4]); n_rlog21 = int(params[base+5])
    n_f21 = int(params[base+6])
    n_rho31 = int(params[base+7]); n_rlog31 = int(params[base+8])
    n_f31 = int(params[base+9])
    n_rho33 = int(params[base+10]); n_rlog33 = int(params[base+11])
    n_f33 = int(params[base+12])
    n_rho32 = int(params[base+13]); n_rlog32 = int(params[base+14])
    n_rho41 = int(params[base+15]); n_rlog41 = int(params[base+16])
    n_f41 = int(params[base+17])
    n_rho42 = int(params[base+18]); n_rlog42 = int(params[base+19])
    n_rho43 = int(params[base+20]); n_rlog43 = int(params[base+21])
    n_f43 = int(params[base+22])
    n_rho44 = int(params[base+23]); n_rlog44 = int(params[base+24])
    n_rho52 = int(params[base+25]); n_rlog52 = int(params[base+26])
    n_rho53 = int(params[base+27]); n_rlog53 = int(params[base+28])
    n_rho54 = int(params[base+29]); n_rlog54 = int(params[base+30])
    n_rho55 = int(params[base+31]); n_rlog55 = int(params[base+32])
    n_rho66 = int(params[base+33]); n_rlog66 = int(params[base+34])
    n_rho77 = int(params[base+35]); n_rlog77 = int(params[base+36])
    n_rho88 = int(params[base+37]); n_rlog88 = int(params[base+38])
    n_mode_re = int(params[base+39]); n_mode_im = int(params[base+40])

    # Unpack coefficient arrays
    offset = base + 41
    kep_coeffs = params[offset:offset+n_kep]; offset += n_kep
    rr_coeffs = params[offset:offset+n_rr]; offset += n_rr
    rho_22 = params[offset:offset+n_rho22]; offset += n_rho22
    rho_log_22 = params[offset:offset+n_rlog22]; offset += n_rlog22
    rho_21 = params[offset:offset+n_rho21]; offset += n_rho21
    rho_log_21 = params[offset:offset+n_rlog21]; offset += n_rlog21
    f_21 = params[offset:offset+n_f21]; offset += n_f21
    rho_31 = params[offset:offset+n_rho31]; offset += n_rho31
    rho_log_31 = params[offset:offset+n_rlog31]; offset += n_rlog31
    f_31 = params[offset:offset+n_f31]; offset += n_f31
    rho_33 = params[offset:offset+n_rho33]; offset += n_rho33
    rho_log_33 = params[offset:offset+n_rlog33]; offset += n_rlog33
    f_33 = params[offset:offset+n_f33]; offset += n_f33
    rho_32 = params[offset:offset+n_rho32]; offset += n_rho32
    rho_log_32 = params[offset:offset+n_rlog32]; offset += n_rlog32
    rho_41 = params[offset:offset+n_rho41]; offset += n_rho41
    rho_log_41 = params[offset:offset+n_rlog41]; offset += n_rlog41
    f_41 = params[offset:offset+n_f41]; offset += n_f41
    rho_42 = params[offset:offset+n_rho42]; offset += n_rho42
    rho_log_42 = params[offset:offset+n_rlog42]; offset += n_rlog42
    rho_43 = params[offset:offset+n_rho43]; offset += n_rho43
    rho_log_43 = params[offset:offset+n_rlog43]; offset += n_rlog43
    f_43 = params[offset:offset+n_f43]; offset += n_f43
    rho_44 = params[offset:offset+n_rho44]; offset += n_rho44
    rho_log_44 = params[offset:offset+n_rlog44]; offset += n_rlog44
    rho_52 = params[offset:offset+n_rho52]; offset += n_rho52
    rho_log_52 = params[offset:offset+n_rlog52]; offset += n_rlog52
    rho_53 = params[offset:offset+n_rho53]; offset += n_rho53
    rho_log_53 = params[offset:offset+n_rlog53]; offset += n_rlog53
    rho_54 = params[offset:offset+n_rho54]; offset += n_rho54
    rho_log_54 = params[offset:offset+n_rlog54]; offset += n_rlog54
    rho_55 = params[offset:offset+n_rho55]; offset += n_rho55
    rho_log_55 = params[offset:offset+n_rlog55]; offset += n_rlog55
    rho_66 = params[offset:offset+n_rho66]; offset += n_rho66
    rho_log_66 = params[offset:offset+n_rlog66]; offset += n_rlog66
    rho_77 = params[offset:offset+n_rho77]; offset += n_rho77
    rho_log_77 = params[offset:offset+n_rlog77]; offset += n_rlog77
    rho_88 = params[offset:offset+n_rho88]; offset += n_rho88
    rho_log_88 = params[offset:offset+n_rlog88]; offset += n_rlog88
    mode_re = params[offset:offset+n_mode_re]; offset += n_mode_re
    mode_im = params[offset:offset+n_mode_im]; offset += n_mode_im

    # Reconstruct complex mode coefficients
    n_mc = len(mode_re)
    mode_coeffs = np.empty(n_mc, dtype=np.complex128)
    for i in range(n_mc):
        mode_coeffs[i] = complex(mode_re[i], mode_im[i])

    # 1. Hamiltonian dynamics
    dHdr, dHdphi, dHdpr, omega, H_val, xi = ham_and_derivs(
        r, phi, pr, pphi, chi_1, chi_2, m_1, m_2, nu, a6v, dsov)

    # 2. Evolution equations
    edot, zdot, xavg = compute_edot_zdot_xavg(e, z, omega, kep_coeffs)

    # 3. Eccentric mode corrections (full PN, 16 modes)
    # Returns 32 floats: re/im pairs in order
    # (2,2), (2,1), (3,1), (3,2), (3,3),
    # (4,1), (4,2), (4,3), (4,4),
    # (5,2), (5,3), (5,4), (5,5),
    # (6,6), (7,7), (8,8)
    (h22_re, h22_im, h21_re, h21_im,
     h31_re, h31_im, h32_re, h32_im, h33_re, h33_im,
     h41_re, h41_im, h42_re, h42_im, h43_re, h43_im, h44_re, h44_im,
     h52_re, h52_im, h53_re, h53_im, h54_re, h54_im, h55_re, h55_im,
     h66_re, h66_im, h77_re, h77_im, h88_re, h88_im) = \
        compute_ecc_mode_corrections(e, z, xavg, mode_coeffs)

    # 4. Flux (16-mode)
    H_times_nu = nu * H_val
    flux = compute_flux(
        omega, e, z, xavg, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_31, prefix_33, prefix_32,
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
        rho_88, rho_log_88,
        h22_re, h22_im, h21_re, h21_im,
        h31_re, h31_im, h32_re, h32_im, h33_re, h33_im,
        h41_re, h41_im, h42_re, h42_im, h43_re, h43_im, h44_re, h44_im,
        h52_re, h52_im, h53_re, h53_im, h54_re, h54_im, h55_re, h55_im,
        h66_re, h66_im, h77_re, h77_im, h88_re, h88_im)

    # 5. RR force corrections
    Fr_corr, Fphi_corr = compute_rr_force_corrections(e, z, xavg, rr_coeffs)

    # 6. RR force
    flux_norm = flux / nu
    f_over_om = flux_norm / omega
    Fr = -pr / pphi * f_over_om * Fr_corr
    Fphi = -f_over_om * Fphi_corr

    # 7. Assemble
    out = np.empty(6)
    out[0] = xi * dHdpr
    out[1] = omega
    out[2] = -dHdr * xi + Fr
    out[3] = Fphi
    out[4] = edot
    out[5] = zdot
    return out


def setup_and_integrate(q, chi_1, chi_2, r0, pr0, pphi0, e0, zeta0,
                        t_end=1e7, rtol=1e-8, atol=1e-9, r_stop=2.5,
                        max_steps=100000):
    """Setup parameters, pack into flat array, and integrate."""
    m_1 = q / (1.0 + q)
    m_2 = 1.0 / (1.0 + q)
    nu = m_1 * m_2
    delta = m_1 - m_2
    ap = chi_1 * m_1 + chi_2 * m_2
    am = chi_1 * m_1 - chi_2 * m_2
    chiS = (chi_1 + chi_2) / 2.0
    chiA = (chi_1 - chi_2) / 2.0

    a6v = a6_NS(nu)
    dsov = dSO(nu, ap, am)

    kep_coeffs = initialize_keplerian_coeffs(nu, delta, chiA, chiS)
    rr_coeffs = initialize_rr_force_coeffs(nu)
    mode_coeffs_c = initialize_ecc_mode_coeffs(nu)
    mode_re = mode_coeffs_c.real.copy()
    mode_im = mode_coeffs_c.imag.copy()

    statics = precompute_waveform_statics(m_1, m_2, nu, delta, ap, chiS, chiA, True)
    (prefix_22, prefix_21, prefix_31, prefix_33, prefix_32,
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
     rho_88, rho_log_88) = statics

    # Pack everything into one float64 array
    sizes = np.array([
        len(kep_coeffs), len(rr_coeffs),
        len(rho_22), len(rho_log_22),
        len(rho_21), len(rho_log_21), len(f_21),
        len(rho_31), len(rho_log_31), len(f_31),
        len(rho_33), len(rho_log_33), len(f_33),
        len(rho_32), len(rho_log_32),
        len(rho_41), len(rho_log_41), len(f_41),
        len(rho_42), len(rho_log_42),
        len(rho_43), len(rho_log_43), len(f_43),
        len(rho_44), len(rho_log_44),
        len(rho_52), len(rho_log_52),
        len(rho_53), len(rho_log_53),
        len(rho_54), len(rho_log_54),
        len(rho_55), len(rho_log_55),
        len(rho_66), len(rho_log_66),
        len(rho_77), len(rho_log_77),
        len(rho_88), len(rho_log_88),
        len(mode_re), len(mode_im),
    ], dtype=np.float64)

    scalars = np.array([chi_1, chi_2, m_1, m_2, nu, a6v, dsov,
                        prefix_22, prefix_21,
                        prefix_31, prefix_33, prefix_32,
                        prefix_41, prefix_42, prefix_43, prefix_44,
                        prefix_52, prefix_53, prefix_54, prefix_55,
                        prefix_66, prefix_77, prefix_88,
                        f_vh_33_6])

    params = np.concatenate([scalars, sizes,
                             kep_coeffs, rr_coeffs,
                             rho_22, rho_log_22,
                             rho_21, rho_log_21, f_21,
                             rho_31, rho_log_31, f_31,
                             rho_33, rho_log_33, f_33,
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
                             mode_re, mode_im])

    y0 = np.array([r0, 0.0, pr0, pphi0, e0, zeta0])

    t_arr, y_arr = integrate_adaptive(rhs_ecc, 0.0, y0, t_end, params,
                                      rtol=rtol, atol=atol,
                                      max_steps=max_steps, r_stop=r_stop)

    # Compute x post-hoc
    n = len(t_arr)
    x_arr = np.empty(n)
    omega_arr = np.empty(n)
    for i in range(n):
        _, _, _, omi, _, _ = ham_and_derivs(
            y_arr[i,0], y_arr[i,1], y_arr[i,2], y_arr[i,3],
            chi_1, chi_2, m_1, m_2, nu, a6v, dsov)
        omega_arr[i] = omi
        _, _, xi = compute_edot_zdot_xavg(y_arr[i,4], y_arr[i,5], omi, kep_coeffs)
        x_arr[i] = xi

    return {
        't': t_arr, 'r': y_arr[:,0], 'phi': y_arr[:,1],
        'pr': y_arr[:,2], 'pphi': y_arr[:,3],
        'e': y_arr[:,4], 'zeta': y_arr[:,5],
        'x': x_arr, 'omega': omega_arr,
    }
