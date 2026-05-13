"""
6-mode variant of dynamics.py — uses only (2,2), (2,1), (3,3), (3,2),
(4,3), (4,4) modes in the flux computation.  ~3x fewer mode evaluations
per RHS call → proportional speedup.

Drop-in replacement: setup_and_integrate_6mode has the same signature
and return value as setup_and_integrate.
"""
import numpy as np
from numba import njit

from hamiltonian import evaluate_H, ham_and_derivs
from fits import a6_NS, dSO
from evolution_equations import initialize_keplerian_coeffs, compute_edot_zdot_xavg
from flux import precompute_waveform_statics, compute_flux_6mode
from ecc_corrections import initialize_rr_force_coeffs, compute_rr_force_corrections
from ecc_mode_corrections import initialize_ecc_mode_coeffs, compute_ecc_mode_corrections
from integrator import integrate_adaptive


@njit(cache=True, fastmath=True)
def rhs_ecc_6mode(t, y, params):
    """6-mode eccentric EOB RHS.  y = [r, phi, pr, pphi, e, zeta]."""
    r = y[0]; phi = y[1]; pr = y[2]; pphi = y[3]
    e = y[4]; z = y[5]

    # Unpack scalars (14 floats)
    chi_1 = params[0]; chi_2 = params[1]
    m_1 = params[2]; m_2 = params[3]; nu = params[4]
    a6v = params[5]; dsov = params[6]
    prefix_22 = params[7]; prefix_21 = params[8]; prefix_33 = params[9]
    prefix_32 = params[10]; prefix_43 = params[11]; prefix_44 = params[12]
    f_vh_33_6 = params[13]

    # Unpack sizes (19 ints as floats)
    base = 14
    n_kep = int(params[base]); n_rr = int(params[base+1])
    n_rho22 = int(params[base+2]); n_rlog22 = int(params[base+3])
    n_rho21 = int(params[base+4]); n_rlog21 = int(params[base+5])
    n_f21 = int(params[base+6])
    n_rho33 = int(params[base+7]); n_rlog33 = int(params[base+8])
    n_f33 = int(params[base+9])
    n_rho32 = int(params[base+10]); n_rlog32 = int(params[base+11])
    n_rho43 = int(params[base+12]); n_rlog43 = int(params[base+13])
    n_f43 = int(params[base+14])
    n_rho44 = int(params[base+15]); n_rlog44 = int(params[base+16])
    n_mode_re = int(params[base+17]); n_mode_im = int(params[base+18])

    # Unpack coefficient arrays
    offset = base + 19
    kep_coeffs = params[offset:offset+n_kep]; offset += n_kep
    rr_coeffs = params[offset:offset+n_rr]; offset += n_rr
    rho_22 = params[offset:offset+n_rho22]; offset += n_rho22
    rho_log_22 = params[offset:offset+n_rlog22]; offset += n_rlog22
    rho_21 = params[offset:offset+n_rho21]; offset += n_rho21
    rho_log_21 = params[offset:offset+n_rlog21]; offset += n_rlog21
    f_21 = params[offset:offset+n_f21]; offset += n_f21
    rho_33 = params[offset:offset+n_rho33]; offset += n_rho33
    rho_log_33 = params[offset:offset+n_rlog33]; offset += n_rlog33
    f_33 = params[offset:offset+n_f33]; offset += n_f33
    rho_32 = params[offset:offset+n_rho32]; offset += n_rho32
    rho_log_32 = params[offset:offset+n_rlog32]; offset += n_rlog32
    rho_43 = params[offset:offset+n_rho43]; offset += n_rho43
    rho_log_43 = params[offset:offset+n_rlog43]; offset += n_rlog43
    f_43 = params[offset:offset+n_f43]; offset += n_f43
    rho_44 = params[offset:offset+n_rho44]; offset += n_rho44
    rho_log_44 = params[offset:offset+n_rlog44]; offset += n_rlog44
    mode_re = params[offset:offset+n_mode_re]; offset += n_mode_re
    mode_im = params[offset:offset+n_mode_im]; offset += n_mode_im

    # Reconstruct complex mode coefficients
    n_mc = len(mode_re)
    mode_coeffs = np.empty(n_mc, dtype=np.complex128)
    for i in range(n_mc):
        mode_coeffs[i] = complex(mode_re[i], mode_im[i])

    # 1. Hamiltonian
    dHdr, dHdphi, dHdpr, omega, H_val, xi = ham_and_derivs(
        r, phi, pr, pphi, chi_1, chi_2, m_1, m_2, nu, a6v, dsov)

    # 2. Evolution equations
    edot, zdot, xavg = compute_edot_zdot_xavg(e, z, omega, kep_coeffs)

    # 3. Eccentric mode corrections (returns all modes, we use first 12)
    ecc_corr = compute_ecc_mode_corrections(e, z, xavg, mode_coeffs)
    h22_re = ecc_corr[0]; h22_im = ecc_corr[1]
    h21_re = ecc_corr[2]; h21_im = ecc_corr[3]
    h33_re = ecc_corr[4]; h33_im = ecc_corr[5]
    h32_re = ecc_corr[6]; h32_im = ecc_corr[7]
    h43_re = ecc_corr[8]; h43_im = ecc_corr[9]
    h44_re = ecc_corr[10]; h44_im = ecc_corr[11]

    # 4. Flux (6-mode)
    H_times_nu = nu * H_val
    flux = compute_flux_6mode(
        omega, e, z, xavg, H_times_nu, nu, pphi,
        prefix_22, prefix_21, prefix_33,
        prefix_32, prefix_43, prefix_44,
        rho_22, rho_log_22, rho_21, rho_log_21, f_21,
        rho_33, rho_log_33, f_33, f_vh_33_6,
        rho_32, rho_log_32,
        rho_43, rho_log_43, f_43,
        rho_44, rho_log_44,
        h22_re, h22_im, h21_re, h21_im, h33_re, h33_im,
        h32_re, h32_im, h43_re, h43_im, h44_re, h44_im)

    # 5. RR force corrections
    Fr_corr, Fphi_corr = compute_rr_force_corrections(e, z, xavg, rr_coeffs)

    # 6. Assemble
    flux_norm = flux / nu
    f_over_om = flux_norm / omega
    Fr = -pr / pphi * f_over_om * Fr_corr
    Fphi = -f_over_om * Fphi_corr

    out = np.empty(6)
    out[0] = xi * dHdpr
    out[1] = omega
    out[2] = -dHdr * xi + Fr
    out[3] = Fphi
    out[4] = edot
    out[5] = zdot
    return out


def setup_and_integrate_6mode(q, chi_1, chi_2, r0, pr0, pphi0, e0, zeta0,
                               t_end=1e7, rtol=1e-8, atol=1e-9, r_stop=2.5,
                               max_steps=100000):
    """6-mode variant of setup_and_integrate — same interface."""
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

    # Full statics (54 items) — extract only the 6-mode subset
    statics = precompute_waveform_statics(m_1, m_2, nu, delta, ap, chiS, chiA, True)
    # 16-mode order: prefix_22[0], prefix_21[1], prefix_31[2], prefix_33[3],
    # prefix_32[4], prefix_41[5], prefix_42[6], prefix_43[7], prefix_44[8],
    # prefix_52..88[9..15], then coefficient arrays[16..]
    prefix_22 = statics[0]
    prefix_21 = statics[1]
    prefix_33 = statics[3]
    prefix_32 = statics[4]
    prefix_43 = statics[7]
    prefix_44 = statics[8]
    # Coefficient arrays for the 6 modes
    rho_22     = statics[16]; rho_log_22 = statics[17]
    rho_21     = statics[18]; rho_log_21 = statics[19]; f_21 = statics[20]
    rho_33     = statics[24]; rho_log_33 = statics[25]; f_33 = statics[26]
    f_vh_33_6  = statics[27]
    rho_32     = statics[28]; rho_log_32 = statics[29]
    rho_43     = statics[35]; rho_log_43 = statics[36]; f_43 = statics[37]
    rho_44     = statics[38]; rho_log_44 = statics[39]

    # Pack into params array (smaller than 16-mode)
    sizes = np.array([
        len(kep_coeffs), len(rr_coeffs),
        len(rho_22), len(rho_log_22),
        len(rho_21), len(rho_log_21), len(f_21),
        len(rho_33), len(rho_log_33), len(f_33),
        len(rho_32), len(rho_log_32),
        len(rho_43), len(rho_log_43), len(f_43),
        len(rho_44), len(rho_log_44),
        len(mode_re), len(mode_im),
    ], dtype=np.float64)

    scalars = np.array([chi_1, chi_2, m_1, m_2, nu, a6v, dsov,
                        prefix_22, prefix_21, prefix_33,
                        prefix_32, prefix_43, prefix_44,
                        f_vh_33_6])

    params = np.concatenate([scalars, sizes,
                             kep_coeffs, rr_coeffs,
                             rho_22, rho_log_22, rho_21, rho_log_21, f_21,
                             rho_33, rho_log_33, f_33,
                             rho_32, rho_log_32,
                             rho_43, rho_log_43, f_43,
                             rho_44, rho_log_44,
                             mode_re, mode_im])

    y0 = np.array([r0, 0.0, pr0, pphi0, e0, zeta0])

    t_arr, y_arr = integrate_adaptive(rhs_ecc_6mode, 0.0, y0, t_end, params,
                                      rtol=rtol, atol=atol,
                                      max_steps=max_steps, r_stop=r_stop)

    # Compute x post-hoc
    n = len(t_arr)
    x_arr = np.empty(n)
    omega_arr = np.empty(n)
    for i in range(n):
        _, _, _, omi, _, _ = ham_and_derivs(
            y_arr[i, 0], y_arr[i, 1], y_arr[i, 2], y_arr[i, 3],
            chi_1, chi_2, m_1, m_2, nu, a6v, dsov)
        omega_arr[i] = omi
        _, _, xi = compute_edot_zdot_xavg(y_arr[i, 4], y_arr[i, 5], omi, kep_coeffs)
        x_arr[i] = xi

    return {
        't': t_arr, 'r': y_arr[:, 0], 'phi': y_arr[:, 1],
        'pr': y_arr[:, 2], 'pphi': y_arr[:, 3],
        'e': y_arr[:, 4], 'zeta': y_arr[:, 5],
        'x': x_arr, 'omega': omega_arr,
    }
