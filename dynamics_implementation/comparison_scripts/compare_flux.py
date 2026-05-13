"""
Compare our 3-mode Numba flux against pySEOBNR's full 35-mode flux.
Generates Nature-quality comparison plots.

Usage:
    conda activate kitp-py310
    python compare_flux.py
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Our code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hamiltonian import evaluate_H, ham_and_derivs
from fits import a6_NS, dSO, GSF_amplitude_fits_numba
from flux import precompute_waveform_statics, compute_flux_3mode, compute_rr_force
from ecc_corrections import (compute_ecc_mode_corrections_default,
                              initialize_rr_force_coeffs, compute_rr_force_corrections)

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'legend.frameon': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'lines.linewidth': 1.0,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})
COL_REF  = '#1a1a1a'
COL_OURS = '#d62728'
COL_BAR  = '#4c72b0'


def setup_pyseobnr(q, chi1, chi2, omega_start=0.015, e0=0.1):
    """Setup pySEOBNR objects for comparison."""
    from pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
        Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt)
    from pyseobnr.eob.waveform.waveform import compute_newtonian_prefixes
    from pyseobnr.eob.waveform.waveform_ecc import SEOBNRv5RRForceEcc
    from pyseobnr.eob.utils.containers import EOBParams
    from pyseobnr.eob.fits.fits_Hamiltonian import a6_NS as a6ref, dSO as dSOref
    from pyseobnr.eob.fits.GSF_fits import GSF_amplitude_fits
    from pyseobnr.eob.dynamics.initial_conditions_aligned_ecc_opt import compute_IC_ecc_opt
    import re

    m_1 = q / (1.0 + q)
    m_2 = 1.0 / (1.0 + q)
    nu = m_1 * m_2
    ap = chi1 * m_1 + chi2 * m_2
    am = chi1 * m_1 - chi2 * m_2

    flags_ecc = dict(flagPN12=1, flagPN1=1, flagPN32=1, flagPN2=1,
                     flagPN52=1, flagPN3=1, flagPA=1, flagPA_modes=1,
                     flagTail=1, flagMemory=1)

    phys_pars = dict(
        m_1=m_1, m_2=m_2, chi_1=chi1, chi_2=chi2,
        a1=abs(chi1), a2=abs(chi2),
        chi1_v=np.array([0., 0., chi1]), chi2_v=np.array([0., 0., chi2]),
        H_val=0.0, lN=np.array([0., 0., 1.]),
        omega=omega_start, omega_circ=omega_start,
        dissipative_ICs="root", eccentricity=e0, EccIC=1,
        rel_anomaly=0.0, flags_ecc=flags_ecc,
        IC_messages=False, r_min=0.0, t_max=1e9)

    eob_pars = EOBParams(phys_pars, {},
                         mode_array=[(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)],
                         special_modes=[(2,1),(4,3)], ecc_model=True)

    prefixes = compute_newtonian_prefixes(m_1, m_2)
    eob_pars.flux_params.prefixes = np.array(prefixes)
    eob_pars.flux_params.prefixes_abs = np.abs(np.array(prefixes))
    eob_pars.flux_params.extra_PN_terms = True

    H = Ham_aligned_opt(eob_pars)
    H.calibration_coeffs.a6 = a6ref(nu)
    H.calibration_coeffs.dSO = dSOref(nu, ap, am)

    gsf_coeffs = GSF_amplitude_fits(nu)
    for key in gsf_coeffs:
        tmp = re.findall(r"h(\d)(\d)_v(\d+)", key)
        if tmp:
            l, m, v = [int(x) for x in tmp[0]]
            eob_pars.flux_params.extra_coeffs[l, m, v] = gsf_coeffs[key]
        else:
            tmp = re.findall(r"h(\d)(\d)_vlog(\d+)", key)
            if tmp:
                l, m, v = [int(x) for x in tmp[0]]
                eob_pars.flux_params.extra_coeffs_log[l, m, v] = gsf_coeffs[key]

    RR = SEOBNRv5RRForceEcc("Ecc")
    RR.initialize(eob_pars)

    r0, pphi0, pr0 = compute_IC_ecc_opt(
        m_1=m_1, m_2=m_2, chi_1=chi1, chi_2=chi2,
        eccentricity=e0, rel_anomaly=0.0, H=H, RR=RR, params=eob_pars)

    return H, RR, eob_pars, m_1, m_2, r0, pr0, pphi0


def compare_rr_force(q, chi1, chi2, e0=0.1):
    """Compare RR force using full pySEOBNR dynamics for state points."""
    from pyseobnr.generate_waveform import generate_modes_opt
    from pyseobnr.eob.dynamics.rhs_aligned_ecc import get_rhs_ecc

    # Run full pySEOBNR to get model + dynamics
    _, _, model = generate_modes_opt(
        q, chi1, chi2, 0.009, eccentricity=e0, rel_anomaly=0.0,
        approximant="SEOBNRv5EHM", debug=True,
        settings={'use_wave_convention': True})

    dyn = model.dynamics
    H_obj = model.H
    RR_obj = model.RR
    eob_pars = model.eob_pars
    m_1 = eob_pars.p_params.m_1
    m_2 = eob_pars.p_params.m_2
    nu = m_1 * m_2
    delta = m_1 - m_2
    ap = chi1 * m_1 + chi2 * m_2
    am = chi1 * m_1 - chi2 * m_2
    chiS = (chi1 + chi2) / 2.0
    chiA = (chi1 - chi2) / 2.0

    # Pick a point in the middle of the dynamics
    idx = len(dyn) // 2
    r, phi, pr, pphi = dyn[idx, 1], dyn[idx, 2], dyn[idx, 3], dyn[idx, 4]
    e, z_val, x_val = dyn[idx, 5], dyn[idx, 6], dyn[idx, 7]

    # Get pySEOBNR RHS at this point
    state = np.array([r, phi, pr, pphi, e, z_val])
    rhs_ref = get_rhs_ecc(0.0, state, H_obj, RR_obj, chi1, chi2, m_1, m_2, eob_pars)
    # rhs_ref = (drdt, dphidt, dprdt, dpphidt, edot, zdot)
    # dpphidt = -dHdphi + Fphi = Fphi (since dHdphi=0 for aligned spins)
    # dprdt = -dHdr * xi + Fr
    Fphi_ref = rhs_ref[3]

    # Our computation
    a6v = a6_NS(nu)
    dsov = dSO(nu, ap, am)
    dHdr, dHdphi, dHdpr, omega, H_val, xi = ham_and_derivs(
        r, phi, pr, pphi, chi1, chi2, m_1, m_2, nu, a6v, dsov)

    statics = precompute_waveform_statics(m_1, m_2, nu, delta, ap, chiS, chiA, True)

    from ecc_mode_corrections import initialize_ecc_mode_coeffs, compute_ecc_mode_corrections
    mode_coeffs = initialize_ecc_mode_coeffs(nu)
    corr = compute_ecc_mode_corrections(e, z_val, x_val, mode_coeffs)

    H_times_nu = nu * H_val
    from flux import compute_flux
    flux_ours = compute_flux(
        omega, e, z_val, x_val, H_times_nu, nu, pphi,
        *statics, *corr)

    rr_coeffs = initialize_rr_force_coeffs(nu)
    Fr_corr, Fphi_corr = compute_rr_force_corrections(e, z_val, x_val, rr_coeffs)
    Fr_ours, Fphi_ours = compute_rr_force(pr, pphi, flux_ours, omega, Fphi_corr, Fr_corr, nu)

    Fr_ref = rhs_ref[2] + dHdr * xi  # dprdt + dHdr*xi = Fr

    return {
        'q': q, 'chi1': chi1, 'chi2': chi2, 'e0': e0,
        'Fr_ref': Fr_ref, 'Fphi_ref': Fphi_ref,
        'Fr_ours': Fr_ours, 'Fphi_ours': Fphi_ours,
        'Fr_rel': abs(Fr_ours - Fr_ref) / abs(Fr_ref) if abs(Fr_ref) > 1e-30 else 0,
        'Fphi_rel': abs(Fphi_ours - Fphi_ref) / abs(Fphi_ref) if abs(Fphi_ref) > 1e-30 else 0,
        'e': e, 'x': x_val, 'r': r,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("STEP 3: Flux & RR force comparison (3-mode vs pySEOBNR)")
    print("=" * 60)

    # Warm up Numba
    from flux import precompute_waveform_statics
    _ = precompute_waveform_statics(0.5, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, True)

    cases = [
        (1.0, 0.0, 0.0, 0.1),
        (3.0, 0.0, 0.0, 0.2),
        (3.0, 0.5, 0.3, 0.1),
        (6.0, 0.9, 0.0, 0.1),
        (10.0, 0.7, 0.7, 0.3),
    ]

    results = []
    for q, chi1, chi2, e0 in cases:
        print(f"\n  q={q}, chi=({chi1},{chi2}), e0={e0}...")
        try:
            r = compare_rr_force(q, chi1, chi2, e0)
            print(f"    Fr:   ours={r['Fr_ours']:.6e}, ref={r['Fr_ref']:.6e}, rel_err={r['Fr_rel']:.4e}")
            print(f"    Fphi: ours={r['Fphi_ours']:.6e}, ref={r['Fphi_ref']:.6e}, rel_err={r['Fphi_rel']:.4e}")
            results.append(r)
        except Exception as ex:
            print(f"    FAILED: {ex}")

    # Plot
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))

        labels = [f"q={r['q']}" for r in results]
        Fr_errs = [r['Fr_rel'] for r in results]
        Fphi_errs = [r['Fphi_rel'] for r in results]

        y = np.arange(len(results))
        axes[0].barh(y, np.log10(np.array(Fr_errs) + 1e-20), color=COL_BAR,
                     edgecolor='0.3', lw=0.4, height=0.6)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xlabel(r'$\log_{10}(|\Delta F_r / F_r|)$')
        axes[0].set_title(r'(a) $F_r$ relative error', loc='left', fontsize=9, fontweight='bold')
        axes[0].invert_yaxis()

        axes[1].barh(y, np.log10(np.array(Fphi_errs) + 1e-20), color=COL_OURS,
                     edgecolor='0.3', lw=0.4, height=0.6)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlabel(r'$\log_{10}(|\Delta F_\varphi / F_\varphi|)$')
        axes[1].set_title(r'(b) $F_\varphi$ relative error', loc='left', fontsize=9, fontweight='bold')
        axes[1].invert_yaxis()

        plt.tight_layout(w_pad=0.8)
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(RESULTS, f'compare_flux.{ext}'))
        plt.close(fig)
        print(f"\n  Saved compare_flux.pdf/png")

    # Save data
    save_data = [{'q': r['q'], 'chi1': r['chi1'], 'chi2': r['chi2'], 'e0': r['e0'],
                  'Fr_rel': r['Fr_rel'], 'Fphi_rel': r['Fphi_rel']} for r in results]
    with open(os.path.join(RESULTS, 'compare_flux.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved compare_flux.json")
    print("\nStep 3 complete.")
