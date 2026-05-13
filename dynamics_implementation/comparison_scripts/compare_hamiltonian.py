"""
Compare our Numba Hamiltonian against pySEOBNR's Cython implementation.
Generates Nature-quality comparison plots.

Usage:
    conda activate kitp-py310
    python compare_hamiltonian.py
"""
import sys, os, json, time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Our code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hamiltonian import evaluate_H, ham_and_derivs
from fits import a6_NS, dSO

# pySEOBNR reference
from pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
    evaluate_H as evaluate_H_cython,
)
from pyseobnr.eob.fits.fits_Hamiltonian import a6_NS as a6_NS_ref, dSO as dSO_ref

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

# ---------- Nature-quality rcParams ----------
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
    'xtick.minor.width': 0.3, 'ytick.minor.width': 0.3,
    'lines.linewidth': 1.0,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})
COL_REF  = '#1a1a1a'
COL_OURS = '#d62728'
COL_BAR  = '#4c72b0'


def generate_test_points(n=300, seed=42):
    """Generate random test points in physically reasonable ranges."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(n):
        q = rng.uniform(1.0, 10.0)
        m_1 = q / (1.0 + q)
        m_2 = 1.0 / (1.0 + q)
        nu = m_1 * m_2
        chi_1 = rng.uniform(-0.99, 0.99)
        chi_2 = rng.uniform(-0.99, 0.99)
        r = rng.uniform(4.0, 50.0)
        prst = rng.uniform(-0.3, 0.3)
        pphi = rng.uniform(2.0, 6.0)
        points.append((r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, q))
    return points


def compare_evaluate_H(points):
    """Compare evaluate_H at all test points."""
    results = {'H_rel_err': [], 'xi_rel_err': [], 'r': [], 'nu': []}

    for r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, q in points:
        ap = chi_1 * m_1 + chi_2 * m_2
        am = chi_1 * m_1 - chi_2 * m_2
        a6 = a6_NS(nu)
        dso = dSO(nu, ap, am)

        # Our Numba
        H_ours, xi_ours = evaluate_H(r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, a6, dso)

        # pySEOBNR reference
        a6_ref = a6_NS_ref(nu)
        dso_ref = dSO_ref(nu, ap, am)
        H_ref, xi_ref = evaluate_H_cython(
            (r, 0.0), (prst, pphi), chi_1, chi_2, m_1, m_2,
            1.0, nu, m_1, m_2, a6_ref, dso_ref
        )

        results['H_rel_err'].append(abs(H_ours - H_ref) / abs(H_ref))
        results['xi_rel_err'].append(abs(xi_ours - xi_ref) / abs(xi_ref))
        results['r'].append(r)
        results['nu'].append(nu)

    for k in results:
        results[k] = np.array(results[k])
    return results


def compare_gradients(points):
    """Compare H gradients (FD vs pySEOBNR analytical)."""
    from pyseobnr.eob.utils.containers import EOBParams
    from pyseobnr.eob.waveform.waveform import compute_newtonian_prefixes

    results = {'dHdr_rel': [], 'dHdpr_rel': [], 'omega_rel': [], 'r': []}

    for r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, q in points:
        ap = chi_1 * m_1 + chi_2 * m_2
        am = chi_1 * m_1 - chi_2 * m_2
        a6 = a6_NS(nu)
        dso = dSO(nu, ap, am)

        # Our FD gradients
        dHdr, dHdphi, dHdpr, omega, H_val, xi = ham_and_derivs(
            r, 0.0, prst, pphi, chi_1, chi_2, m_1, m_2, nu, a6, dso
        )

        # pySEOBNR analytical gradients
        a6_ref = a6_NS_ref(nu)
        dso_ref = dSO_ref(nu, ap, am)

        phys_pars = dict(
            m_1=m_1, m_2=m_2, chi_1=chi_1, chi_2=chi_2,
            a1=abs(chi_1), a2=abs(chi_2),
            chi1_v=np.array([0., 0., chi_1]),
            chi2_v=np.array([0., 0., chi_2]),
            H_val=0.0, lN=np.array([0., 0., 1.]),
            omega=0.01, omega_circ=0.01,
            dissipative_ICs="root", eccentricity=0.1, EccIC=1,
            rel_anomaly=0.0, flags_ecc=dict(
                flagPN12=1, flagPN1=1, flagPN32=1, flagPN2=1,
                flagPN52=1, flagPN3=1, flagPA=1, flagPA_modes=1,
                flagTail=1, flagMemory=1),
            IC_messages=False, r_min=0.0, t_max=1e9,
        )
        eob_pars = EOBParams(phys_pars, {},
                             mode_array=[(2,2)], special_modes=[],
                             ecc_model=True)
        H_obj = Ham_aligned_opt(eob_pars)
        H_obj.calibration_coeffs.a6 = a6_ref
        H_obj.calibration_coeffs.dSO = dso_ref

        try:
            dyn = H_obj.dynamics((r, 0.0), (prst, pphi), chi_1, chi_2, m_1, m_2)
            dHdr_ref, dHdphi_ref, dHdpr_ref, omega_ref = dyn[0], dyn[1], dyn[2], dyn[3]

            if abs(dHdr_ref) > 1e-15:
                results['dHdr_rel'].append(abs(dHdr - dHdr_ref) / abs(dHdr_ref))
            if abs(dHdpr_ref) > 1e-15:
                results['dHdpr_rel'].append(abs(dHdpr - dHdpr_ref) / abs(dHdpr_ref))
            if abs(omega_ref) > 1e-15:
                results['omega_rel'].append(abs(omega - omega_ref) / abs(omega_ref))
            results['r'].append(r)
        except Exception:
            pass

    for k in results:
        results[k] = np.array(results[k])
    return results


def plot_results(h_results, g_results):
    """Generate Nature-quality comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))

    # Row 1: evaluate_H comparison
    ax = axes[0, 0]
    ax.hist(np.log10(h_results['H_rel_err'] + 1e-20), bins=30, color=COL_BAR,
            edgecolor='0.3', lw=0.4, alpha=0.85)
    ax.axvline(np.log10(np.median(h_results['H_rel_err'])), color=COL_OURS,
               ls='--', lw=1.2, label=f"median={np.median(h_results['H_rel_err']):.1e}")
    ax.set_xlabel(r'$\log_{10}(|\Delta H/H|)$')
    ax.set_ylabel('Count')
    ax.set_title(r'(a) $H$ relative error', loc='left', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.hist(np.log10(h_results['xi_rel_err'] + 1e-20), bins=30, color=COL_BAR,
            edgecolor='0.3', lw=0.4, alpha=0.85)
    ax.axvline(np.log10(np.median(h_results['xi_rel_err'])), color=COL_OURS,
               ls='--', lw=1.2, label=f"median={np.median(h_results['xi_rel_err']):.1e}")
    ax.set_xlabel(r'$\log_{10}(|\Delta\xi/\xi|)$')
    ax.set_title(r'(b) $\xi$ relative error', loc='left', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.scatter(h_results['r'], np.log10(h_results['H_rel_err'] + 1e-20),
               s=4, c=h_results['nu'], cmap='viridis', alpha=0.7, rasterized=True)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\log_{10}(|\Delta H/H|)$')
    ax.set_title(r'(c) $H$ error vs $r$', loc='left', fontsize=9, fontweight='bold')

    # Row 2: gradient comparison
    if len(g_results['dHdr_rel']) > 0:
        ax = axes[1, 0]
        ax.hist(np.log10(g_results['dHdr_rel'] + 1e-20), bins=30, color=COL_BAR,
                edgecolor='0.3', lw=0.4, alpha=0.85)
        med = np.median(g_results['dHdr_rel'])
        ax.axvline(np.log10(med), color=COL_OURS, ls='--', lw=1.2,
                   label=f"median={med:.1e}")
        ax.set_xlabel(r'$\log_{10}(|\Delta \partial_r H|/|\partial_r H|)$')
        ax.set_ylabel('Count')
        ax.set_title(r'(d) $\partial_r H$ (FD vs analytical)', loc='left', fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)

    if len(g_results['dHdpr_rel']) > 0:
        ax = axes[1, 1]
        ax.hist(np.log10(g_results['dHdpr_rel'] + 1e-20), bins=30, color=COL_BAR,
                edgecolor='0.3', lw=0.4, alpha=0.85)
        med = np.median(g_results['dHdpr_rel'])
        ax.axvline(np.log10(med), color=COL_OURS, ls='--', lw=1.2,
                   label=f"median={med:.1e}")
        ax.set_xlabel(r'$\log_{10}(|\Delta \partial_{p_r} H|/|\partial_{p_r} H|)$')
        ax.set_title(r'(e) $\partial_{p_r} H$ (FD vs analytical)', loc='left', fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)

    if len(g_results['omega_rel']) > 0:
        ax = axes[1, 2]
        ax.hist(np.log10(g_results['omega_rel'] + 1e-20), bins=30, color=COL_BAR,
                edgecolor='0.3', lw=0.4, alpha=0.85)
        med = np.median(g_results['omega_rel'])
        ax.axvline(np.log10(med), color=COL_OURS, ls='--', lw=1.2,
                   label=f"median={med:.1e}")
        ax.set_xlabel(r'$\log_{10}(|\Delta\omega/\omega|)$')
        ax.set_title(r'(f) $\omega$ (FD vs analytical)', loc='left', fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'compare_hamiltonian.{ext}'))
    plt.close(fig)
    print(f"  Saved compare_hamiltonian.pdf/png")


if __name__ == '__main__':
    print("=" * 60)
    print("STEP 1: Hamiltonian comparison (Numba vs pySEOBNR)")
    print("=" * 60)

    # Warm up Numba
    print("Warming up Numba JIT...")
    _ = evaluate_H(10.0, 0.0, 3.5, 0.0, 0.0, 0.5, 0.5, 0.25, 0.0, 0.0)
    _ = ham_and_derivs(10.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.5, 0.5, 0.25, 0.0, 0.0)

    points = generate_test_points(300)
    print(f"Testing {len(points)} random points...")

    # Compare evaluate_H
    print("\n--- evaluate_H ---")
    h_results = compare_evaluate_H(points)
    print(f"  H  relative error: median={np.median(h_results['H_rel_err']):.2e}, "
          f"max={np.max(h_results['H_rel_err']):.2e}")
    print(f"  xi relative error: median={np.median(h_results['xi_rel_err']):.2e}, "
          f"max={np.max(h_results['xi_rel_err']):.2e}")

    # Compare gradients
    print("\n--- Gradients (FD vs analytical) ---")
    g_results = compare_gradients(points[:100])
    if len(g_results['dHdr_rel']) > 0:
        print(f"  dH/dr  relative error: median={np.median(g_results['dHdr_rel']):.2e}, "
              f"max={np.max(g_results['dHdr_rel']):.2e}")
        print(f"  dH/dpr relative error: median={np.median(g_results['dHdpr_rel']):.2e}, "
              f"max={np.max(g_results['dHdpr_rel']):.2e}")
        print(f"  omega  relative error: median={np.median(g_results['omega_rel']):.2e}, "
              f"max={np.max(g_results['omega_rel']):.2e}")

    # Timing
    print("\n--- Timing ---")
    t0 = time.perf_counter()
    for r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, q in points[:100]:
        ap = chi_1 * m_1 + chi_2 * m_2
        am = chi_1 * m_1 - chi_2 * m_2
        evaluate_H(r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, a6_NS(nu), dSO(nu, ap, am))
    t_numba = (time.perf_counter() - t0) / 100 * 1e6
    print(f"  Numba evaluate_H: {t_numba:.2f} us/call")

    t0 = time.perf_counter()
    for r, prst, pphi, chi_1, chi_2, m_1, m_2, nu, q in points[:100]:
        ap = chi_1 * m_1 + chi_2 * m_2
        am = chi_1 * m_1 - chi_2 * m_2
        ham_and_derivs(r, 0.0, prst, pphi, chi_1, chi_2, m_1, m_2, nu, a6_NS(nu), dSO(nu, ap, am))
    t_grad = (time.perf_counter() - t0) / 100 * 1e6
    print(f"  Numba ham_and_derivs (H + 3 FD gradients): {t_grad:.2f} us/call")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(h_results, g_results)

    # Save data
    save_data = {
        'H_rel_err_median': float(np.median(h_results['H_rel_err'])),
        'H_rel_err_max': float(np.max(h_results['H_rel_err'])),
        'xi_rel_err_median': float(np.median(h_results['xi_rel_err'])),
        'xi_rel_err_max': float(np.max(h_results['xi_rel_err'])),
        'timing_evaluate_H_us': t_numba,
        'timing_ham_and_derivs_us': t_grad,
        'n_points': len(points),
    }
    if len(g_results['dHdr_rel']) > 0:
        save_data['dHdr_rel_median'] = float(np.median(g_results['dHdr_rel']))
        save_data['dHdpr_rel_median'] = float(np.median(g_results['dHdpr_rel']))
        save_data['omega_rel_median'] = float(np.median(g_results['omega_rel']))

    with open(os.path.join(RESULTS, 'compare_hamiltonian.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved compare_hamiltonian.json")
    print("\nStep 1 complete.")
