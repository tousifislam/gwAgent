"""
Compare our Numba evolution equations against pySEOBNR's Cython implementation.
Generates Nature-quality comparison plots.

Usage:
    conda activate kitp-py310
    python compare_evolution_eqs.py
"""
import sys, os, json, time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Our code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evolution_equations import initialize_keplerian_coeffs, compute_edot_zdot_xavg

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

# Nature-quality rcParams
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


def get_pyseobnr_evol_eqs(q, chi1, chi2):
    """Create a pySEOBNR evolution equations instance."""
    from pyseobnr.eob.dynamics.Keplerian_evolution_equations_flags._implementation import edot_zdot_xavg_flags

    m_1 = q / (1.0 + q)
    m_2 = 1.0 / (1.0 + q)
    nu = m_1 * m_2
    delta = m_1 - m_2
    chiA = (chi1 - chi2) / 2.0
    chiS = (chi1 + chi2) / 2.0

    evol = edot_zdot_xavg_flags()
    evol.initialize(
        chiA=chiA, chiS=chiS, delta=delta,
        flagPN1=1, flagPN2=1, flagPN3=1, flagPN32=1, flagPN52=1,
        nu=nu,
    )
    return evol, nu, delta, chiA, chiS


def compare_single_config(q, chi1, chi2, label):
    """Compare for one binary configuration across a grid of (e, z, omega)."""
    evol_ref, nu, delta, chiA, chiS = get_pyseobnr_evol_eqs(q, chi1, chi2)
    coeffs = initialize_keplerian_coeffs(nu, delta, chiA, chiS)

    es = np.linspace(0.01, 0.6, 20)
    zs = np.linspace(0.0, 2 * np.pi, 12)
    omegas = np.array([0.01, 0.02, 0.03, 0.05, 0.07])

    edot_errs, zdot_errs, xavg_errs = [], [], []
    e_vals, omega_vals = [], []

    for e in es:
        for z in zs:
            for omega in omegas:
                # Our Numba
                edot_n, zdot_n, xavg_n = compute_edot_zdot_xavg(e, z, omega, coeffs)

                # pySEOBNR reference
                evol_ref.compute(e, omega, z)
                edot_r = evol_ref.get("edot")
                zdot_r = evol_ref.get("zdot")
                xavg_r = evol_ref.get("xavg_omegainst")

                if abs(edot_r) > 1e-30:
                    edot_errs.append(abs(edot_n - edot_r) / abs(edot_r))
                if abs(zdot_r) > 1e-30:
                    zdot_errs.append(abs(zdot_n - zdot_r) / abs(zdot_r))
                if abs(xavg_r) > 1e-30:
                    xavg_errs.append(abs(xavg_n - xavg_r) / abs(xavg_r))
                e_vals.append(e)
                omega_vals.append(omega)

    return {
        'label': label,
        'edot_errs': np.array(edot_errs),
        'zdot_errs': np.array(zdot_errs),
        'xavg_errs': np.array(xavg_errs),
        'e_vals': np.array(e_vals),
        'omega_vals': np.array(omega_vals),
    }


def plot_results(all_results):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    for res in all_results:
        lbl = res['label']
        if len(res['edot_errs']) > 0:
            axes[0].hist(np.log10(res['edot_errs'] + 1e-20), bins=30, alpha=0.5, label=lbl)
        if len(res['zdot_errs']) > 0:
            axes[1].hist(np.log10(res['zdot_errs'] + 1e-20), bins=30, alpha=0.5, label=lbl)
        if len(res['xavg_errs']) > 0:
            axes[2].hist(np.log10(res['xavg_errs'] + 1e-20), bins=30, alpha=0.5, label=lbl)

    axes[0].set_xlabel(r'$\log_{10}(|\Delta\dot{e}/\dot{e}|)$')
    axes[0].set_ylabel('Count')
    axes[0].set_title(r'(a) $\dot{e}$ relative error', loc='left', fontsize=9, fontweight='bold')
    axes[0].legend(fontsize=6)

    axes[1].set_xlabel(r'$\log_{10}(|\Delta\dot{\zeta}/\dot{\zeta}|)$')
    axes[1].set_title(r'(b) $\dot{\zeta}$ relative error', loc='left', fontsize=9, fontweight='bold')

    axes[2].set_xlabel(r'$\log_{10}(|\Delta x_{\rm avg}/x_{\rm avg}|)$')
    axes[2].set_title(r'(c) $x_{\rm avg}$ relative error', loc='left', fontsize=9, fontweight='bold')

    plt.tight_layout(w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'compare_evolution_eqs.{ext}'))
    plt.close(fig)
    print("  Saved compare_evolution_eqs.pdf/png")


if __name__ == '__main__':
    print("=" * 60)
    print("STEP 2: Evolution equations comparison (Numba vs pySEOBNR)")
    print("=" * 60)

    # Warm up
    coeffs = initialize_keplerian_coeffs(0.25, 0.0, 0.0, 0.0)
    _ = compute_edot_zdot_xavg(0.1, 0.5, 0.02, coeffs)

    cases = [
        (1.0, 0.0, 0.0, r"$q{=}1,\chi{=}0$"),
        (3.0, 0.5, 0.3, r"$q{=}3,\chi{=}(0.5,0.3)$"),
        (6.0, 0.9, 0.0, r"$q{=}6,\chi{=}(0.9,0)$"),
        (10.0, 0.7, 0.7, r"$q{=}10,\chi{=}(0.7,0.7)$"),
    ]

    all_results = []
    for q, chi1, chi2, label in cases:
        print(f"\n  {label}...")
        res = compare_single_config(q, chi1, chi2, label)
        print(f"    edot: median={np.median(res['edot_errs']):.2e}, max={np.max(res['edot_errs']):.2e}")
        print(f"    zdot: median={np.median(res['zdot_errs']):.2e}, max={np.max(res['zdot_errs']):.2e}")
        print(f"    xavg: median={np.median(res['xavg_errs']):.2e}, max={np.max(res['xavg_errs']):.2e}")
        all_results.append(res)

    # Timing
    print("\n--- Timing ---")
    coeffs = initialize_keplerian_coeffs(0.25, 0.6, 0.1, 0.2)
    t0 = time.perf_counter()
    for _ in range(10000):
        compute_edot_zdot_xavg(0.2, 1.5, 0.03, coeffs)
    t_us = (time.perf_counter() - t0) / 10000 * 1e6
    print(f"  compute_edot_zdot_xavg: {t_us:.2f} us/call")

    print("\nGenerating plots...")
    plot_results(all_results)

    # Save data
    save_data = {
        'timing_us': t_us,
        'cases': [],
    }
    for res in all_results:
        save_data['cases'].append({
            'label': res['label'],
            'edot_median': float(np.median(res['edot_errs'])),
            'edot_max': float(np.max(res['edot_errs'])),
            'zdot_median': float(np.median(res['zdot_errs'])),
            'zdot_max': float(np.max(res['zdot_errs'])),
            'xavg_median': float(np.median(res['xavg_errs'])),
            'xavg_max': float(np.max(res['xavg_errs'])),
        })
    with open(os.path.join(RESULTS, 'compare_evolution_eqs.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print("  Saved compare_evolution_eqs.json")
    print("\nStep 2 complete.")
