"""
Compare full Numba dynamics against pySEOBNR.
Uses pySEOBNR for initial conditions, then integrates with our Numba code.
Generates Nature-quality multi-panel comparison plots.

Usage:
    conda activate kitp-py310
    python compare_dynamics.py
"""
import sys, os, json, time, warnings, pickle
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dynamics import setup_and_integrate

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
COL_TARG = '#2ca02c'


def get_pyseobnr_dynamics(q, chi1, chi2, e0, omega0=0.009):
    """Run full pySEOBNR and extract dynamics + ICs."""
    from pyseobnr.generate_waveform import generate_modes_opt
    t0 = time.perf_counter()
    _, _, model = generate_modes_opt(
        q, chi1, chi2, omega0, eccentricity=e0, rel_anomaly=0.0,
        approximant="SEOBNRv5EHM", debug=True,
        settings={'use_wave_convention': True})
    elapsed = (time.perf_counter() - t0) * 1000

    dyn = model.dynamics
    # Extract ICs from the first row of dynamics
    r0 = dyn[0, 1]
    pr0 = dyn[0, 3]
    pphi0 = dyn[0, 4]

    return {
        't': dyn[:, 0], 'r': dyn[:, 1], 'phi': dyn[:, 2],
        'pr': dyn[:, 3], 'pphi': dyn[:, 4],
        'e': dyn[:, 5], 'zeta': dyn[:, 6], 'x': dyn[:, 7],
        'r0': r0, 'pr0': pr0, 'pphi0': pphi0,
        'elapsed_ms': elapsed,
    }


def compare_case(q, chi1, chi2, e0, label, rtol=1e-8):
    """Compare one case."""
    print(f"\n  {label}...", flush=True)

    # pySEOBNR reference
    try:
        ref = get_pyseobnr_dynamics(q, chi1, chi2, e0)
    except Exception as ex:
        print(f"    pySEOBNR FAILED: {ex}")
        return None

    print(f"    pySEOBNR: {ref['elapsed_ms']:.0f}ms, t_end={ref['t'][-1]:.0f}M, "
          f"N={len(ref['t'])}", flush=True)

    # Our dynamics (using pySEOBNR ICs)
    t0 = time.perf_counter()
    try:
        ours = setup_and_integrate(
            q, chi1, chi2, ref['r0'], ref['pr0'], ref['pphi0'], e0, 0.0,
            rtol=rtol, r_stop=2.5)
    except Exception as ex:
        print(f"    OUR DYNAMICS FAILED: {ex}")
        return None
    our_elapsed = (time.perf_counter() - t0) * 1000

    print(f"    Ours: {our_elapsed:.1f}ms, t_end={ours['t'][-1]:.0f}M, "
          f"N={len(ours['t'])}", flush=True)

    # Compare in overlapping time range
    t_max_common = min(ref['t'][-1], ours['t'][-1])
    mask = ref['t'] <= t_max_common
    t_common = ref['t'][mask]

    if len(t_common) < 10:
        print("    WARNING: too few overlapping points")
        return None

    # Interpolate ours onto ref time grid
    e_interp = np.interp(t_common, ours['t'], ours['e'])
    x_interp = np.interp(t_common, ours['t'], ours['x'])

    de = e_interp - ref['e'][mask]
    dx = x_interp - ref['x'][mask]

    max_de = np.max(np.abs(de))
    max_dx = np.max(np.abs(dx))

    print(f"    max|Δe| = {max_de:.4e}, max|Δx| = {max_dx:.4e}")
    print(f"    Δt_end = {ours['t'][-1] - ref['t'][-1]:.1f} M")

    return {
        'label': label, 'q': q, 'chi1': chi1, 'chi2': chi2, 'e0': e0,
        'ref': ref, 'ours': ours,
        'max_de': max_de, 'max_dx': max_dx,
        't_end_diff': ours['t'][-1] - ref['t'][-1],
        'our_elapsed_ms': our_elapsed, 'ref_elapsed_ms': ref['elapsed_ms'],
    }


def plot_validation(results):
    """Multi-panel validation plot: e(t) and x(t) for each case."""
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 2, figsize=(6.5, 2.2 * n_cases), squeeze=False)

    for ci, res in enumerate(results):
        ref, ours = res['ref'], res['ours']

        # Align at merger for display
        t_ref = ref['t'] - ref['t'][-1]
        t_ours = ours['t'] - ours['t'][-1]

        ax = axes[ci, 0]
        ax.plot(t_ref / 1e3, ref['e'], COL_REF, lw=1.0, label='pySEOBNR')
        ax.plot(t_ours / 1e3, ours['e'], COL_OURS, lw=0.8, ls='--', label='This work')
        ax.set_ylabel(r'$e(t)$')
        ax.text(0.97, 0.92, res['label'], transform=ax.transAxes,
                fontsize=7, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='0.7',
                          boxstyle='round,pad=0.25', alpha=0.9))
        if ci == 0:
            ax.legend(loc='lower left', fontsize=7)

        ax = axes[ci, 1]
        ax.plot(t_ref / 1e3, ref['x'], COL_REF, lw=1.0)
        ax.plot(t_ours / 1e3, ours['x'], COL_OURS, lw=0.8, ls='--')
        ax.set_ylabel(r'$x(t)$')

    for j in range(2):
        axes[-1, j].set_xlabel(r'$t\;[10^3\,M]$')

    plt.tight_layout(h_pad=0.3, w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'compare_dynamics.{ext}'))
    plt.close(fig)
    print(f"\n  Saved compare_dynamics.pdf/png")


CASES = [
    (1,   0,   0,   0.3,  r"$q{=}1,\;\chi{=}0,\;e_0{=}0.3$"),
    (3,   0.5, 0.3, 0.2,  r"$q{=}3,\;\chi{=}(0.5,0.3),\;e_0{=}0.2$"),
    (6,   0.9, 0,   0.1,  r"$q{=}6,\;\chi{=}(0.9,0),\;e_0{=}0.1$"),
    (10,  0.7, 0.7, 0.4,  r"$q{=}10,\;\chi{=}(0.7,0.7),\;e_0{=}0.4$"),
]


if __name__ == '__main__':
    print("=" * 60)
    print("FULL DYNAMICS: Numba vs pySEOBNR")
    print("=" * 60)

    # Warmup JIT (one small integration)
    print("Warming up Numba JIT (first call compiles)...", flush=True)
    try:
        _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                                t_end=100, rtol=1e-4, max_steps=50)
        print("  Warmup complete.", flush=True)
    except Exception as ex:
        print(f"  Warmup failed: {ex}", flush=True)

    results = []
    for q, chi1, chi2, e0, label in CASES:
        res = compare_case(q, chi1, chi2, e0, label)
        if res is not None:
            results.append(res)

    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Case':<45s} {'max|Δe|':>10s} {'max|Δx|':>10s} {'Δt_end':>8s} {'ms_ours':>8s} {'ms_ref':>8s}")
        for r in results:
            print(f"{r['label']:<45s} {r['max_de']:10.4e} {r['max_dx']:10.4e} "
                  f"{r['t_end_diff']:8.0f} {r['our_elapsed_ms']:8.1f} {r['ref_elapsed_ms']:8.0f}")

        plot_validation(results)

        # Save data
        save_data = []
        for r in results:
            save_data.append({
                'label': r['label'], 'q': r['q'], 'chi1': r['chi1'],
                'chi2': r['chi2'], 'e0': r['e0'],
                'max_de': r['max_de'], 'max_dx': r['max_dx'],
                't_end_diff': r['t_end_diff'],
                'our_elapsed_ms': r['our_elapsed_ms'],
                'ref_elapsed_ms': r['ref_elapsed_ms'],
            })
        with open(os.path.join(RESULTS, 'compare_dynamics.json'), 'w') as f:
            json.dump(save_data, f, indent=2)

        print("  Saved compare_dynamics.json")

    print("\nDone.")
