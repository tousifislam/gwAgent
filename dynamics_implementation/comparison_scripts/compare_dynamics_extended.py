"""
Extended dynamics comparison: 10+ systems with Δe(t) and Δx(t) residual panels.
Nature-quality multi-panel figure.

Usage:
    conda activate kitp-py310
    python compare_dynamics_extended.py
"""
import sys, os, json, time, warnings, pickle
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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
    'xtick.minor.width': 0.3, 'ytick.minor.width': 0.3,
    'lines.linewidth': 1.0,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})
COL_REF  = '#1a1a1a'
COL_OURS = '#d62728'
COL_DE   = '#4c72b0'
COL_DX   = '#e377c2'


CASES = [
    # q, chi1, chi2, e0, label
    # --- Low eccentricity ---
    (1,   0.5, 0.5, 0.01, r"$q{=}1,\;\chi{=}(0.5,0.5),\;e_0{=}0.01$"),
    (1,   0.0, 0.0, 0.1,  r"$q{=}1,\;\chi{=}0,\;e_0{=}0.1$"),
    # --- Moderate eccentricity ---
    (2,   0.3, 0.3, 0.15, r"$q{=}2,\;\chi{=}(0.3,0.3),\;e_0{=}0.15$"),
    (3,   0.0, 0.0, 0.2,  r"$q{=}3,\;\chi{=}0,\;e_0{=}0.2$"),
    (3,   0.5, 0.3, 0.2,  r"$q{=}3,\;\chi{=}(0.5,0.3),\;e_0{=}0.2$"),
    (2,   0.0, 0.8, 0.2,  r"$q{=}2,\;\chi{=}(0,0.8),\;e_0{=}0.2$"),
    # --- High eccentricity ---
    (1,   0.0, 0.0, 0.3,  r"$q{=}1,\;\chi{=}0,\;e_0{=}0.3$"),
    (5,   0.3, 0.1, 0.3,  r"$q{=}5,\;\chi{=}(0.3,0.1),\;e_0{=}0.3$"),
    (3,   0.0, 0.0, 0.4,  r"$q{=}3,\;\chi{=}0,\;e_0{=}0.4$"),
    (10,  0.7, 0.7, 0.4,  r"$q{=}10,\;\chi{=}(0.7,0.7),\;e_0{=}0.4$"),
    (1,   0.0, 0.0, 0.5,  r"$q{=}1,\;\chi{=}0,\;e_0{=}0.5$"),
    (2,   0.3, 0.0, 0.5,  r"$q{=}2,\;\chi{=}(0.3,0),\;e_0{=}0.5$"),
    (5,   0.0, 0.0, 0.5,  r"$q{=}5,\;\chi{=}0,\;e_0{=}0.5$"),
    (1,   0.0, 0.0, 0.6,  r"$q{=}1,\;\chi{=}0,\;e_0{=}0.6$"),
    (3,   0.3, 0.1, 0.6,  r"$q{=}3,\;\chi{=}(0.3,0.1),\;e_0{=}0.6$"),
]


def get_pyseobnr_dynamics(q, chi1, chi2, e0, omega0=0.009):
    from pyseobnr.generate_waveform import generate_modes_opt
    t0 = time.perf_counter()
    _, _, model = generate_modes_opt(
        q, chi1, chi2, omega0, eccentricity=e0, rel_anomaly=0.0,
        approximant="SEOBNRv5EHM", debug=True,
        settings={'use_wave_convention': True})
    elapsed = (time.perf_counter() - t0) * 1000
    dyn = model.dynamics
    return {
        't': dyn[:, 0], 'r': dyn[:, 1], 'phi': dyn[:, 2],
        'pr': dyn[:, 3], 'pphi': dyn[:, 4],
        'e': dyn[:, 5], 'zeta': dyn[:, 6], 'x': dyn[:, 7],
        'r0': dyn[0, 1], 'pr0': dyn[0, 3], 'pphi0': dyn[0, 4],
        'elapsed_ms': elapsed,
    }


def compare_case(q, chi1, chi2, e0, label, rtol=1e-8):
    print(f"\n  {label}...", flush=True)
    try:
        ref = get_pyseobnr_dynamics(q, chi1, chi2, e0)
    except Exception as ex:
        print(f"    pySEOBNR FAILED: {ex}")
        return None

    try:
        t0 = time.perf_counter()
        ours = setup_and_integrate(
            q, chi1, chi2, ref['r0'], ref['pr0'], ref['pphi0'], e0, 0.0, rtol=rtol)
        our_elapsed = (time.perf_counter() - t0) * 1000
    except Exception as ex:
        print(f"    OUR DYNAMICS FAILED: {ex}")
        return None

    # Interpolate onto common time grid (use pySEOBNR times)
    t_max_common = min(ref['t'][-1], ours['t'][-1])
    mask = ref['t'] <= t_max_common
    t_common = ref['t'][mask]
    if len(t_common) < 10:
        print("    Too few overlapping points")
        return None

    # Cubic spline interpolation of our solution
    if len(ours['t']) > 3:
        e_interp = CubicSpline(ours['t'], ours['e'])(t_common)
        x_interp = CubicSpline(ours['t'], ours['x'])(t_common)
    else:
        e_interp = np.interp(t_common, ours['t'], ours['e'])
        x_interp = np.interp(t_common, ours['t'], ours['x'])

    de = e_interp - ref['e'][mask]
    dx = x_interp - ref['x'][mask]

    max_de = np.max(np.abs(de))
    max_dx = np.max(np.abs(dx))
    max_de_rel = np.max(np.abs(de) / (np.abs(ref['e'][mask]) + 1e-10))
    max_dx_rel = np.max(np.abs(dx) / (np.abs(ref['x'][mask]) + 1e-10))

    print(f"    max|Δe|={max_de:.4e}  max|Δx|={max_dx:.4e}  "
          f"Δt_end={ours['t'][-1]-ref['t'][-1]:.1f}M  "
          f"ours={our_elapsed:.1f}ms  ref={ref['elapsed_ms']:.0f}ms", flush=True)

    return {
        'label': label, 'q': q, 'chi1': chi1, 'chi2': chi2, 'e0': e0,
        'ref': ref, 'ours': ours,
        't_common': t_common, 'de': de, 'dx': dx,
        'max_de': max_de, 'max_dx': max_dx,
        'max_de_rel': max_de_rel, 'max_dx_rel': max_dx_rel,
        't_end_diff': ours['t'][-1] - ref['t'][-1],
        'our_elapsed_ms': our_elapsed, 'ref_elapsed_ms': ref['elapsed_ms'],
    }


def plot_dynamics_and_residuals(results):
    """4-column plot: e(t), x(t), Δe(t), Δx(t) for each case."""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(7.5, 1.8 * n), squeeze=False)

    for ci, res in enumerate(results):
        ref, ours = res['ref'], res['ours']

        # Align at merger for display
        t_ref = (ref['t'] - ref['t'][-1]) / 1e3
        t_ours = (ours['t'] - ours['t'][-1]) / 1e3
        t_common_aligned = (res['t_common'] - ref['t'][-1]) / 1e3

        # e(t)
        ax = axes[ci, 0]
        ax.plot(t_ref, ref['e'], COL_REF, lw=0.9, label='pySEOBNR')
        ax.plot(t_ours, ours['e'], COL_OURS, lw=0.7, ls='--', label='This work')
        ax.set_ylabel(r'$e$')
        if ci == 0:
            ax.legend(loc='upper right', fontsize=6)

        # x(t)
        ax = axes[ci, 1]
        ax.plot(t_ref, ref['x'], COL_REF, lw=0.9)
        ax.plot(t_ours, ours['x'], COL_OURS, lw=0.7, ls='--')
        ax.set_ylabel(r'$x$')

        # Δe(t)
        ax = axes[ci, 2]
        ax.plot(t_common_aligned, res['de'], COL_DE, lw=0.7)
        ax.axhline(0, color='0.5', ls='-', lw=0.3)
        ax.set_ylabel(r'$\Delta e$')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        # Δx(t)
        ax = axes[ci, 3]
        ax.plot(t_common_aligned, res['dx'], COL_DX, lw=0.7)
        ax.axhline(0, color='0.5', ls='-', lw=0.3)
        ax.set_ylabel(r'$\Delta x$')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        # Label on first column
        axes[ci, 0].text(
            0.03, 0.92, res['label'], transform=axes[ci, 0].transAxes,
            fontsize=5.5, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='0.7',
                      boxstyle='round,pad=0.2', alpha=0.9))

    # Column titles
    for j, title in enumerate([r'$e(t)$', r'$x(t)$', r'$\Delta e(t)$', r'$\Delta x(t)$']):
        axes[0, j].set_title(title, loc='center', fontsize=9, fontweight='bold')

    for j in range(4):
        axes[-1, j].set_xlabel(r'$t\;[10^3\,M]$')

    plt.tight_layout(h_pad=0.15, w_pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'dynamics_residuals.{ext}'))
    plt.close(fig)
    print(f"\n  Saved dynamics_residuals.pdf/png")


def plot_error_summary(results):
    """Bar chart of max|Δe| and max|Δx| for all cases."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    n = len(results)
    labels = [r['label'] for r in results]
    y = np.arange(n)

    # max|Δe|
    ax = axes[0]
    vals = [r['max_de'] for r in results]
    ax.barh(y, vals, height=0.6, color=COL_DE, edgecolor='0.3', lw=0.4, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v * 1.1, i, f'{v:.3e}', va='center', fontsize=5.5, color=COL_DE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel(r'max $|\Delta e|$')
    ax.set_xscale('log')
    ax.set_title(r'(a) Eccentricity error', loc='left', fontsize=9, fontweight='bold')
    ax.invert_yaxis()

    # max|Δx|
    ax = axes[1]
    vals = [r['max_dx'] for r in results]
    ax.barh(y, vals, height=0.6, color=COL_DX, edgecolor='0.3', lw=0.4, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v * 1.1, i, f'{v:.3e}', va='center', fontsize=5.5, color=COL_DX)
    ax.set_yticks(y)
    ax.set_yticklabels([], fontsize=5.5)
    ax.set_xlabel(r'max $|\Delta x|$')
    ax.set_xscale('log')
    ax.set_title(r'(b) Frequency parameter error', loc='left', fontsize=9, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout(w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'error_summary.{ext}'))
    plt.close(fig)
    print(f"  Saved error_summary.pdf/png")


if __name__ == '__main__':
    print("=" * 60)
    print("EXTENDED DYNAMICS COMPARISON (12 systems)")
    print("=" * 60)

    # Warmup
    print("Warming up JIT...", flush=True)
    try:
        _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                                t_end=100, rtol=1e-4, max_steps=50)
        print("  Done.", flush=True)
    except Exception as ex:
        print(f"  Warmup failed: {ex}", flush=True)

    results = []
    for q, chi1, chi2, e0, label in CASES:
        res = compare_case(q, chi1, chi2, e0, label)
        if res is not None:
            results.append(res)

    if not results:
        print("No results!")
        sys.exit(1)

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"{'Case':<48s} {'max|Δe|':>10s} {'max|Δx|':>10s} {'Δt_end':>8s} {'ms':>7s} {'speedup':>8s}")
    print("-" * 100)
    for r in results:
        sp = r['ref_elapsed_ms'] / r['our_elapsed_ms'] if r['our_elapsed_ms'] > 0 else 0
        print(f"{r['label']:<48s} {r['max_de']:10.4e} {r['max_dx']:10.4e} "
              f"{r['t_end_diff']:8.1f} {r['our_elapsed_ms']:7.1f} {sp:7.1f}x")

    # Plots
    print("\nGenerating plots...", flush=True)
    plot_dynamics_and_residuals(results)
    plot_error_summary(results)

    # Save data
    save_data = []
    for r in results:
        save_data.append({
            'label': r['label'], 'q': r['q'], 'chi1': r['chi1'],
            'chi2': r['chi2'], 'e0': r['e0'],
            'max_de': float(r['max_de']), 'max_dx': float(r['max_dx']),
            'max_de_rel': float(r['max_de_rel']),
            'max_dx_rel': float(r['max_dx_rel']),
            't_end_diff': float(r['t_end_diff']),
            'our_elapsed_ms': float(r['our_elapsed_ms']),
            'ref_elapsed_ms': float(r['ref_elapsed_ms']),
        })
    with open(os.path.join(RESULTS, 'dynamics_extended.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print("  Saved dynamics_extended.json")
    print("\nDone.")
