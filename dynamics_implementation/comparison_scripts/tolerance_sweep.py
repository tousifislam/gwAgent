"""
Tolerance sweep: compute L2 norm error of dynamics (e, x) vs pySEOBNR
for different rtol/atol values, for two representative systems.

Quantifies what tolerance is needed for a given accuracy.

Usage:
    conda activate kitp-py310
    python comparison_scripts/tolerance_sweep.py
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dynamics import setup_and_integrate

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'legend.frameon': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'lines.linewidth': 1.0,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})


def get_pyseobnr_dynamics(q, chi1, chi2, e0, omega0=0.009):
    from pyseobnr.generate_waveform import generate_modes_opt
    _, _, model = generate_modes_opt(
        q, chi1, chi2, omega0, eccentricity=e0, rel_anomaly=0.0,
        approximant="SEOBNRv5EHM", debug=True,
        settings={'use_wave_convention': True})
    dyn = model.dynamics
    return dyn[:, 0], dyn[:, 5], dyn[:, 7], dyn[0, 1], dyn[0, 3], dyn[0, 4]


def l2_norm_error(y_ref, y_ours):
    """L2 norm error: ||y_ref - y_ours|| / ||y_ref||"""
    return np.sqrt(np.sum((y_ref - y_ours)**2) / np.sum(y_ref**2))


def run_sweep(q, chi1, chi2, e0, label):
    """Run tolerance sweep for one system."""
    print(f"\n{label}:", flush=True)

    # Get pySEOBNR reference
    t_ref, e_ref, x_ref, r0, pr0, pphi0 = get_pyseobnr_dynamics(q, chi1, chi2, e0)

    rtols = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    results = []

    for rtol in rtols:
        atol = rtol * 0.1  # atol = rtol/10

        # Time the integration (3 runs, median)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            ours = setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0,
                                       rtol=rtol, atol=atol)
            times.append((time.perf_counter() - t0) * 1000)
        med_time = sorted(times)[1]

        # Compute L2 norm errors on common time grid
        t_max = min(t_ref[-1], ours['t'][-1])
        mask = t_ref <= t_max
        if np.sum(mask) < 50:
            continue

        t_common = t_ref[mask]
        e_interp = CubicSpline(ours['t'], ours['e'])(t_common)
        x_interp = CubicSpline(ours['t'], ours['x'])(t_common)

        l2_e = l2_norm_error(e_ref[mask], e_interp)
        l2_x = l2_norm_error(x_ref[mask], x_interp)
        max_de = np.max(np.abs(e_interp - e_ref[mask]))
        max_dx = np.max(np.abs(x_interp - x_ref[mask]))
        n_steps = len(ours['t'])

        print(f"  rtol={rtol:.0e}: L2_e={l2_e:.4e} L2_x={l2_x:.4e} "
              f"max|Δe|={max_de:.4e} max|Δx|={max_dx:.4e} "
              f"N={n_steps} t={med_time:.1f}ms", flush=True)

        results.append({
            'rtol': float(rtol), 'atol': float(atol),
            'l2_e': float(l2_e), 'l2_x': float(l2_x),
            'max_de': float(max_de), 'max_dx': float(max_dx),
            'n_steps': n_steps, 'time_ms': float(med_time),
        })

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("TOLERANCE SWEEP: L2 norm error vs rtol")
    print("=" * 60)

    # Warmup
    _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                            t_end=100, rtol=1e-4, max_steps=50)

    systems = [
        (3.0, 0.0, 0.0, 0.2, r'$q=3,\;\chi=0,\;e_0=0.2$'),
        (5.0, 0.3, 0.1, 0.3, r'$q=5,\;\chi=(0.3,0.1),\;e_0=0.3$'),
    ]

    all_results = {}
    for q, chi1, chi2, e0, label in systems:
        results = run_sweep(q, chi1, chi2, e0, label)
        all_results[label] = results

    # ---- Plot: L2 norm vs rtol (left) and L2 norm vs time (right) ----
    colors = ['#4c72b0', '#d62728']
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    for si, (label, results) in enumerate(all_results.items()):
        rtols = [r['rtol'] for r in results]
        l2_e = [r['l2_e'] for r in results]
        l2_x = [r['l2_x'] for r in results]
        times = [r['time_ms'] for r in results]
        col = colors[si]

        # Top left: L2_e vs rtol
        axes[0, 0].loglog(rtols, l2_e, 'o-', color=col, ms=4, lw=1, label=label)
        # Top right: L2_x vs rtol
        axes[0, 1].loglog(rtols, l2_x, 's-', color=col, ms=4, lw=1, label=label)
        # Bottom left: L2_e vs time (Pareto)
        axes[1, 0].loglog(times, l2_e, 'o-', color=col, ms=4, lw=1, label=label)
        # Bottom right: L2_x vs time (Pareto)
        axes[1, 1].loglog(times, l2_x, 's-', color=col, ms=4, lw=1, label=label)

    axes[0, 0].set_xlabel('rtol'); axes[0, 0].set_ylabel(r'$L_2$ norm error ($e$)')
    axes[0, 0].set_title(r'(a) $\|e - e_{\rm ref}\|/\|e_{\rm ref}\|$', loc='left', fontsize=9, fontweight='bold')
    axes[0, 0].legend(fontsize=6)

    axes[0, 1].set_xlabel('rtol'); axes[0, 1].set_ylabel(r'$L_2$ norm error ($x$)')
    axes[0, 1].set_title(r'(b) $\|x - x_{\rm ref}\|/\|x_{\rm ref}\|$', loc='left', fontsize=9, fontweight='bold')

    axes[1, 0].set_xlabel('Time [ms]'); axes[1, 0].set_ylabel(r'$L_2$ norm error ($e$)')
    axes[1, 0].set_title('(c) Accuracy vs speed ($e$)', loc='left', fontsize=9, fontweight='bold')
    axes[1, 0].legend(fontsize=6)

    axes[1, 1].set_xlabel('Time [ms]'); axes[1, 1].set_ylabel(r'$L_2$ norm error ($x$)')
    axes[1, 1].set_title('(d) Accuracy vs speed ($x$)', loc='left', fontsize=9, fontweight='bold')

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'tolerance_sweep.{ext}'))
    plt.close(fig)
    print(f"\nSaved tolerance_sweep.pdf/png")

    # Save data
    with open(os.path.join(RESULTS, 'tolerance_sweep.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Saved tolerance_sweep.json")

    # Summary: find the rtol where L2_e error saturates
    print("\n--- Summary ---")
    for label, results in all_results.items():
        l2_e_vals = [r['l2_e'] for r in results]
        l2_x_vals = [r['l2_x'] for r in results]
        times_vals = [r['time_ms'] for r in results]
        rtol_vals = [r['rtol'] for r in results]

        # Find saturation: where tighter rtol doesn't improve L2 by more than 10%
        for i in range(len(results) - 1):
            if l2_e_vals[i] > 0 and l2_e_vals[i+1] > 0:
                improvement = (l2_e_vals[i+1] - l2_e_vals[i]) / l2_e_vals[i]
                if improvement > -0.1:  # less than 10% improvement
                    print(f"  {label}: L2_e saturates at rtol={rtol_vals[i]:.0e} "
                          f"(L2_e={l2_e_vals[i]:.4e}, time={times_vals[i]:.0f}ms)")
                    break

    print("\nDone.")
