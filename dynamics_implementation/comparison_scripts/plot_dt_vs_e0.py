"""
Plot Δt_end vs e0 for q=1, q=5, q=10 (non-spinning).
Also regenerates the flux comparison and progress plots.

Usage:
    conda activate kitp-py310
    python comparison_scripts/plot_dt_vs_e0.py
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

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
COL_Q1  = '#4c72b0'
COL_Q5  = '#d62728'
COL_Q10 = '#2ca02c'


def get_pyseobnr_dynamics(q, chi1, chi2, e0, omega0=0.009):
    from pyseobnr.generate_waveform import generate_modes_opt
    _, _, model = generate_modes_opt(
        q, chi1, chi2, omega0, eccentricity=e0, rel_anomaly=0.0,
        approximant="SEOBNRv5EHM", debug=True,
        settings={'use_wave_convention': True})
    dyn = model.dynamics
    return dyn[:, 0], dyn[:, 5], dyn[:, 7], dyn[0, 1], dyn[0, 3], dyn[0, 4]


def measure_dt(q, chi1, chi2, e0):
    """Compute Δt_end and max|Δe| for one case."""
    try:
        t_ref, e_ref, x_ref, r0, pr0, pphi0 = get_pyseobnr_dynamics(q, chi1, chi2, e0)
        ours = setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0, rtol=1e-8)
        dt = ours['t'][-1] - t_ref[-1]

        t_max = min(ours['t'][-1], t_ref[-1])
        mask = t_ref <= t_max
        e_interp = np.interp(t_ref[mask], ours['t'], ours['e'])
        x_interp = np.interp(t_ref[mask], ours['t'], ours['x'])
        max_de = np.max(np.abs(e_interp - e_ref[mask]))
        max_dx = np.max(np.abs(x_interp - x_ref[mask]))
        return dt, max_de, max_dx
    except Exception as ex:
        print(f"    FAILED: {ex}")
        return None, None, None


if __name__ == '__main__':
    print("=" * 60)
    print("Δt_end vs e0 for q=1, q=5, q=10")
    print("=" * 60)

    # Warmup
    _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                            t_end=100, rtol=1e-4, max_steps=50)

    e0_vals = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    configs = [
        (1.0,  0.0, 0.0, COL_Q1,  r'$q=1$'),
        (5.0,  0.0, 0.0, COL_Q5,  r'$q=5$'),
        (10.0, 0.0, 0.0, COL_Q10, r'$q=10$'),
    ]

    all_data = {}
    for q, chi1, chi2, col, label in configs:
        print(f"\n{label}:", flush=True)
        dts, des, dxs, e0s_ok = [], [], [], []
        for e0 in e0_vals:
            print(f"  e0={e0:.2f}...", end='', flush=True)
            dt, de, dx = measure_dt(q, chi1, chi2, e0)
            if dt is not None:
                dts.append(dt)
                des.append(de)
                dxs.append(dx)
                e0s_ok.append(e0)
                print(f" Δt={dt:.1f}M, max|Δe|={de:.4e}, max|Δx|={dx:.4e}")
            else:
                print(" failed")
        all_data[label] = {'e0': e0s_ok, 'dt': dts, 'de': des, 'dx': dxs}

    # ---- Plot 1: Δt_end vs e0 ----
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    # (a) Δt_end
    ax = axes[0]
    for (q, chi1, chi2, col, label), data in zip(configs, all_data.values()):
        ax.plot(data['e0'], data['dt'], 'o-', color=col, ms=4, lw=1.0, label=label)
    ax.axhline(0, color='0.5', ls='-', lw=0.3)
    ax.set_xlabel(r'$e_0$')
    ax.set_ylabel(r'$\Delta t_{\rm end}\;[M]$')
    ax.set_title(r'(a) Inspiral time offset', loc='left', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

    # (b) max|Δe|
    ax = axes[1]
    for (q, chi1, chi2, col, label), data in zip(configs, all_data.values()):
        ax.semilogy(data['e0'], data['de'], 'o-', color=col, ms=4, lw=1.0, label=label)
    ax.set_xlabel(r'$e_0$')
    ax.set_ylabel(r'max $|\Delta e|$')
    ax.set_title(r'(b) Eccentricity error', loc='left', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

    # (c) max|Δx|
    ax = axes[2]
    for (q, chi1, chi2, col, label), data in zip(configs, all_data.values()):
        ax.semilogy(data['e0'], data['dx'], 'o-', color=col, ms=4, lw=1.0, label=label)
    ax.set_xlabel(r'$e_0$')
    ax.set_ylabel(r'max $|\Delta x|$')
    ax.set_title(r'(c) Frequency error', loc='left', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

    plt.tight_layout(w_pad=0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'dt_vs_e0.{ext}'))
    plt.close(fig)
    print(f"\nSaved dt_vs_e0.pdf/png")

    # Save data
    save = {}
    for (q, chi1, chi2, col, label) in configs:
        d = all_data[label]
        save[label] = {k: [float(v) for v in vals] for k, vals in d.items()}
    with open(os.path.join(RESULTS, 'dt_vs_e0.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print("Saved dt_vs_e0.json")
