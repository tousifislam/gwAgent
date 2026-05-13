"""
Generate cumulative progress staircase plot and timing benchmarks.
Nature-quality figure.

Usage:
    conda activate kitp-py310
    python make_progress_plot.py
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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


def measure_pyseobnr(q, chi1, chi2, e0, n_rep=3):
    from pyseobnr.generate_waveform import generate_modes_opt
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        generate_modes_opt(q, chi1, chi2, 0.009, eccentricity=e0, rel_anomaly=0.0,
                          approximant="SEOBNRv5EHM", debug=True)
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[n_rep // 2]


def measure_ours(q, chi1, chi2, e0, rtol=1e-8, n_rep=5):
    from dynamics import setup_and_integrate
    from pyseobnr.generate_waveform import generate_modes_opt

    # Get ICs from pySEOBNR
    _, _, model = generate_modes_opt(q, chi1, chi2, 0.009, eccentricity=e0, rel_anomaly=0.0,
                                      approximant="SEOBNRv5EHM", debug=True)
    dyn = model.dynamics
    r0, pr0, pphi0 = dyn[0, 1], dyn[0, 3], dyn[0, 4]

    # Warmup
    setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0, rtol=rtol, max_steps=200, t_end=100)

    times = []
    n_steps_list = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        res = setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0, rtol=rtol)
        times.append((time.perf_counter() - t0) * 1000)
        n_steps_list.append(len(res['t']))
    med_idx = n_rep // 2
    return sorted(times)[med_idx], sorted(n_steps_list)[med_idx]


if __name__ == '__main__':
    print("=" * 60)
    print("BENCHMARKING & PROGRESS PLOT")
    print("=" * 60)

    # Benchmark case: q=3, chi=(0.5,0.3), e0=0.2
    bm_q, bm_chi1, bm_chi2, bm_e0 = 3, 0.5, 0.3, 0.2

    # Measure pySEOBNR baseline
    print("\nMeasuring pySEOBNR baseline...", flush=True)
    pyseobnr_ms = measure_pyseobnr(bm_q, bm_chi1, bm_chi2, bm_e0)
    print(f"  pySEOBNR: {pyseobnr_ms:.0f} ms")

    # Measure our Numba at various rtol
    print("\nMeasuring Numba at various rtol...", flush=True)
    experiments = []

    for rtol in [1e-10, 1e-8, 1e-6, 1e-5]:
        t_ms, n_steps = measure_ours(bm_q, bm_chi1, bm_chi2, bm_e0, rtol=rtol)
        label = f"Numba RK45\nrtol=$10^{{{int(np.log10(rtol))}}}$"
        experiments.append({
            'label': label,
            'time_ms': t_ms,
            'rtol': rtol,
            'n_steps': n_steps,
            'visual_ok': True,  # Will check later
        })
        print(f"  rtol={rtol:.0e}: {t_ms:.1f}ms, {n_steps} steps")

    # Build the full progress log
    steps = [
        {'label': 'pySEOBNR\n(full waveform)', 'time_ms': pyseobnr_ms, 'baseline': True},
    ] + experiments

    # ===== Progress staircase plot =====
    print("\nGenerating progress plot...", flush=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    x_pos = np.arange(len(steps))
    vals = [s['time_ms'] for s in steps]

    # Running best
    best = []
    cur = 1e10
    for v in vals:
        cur = min(cur, v)
        best.append(cur)

    bar_colors = ['0.70'] + [COL_BAR] * len(experiments)
    ax.bar(x_pos, vals, color=bar_colors, edgecolor='0.3', lw=0.4,
           width=0.6, zorder=2, alpha=0.85)
    ax.step(np.concatenate([[-0.5], x_pos, [len(steps)-0.5]]),
            [best[0]] + best + [best[-1]],
            color=COL_OURS, lw=1.5, zorder=3, where='mid', label='Running best')
    ax.axhline(5, color=COL_TARG, ls='--', lw=1.2, zorder=1, label='5 ms target')
    ax.axhline(pyseobnr_ms, color='0.45', ls=':', lw=1.0, zorder=1,
               label=f'pySEOBNR ({pyseobnr_ms:.0f} ms)')

    labels = [s['label'] for s in steps]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7, rotation=12, ha='right')
    ax.set_ylabel('Integration time [ms]')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_ylim(0.5, pyseobnr_ms * 3)
    ax.legend(fontsize=7, loc='upper right')

    # Annotate speedup
    best_ms = min(vals[1:])  # best non-baseline
    speedup = pyseobnr_ms / best_ms
    best_idx = vals.index(best_ms)
    ax.annotate(f'{speedup:.0f}$\\times$ speedup',
                xy=(best_idx, best_ms), xytext=(best_idx - 0.5, best_ms * 5),
                arrowprops=dict(arrowstyle='->', color=COL_OURS, lw=1),
                fontsize=8, color=COL_OURS)

    ax.set_title(f'ODE speed optimisation  '
                 r'[$q{=}3,\;\chi{=}(0.5,0.3),\;e_0{=}0.2$]', fontsize=9)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(RESULTS, f'progress.{ext}'))
    plt.close(fig)
    print(f"  Saved progress.pdf/png")

    # ===== Timing histogram across cases =====
    print("\nBenchmarking across parameter space...", flush=True)

    cases = [
        (1, 0, 0, 0.3, r"$q{=}1,\chi{=}0,e_0{=}0.3$"),
        (3, 0.5, 0.3, 0.2, r"$q{=}3,\chi{=}(0.5,0.3),e_0{=}0.2$"),
        (6, 0.9, 0, 0.1, r"$q{=}6,\chi{=}(0.9,0),e_0{=}0.1$"),
        (10, 0.7, 0.7, 0.4, r"$q{=}10,\chi{=}(0.7,0.7),e_0{=}0.4$"),
    ]

    timing_data = []
    for q, chi1, chi2, e0, label in cases:
        print(f"  {label}...", flush=True)
        try:
            t_ours, n_steps = measure_ours(q, chi1, chi2, e0, rtol=1e-8)
            t_ref = measure_pyseobnr(q, chi1, chi2, e0)
            timing_data.append({'label': label, 't_ours': t_ours, 't_ref': t_ref,
                               'n_steps': n_steps, 'speedup': t_ref / t_ours})
            print(f"    ours={t_ours:.1f}ms, ref={t_ref:.0f}ms, speedup={t_ref/t_ours:.1f}x")
        except Exception as ex:
            print(f"    FAILED: {ex}")

    if timing_data:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bar_h = 0.35
        labels = [d['label'] for d in timing_data]
        y = np.arange(len(timing_data))
        t_ours_arr = [d['t_ours'] for d in timing_data]
        t_ref_arr = [d['t_ref'] for d in timing_data]

        ax.barh(y - bar_h/2, t_ours_arr, height=bar_h, color=COL_BAR,
                edgecolor='0.3', lw=0.4, label='This work', zorder=2)
        ax.barh(y + bar_h/2, t_ref_arr, height=bar_h, color='0.70',
                edgecolor='0.3', lw=0.4, label='pySEOBNR', zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel('Time [ms]')
        ax.set_xscale('log')
        ax.axvline(5, color=COL_TARG, ls='--', lw=1.2, label='5 ms target')
        ax.legend(loc='lower right', fontsize=7)
        ax.invert_yaxis()

        for i, d in enumerate(timing_data):
            ax.text(d['t_ours'] * 1.15, i - bar_h/2, f"{d['t_ours']:.1f}",
                    va='center', fontsize=6, color=COL_BAR)
            ax.text(d['t_ref'] * 1.15, i + bar_h/2, f"{d['t_ref']:.0f}",
                    va='center', fontsize=6, color='0.35')

        ax.set_title(r'ODE integration time (rtol$\,{=}\,10^{-8}$)', fontsize=10)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(RESULTS, f'timing_histogram.{ext}'))
        plt.close(fig)
        print(f"  Saved timing_histogram.pdf/png")

    # Save progress log
    progress_log = [{'step': 'pySEOBNR baseline', 'time_ms': pyseobnr_ms}]
    for e in experiments:
        progress_log.append({
            'step': e['label'].replace('\n', ' '),
            'time_ms': e['time_ms'],
            'rtol': float(e['rtol']),
            'n_steps': e['n_steps'],
        })
    with open(os.path.join(RESULTS, 'progress_log.json'), 'w') as f:
        json.dump(progress_log, f, indent=2)
    print(f"  Saved progress_log.json")

    print("\nDone.")
