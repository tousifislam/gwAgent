"""
Generate training + validation data for small-spin eccentric modulation learning.

Parameters: q=[1,10], chi1/chi2=[-0.5,0.5], e0=[0.001,0.5], omega0=0.0085
Ansatz decomposition: h22_ecc(x,e,zeta,nu) baseline, models learn residuals.

Usage:
    conda activate kitp-py310
    cd modulation_learning/spin_05_04_26
    python scripts/generate_data.py
"""
import sys, os, time, json, pickle, warnings, datetime
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DYN_SRC = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/dyn_rewrite/src'
sys.path.insert(0, DYN_SRC)
from dynamics import setup_and_integrate

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')
COMMON = os.path.join(RESULTS, 'common')
TRACKING = os.path.join(BASE, 'tracking')
os.makedirs(COMMON, exist_ok=True)
os.makedirs(TRACKING, exist_ok=True)
LOGFILE = os.path.join(TRACKING, 'progress_log.md')

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


def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, 'a') as f:
        f.write(msg + '\n')


# ====================================================================
# Ansatz: h22_ecc from workflow
# ====================================================================
def h22_ecc_ansatz(x, e, zeta, nu):
    """Eccentric correction to the (2,2) mode, truncated at O(epsilon^2)."""
    e2 = e * e
    e3 = e2 * e
    eiz = np.exp(1j * zeta)
    emiz = np.exp(-1j * zeta)

    leading = (4.0 + 2.0 * e2 * eiz**2 + e * emiz + 5.0 * e * eiz) / (4.0 * (1.0 - e2))

    term_const = e * (26.0 * nu / 7.0 - 559.0 / 84.0)
    term_em2iz = e * np.exp(-2j * zeta) * (15.0 * nu / 14.0 - 95.0 / 168.0)
    term_em3iz = e2 * np.exp(-3j * zeta) * (9.0 * nu / 56.0 + 1.0 / 112.0)
    term_e3iz = e2 * np.exp(3j * zeta) * (nu / 8.0 - 49.0 / 48.0)
    term_e2iz = np.exp(2j * zeta) * (e3 * (6.0 * nu / 7.0 - 41.0 / 21.0)
                                      + e * (nu / 14.0 - 153.0 / 56.0))
    term_emiz = emiz * (e2 * (7.0 * nu / 8.0 - 59.0 / 48.0)
                        + 27.0 * nu / 14.0 - 23.0 / 14.0)
    term_eiz = eiz * (e2 * (143.0 * nu / 56.0 - 2071.0 / 336.0)
                      + nu / 14.0 - 13.0 / 7.0)

    curly = (term_const + term_em3iz + term_e3iz + term_em2iz
             + term_e2iz + term_emiz + term_eiz)

    pa_term = (x * e * 1.0) / (1.0 - e2)**2 * curly  # epsilon=1

    return leading + pa_term


# ====================================================================
# Sampling
# ====================================================================
def sample_params(n, seed):
    sampler = LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n)
    q    = 1.0 + 9.0 * samples[:, 0]
    chi1 = -0.5 + 1.0 * samples[:, 1]
    chi2 = -0.5 + 1.0 * samples[:, 2]
    e0   = 0.001 + 0.499 * samples[:, 3]
    return q, chi1, chi2, e0


# ====================================================================
# pySEOBNR waveform generation
# ====================================================================
def generate_seob(q, chi1, chi2, e0, omega0=0.0085):
    from pyseobnr.generate_waveform import generate_modes_opt
    t_ecc, modes_ecc, model_ecc = generate_modes_opt(
        q, chi1, chi2, omega0, eccentricity=e0, rel_anomaly=0.0,
        approximant="SEOBNRv5EHM", debug=True, settings={'use_wave_convention': True})
    h22_ecc = modes_ecc["2,2"]
    dyn = model_ecc.dynamics
    r0, pr0, pphi0 = dyn[0, 1], dyn[0, 3], dyn[0, 4]

    t_cir, modes_cir, _ = generate_modes_opt(
        q, chi1, chi2, omega0, eccentricity=0, rel_anomaly=0,
        approximant="SEOBNRv5EHM", debug=True, settings={'use_wave_convention': True})
    h22_cir = modes_cir["2,2"]
    return t_ecc, h22_ecc, t_cir, h22_cir, r0, pr0, pphi0


# ====================================================================
# Envelope monotonicity enforcement
# ====================================================================
def enforce_envelope_monotonicity(xi, t):
    from scipy.signal import argrelextrema
    xi_out = xi.copy()
    order = max(10, len(xi) // 300)
    peaks = argrelextrema(xi, np.greater_equal, order=order)[0]
    troughs = argrelextrema(xi, np.less_equal, order=order)[0]
    if len(peaks) < 3 or len(troughs) < 3:
        return xi_out
    peak_vals = xi[peaks].copy()
    for i in range(1, len(peak_vals)):
        peak_vals[i] = min(peak_vals[i], peak_vals[i - 1])
    trough_vals = xi[troughs].copy()
    for i in range(1, len(trough_vals)):
        trough_vals[i] = max(trough_vals[i], trough_vals[i - 1])
    upper_orig = np.interp(t, t[peaks], xi[peaks])
    upper_mono = np.interp(t, t[peaks], peak_vals)
    lower_orig = np.interp(t, t[troughs], xi[troughs])
    lower_mono = np.interp(t, t[troughs], trough_vals)
    env_range_orig = np.maximum(upper_orig - lower_orig, 1e-15)
    env_range_mono = np.maximum(upper_mono - lower_mono, 0.0)
    mid_orig = (upper_orig + lower_orig) / 2
    mid_mono = (upper_mono + lower_mono) / 2
    xi_out = mid_mono + (xi - mid_orig) * env_range_mono / env_range_orig
    return xi_out


# ====================================================================
# Modulation computation
# ====================================================================
def compute_modulations_dense(t_ecc, h22_ecc, t_cir, h22_cir):
    A_ecc_raw = np.abs(h22_ecc)
    i_peak_ecc = np.argmax(A_ecc_raw)
    t_ecc_aligned = t_ecc - t_ecc[i_peak_ecc]

    A_cir_raw = np.abs(h22_cir)
    i_peak_cir = np.argmax(A_cir_raw)
    t_cir_aligned = t_cir - t_cir[i_peak_cir]

    dt = 1.0
    t_start = max(t_ecc_aligned[0], t_cir_aligned[0])
    t_end = min(t_ecc_aligned[-1], t_cir_aligned[-1])
    t_dense = np.arange(t_start, t_end, dt)

    if len(t_dense) < 1000:
        return None

    h_ecc_dense = CubicSpline(t_ecc_aligned, np.real(h22_ecc))(t_dense) + \
                  1j * CubicSpline(t_ecc_aligned, np.imag(h22_ecc))(t_dense)
    h_cir_dense = CubicSpline(t_cir_aligned, np.real(h22_cir))(t_dense) + \
                  1j * CubicSpline(t_cir_aligned, np.imag(h22_cir))(t_dense)

    A_ecc = np.abs(h_ecc_dense)
    A_cir = np.abs(h_cir_dense)

    good = A_cir > 1e-20
    xi_amp = np.zeros_like(A_ecc)
    xi_amp[good] = (A_ecc[good] - A_cir[good]) / A_cir[good]

    phi_ecc = np.unwrap(np.angle(h_ecc_dense))
    phi_cir = np.unwrap(np.angle(h_cir_dense))
    omega_ecc = np.gradient(phi_ecc, dt)
    omega_cir = np.gradient(phi_cir, dt)

    good_w = np.abs(omega_cir) > 1e-20
    xi_omega = np.zeros_like(omega_ecc)
    xi_omega[good_w] = (omega_ecc[good_w] - omega_cir[good_w]) / omega_cir[good_w]

    inspiral_mask = t_dense < 0
    if np.sum(inspiral_mask) > 100:
        xi_amp[inspiral_mask] = enforce_envelope_monotonicity(
            xi_amp[inspiral_mask], t_dense[inspiral_mask])
        xi_omega[inspiral_mask] = enforce_envelope_monotonicity(
            xi_omega[inspiral_mask], t_dense[inspiral_mask])

    taper_mask = t_dense > 50.0
    xi_amp[taper_mask] = 0.0
    xi_omega[taper_mask] = 0.0

    return {
        't': t_dense,
        'h_ecc': h_ecc_dense, 'h_cir': h_cir_dense,
        'A_ecc': A_ecc, 'A_cir': A_cir,
        'xi_amp': xi_amp, 'xi_omega': xi_omega,
        'omega_ecc': omega_ecc, 'omega_cir': omega_cir,
        't_peak_ecc': t_ecc[i_peak_ecc],
        't_peak_cir': t_cir[i_peak_cir],
    }


# ====================================================================
# Process one waveform
# ====================================================================
def process_one(q, chi1, chi2, e0, idx):
    t0 = time.perf_counter()
    t_ecc, h22_ecc, t_cir, h22_cir, r0, pr0, pphi0 = generate_seob(q, chi1, chi2, e0)
    t_seob = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    ode = setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0, rtol=1e-8)
    t_ode = (time.perf_counter() - t1) * 1000

    mods = compute_modulations_dense(t_ecc, h22_ecc, t_cir, h22_cir)
    if mods is None:
        return None

    t_ode_aligned = ode['t'] + t_ecc[0] - mods['t_peak_ecc']
    n = len(mods['t'])
    if len(ode['t']) < 4:
        return None

    t_ode_min, t_ode_max = t_ode_aligned[0], t_ode_aligned[-1]
    valid = (mods['t'] >= t_ode_min) & (mods['t'] <= t_ode_max)

    e_ode = np.zeros(n)
    x_ode = np.zeros(n)
    zeta_ode = np.zeros(n)

    if np.sum(valid) > 10:
        e_ode[valid] = CubicSpline(t_ode_aligned, ode['e'])(mods['t'][valid])
        x_ode[valid] = CubicSpline(t_ode_aligned, ode['x'])(mods['t'][valid])
        zeta_ode[valid] = CubicSpline(t_ode_aligned, ode['zeta'])(mods['t'][valid])

    e_ode = np.clip(e_ode, 0, 0.999)
    x_ode = np.clip(x_ode, 1e-6, 1.0)

    m_1 = q / (1 + q); m_2 = 1 / (1 + q); nu = m_1 * m_2
    chi_S = (chi1 + chi2) / 2
    chi_A = (chi1 - chi2) / 2
    chi_eff = (q * chi1 + chi2) / (1 + q)

    # Ansatz modulations from h22_ecc
    xi_amp_ansatz = np.abs(h22_ecc_ansatz(x_ode, e_ode, zeta_ode, nu)) - 1.0
    xi_omega_ansatz = xi_amp_ansatz / 0.9  # Relation III approximation

    # Residuals
    delta_xi_amp = mods['xi_amp'] - xi_amp_ansatz
    delta_xi_omega = mods['xi_omega'] - xi_omega_ansatz

    return {
        'idx': idx, 'q': q, 'chi1': chi1, 'chi2': chi2, 'e0': e0,
        'nu': nu, 'chi_S': chi_S, 'chi_A': chi_A, 'chi_eff': chi_eff,
        'e': e_ode, 'x': x_ode, 'zeta': zeta_ode,
        't': mods['t'],
        'xi_amp': mods['xi_amp'], 'xi_omega': mods['xi_omega'],
        'xi_amp_ansatz': xi_amp_ansatz, 'xi_omega_ansatz': xi_omega_ansatz,
        'delta_xi_amp': delta_xi_amp, 'delta_xi_omega': delta_xi_omega,
        'h_ecc': mods['h_ecc'], 'h_cir': mods['h_cir'],
        'A_ecc': mods['A_ecc'], 'A_cir': mods['A_cir'],
        'omega_ecc': mods['omega_ecc'], 'omega_cir': mods['omega_cir'],
        't_ecc_start': t_ecc[0],
        't_peak_ecc': mods['t_peak_ecc'],
        't_seob_ms': t_seob, 't_ode_ms': t_ode,
        'n_pts': n, 'wf_length_M': t_ecc[-1] - t_ecc[0],
        'dt': 1.0,
    }


# ====================================================================
# Plotting
# ====================================================================
def plot_common(train_data, val_data):
    all_data = train_data + val_data

    # 1. Parameter space (q, e0) colored by chi_eff
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for data, label, marker in [(train_data, 'Train', 'o'), (val_data, 'Val', 's')]:
        qs = [d['q'] for d in data]; e0s = [d['e0'] for d in data]
        chis = [d['chi_eff'] for d in data]
        sc = ax.scatter(qs, e0s, c=chis, cmap='coolwarm', s=15, marker=marker,
                        alpha=0.7, label=label, edgecolors='0.3', linewidths=0.3,
                        vmin=-0.5, vmax=0.5)
    ax.set_xlabel(r'$q$'); ax.set_ylabel(r'$e_0$')
    ax.set_title('Parameter space', fontsize=9, fontweight='bold'); ax.legend(fontsize=7)
    plt.colorbar(sc, ax=ax, label=r'$\chi_{\rm eff}$')
    plt.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'parameter_space.{ext}'))
    plt.close(fig)

    # 2. Parameter space spin (chi1, chi2) colored by e0
    fig, ax = plt.subplots(figsize=(5, 4))
    for data, label, marker in [(train_data, 'Train', 'o'), (val_data, 'Val', 's')]:
        c1 = [d['chi1'] for d in data]; c2 = [d['chi2'] for d in data]
        e0s = [d['e0'] for d in data]
        sc = ax.scatter(c1, c2, c=e0s, cmap='viridis', s=15, marker=marker,
                        alpha=0.7, label=label, edgecolors='0.3', linewidths=0.3,
                        vmin=0, vmax=0.5)
    ax.set_xlabel(r'$\chi_1$'); ax.set_ylabel(r'$\chi_2$')
    ax.set_title('Spin parameter space', fontsize=9, fontweight='bold'); ax.legend(fontsize=7)
    plt.colorbar(sc, ax=ax, label=r'$e_0$')
    plt.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'parameter_space_spin.{ext}'))
    plt.close(fig)

    # 3. Waveform length histogram
    wf_lengths = [d['wf_length_M'] for d in all_data]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(np.array(wf_lengths) / 1e3, bins=30, color='#4c72b0', edgecolor='0.3', lw=0.4, alpha=0.85)
    ax.set_xlabel(r'Waveform length $[10^3\,M]$'); ax.set_ylabel('Count')
    ax.set_title('Waveform length distribution', fontsize=9, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'wf_length_histogram.{ext}'))
    plt.close(fig)

    # 4. Waveform length vs q colored by e0 bins
    fig, ax = plt.subplots(figsize=(5, 3.5))
    e0_bins = [(0, 0.05, '#4c72b0', '$e_0<0.05$'),
               (0.05, 0.1, '#55a868', '$0.05{-}0.1$'),
               (0.1, 0.2, '#c44e52', '$0.1{-}0.2$'),
               (0.2, 0.3, '#8172b2', '$0.2{-}0.3$'),
               (0.3, 0.5, '#ccb974', '$0.3{-}0.5$')]
    for lo, hi, col, label in e0_bins:
        subset = [d for d in all_data if lo <= d['e0'] < hi]
        if subset:
            qs = [d['q'] for d in subset]; wls = [d['wf_length_M'] / 1e3 for d in subset]
            ax.scatter(qs, wls, s=12, color=col, alpha=0.7, label=label,
                       edgecolors='0.3', linewidths=0.3)
    ax.set_xlabel(r'$q$'); ax.set_ylabel(r'Waveform length $[10^3\,M]$')
    ax.set_title('Length vs $q$ by eccentricity', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, ncol=2); plt.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'wf_length_vs_q.{ext}'))
    plt.close(fig)

    # 5. Waveform length vs chi_eff colored by e0
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for lo, hi, col, label in e0_bins:
        subset = [d for d in all_data if lo <= d['e0'] < hi]
        if subset:
            chis = [d['chi_eff'] for d in subset]; wls = [d['wf_length_M'] / 1e3 for d in subset]
            ax.scatter(chis, wls, s=12, color=col, alpha=0.7, label=label,
                       edgecolors='0.3', linewidths=0.3)
    ax.set_xlabel(r'$\chi_{\rm eff}$'); ax.set_ylabel(r'Waveform length $[10^3\,M]$')
    ax.set_title(r'Length vs $\chi_{\rm eff}$ by eccentricity', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, ncol=2); plt.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'wf_length_vs_chieff.{ext}'))
    plt.close(fig)

    # 6. Modulation examples: 4 cases (low q/low e, low q/high e, high q/low e, high q/high e)
    #    All at low |chi_eff|, showing SEOB (black), ansatz (blue dashed), residual (red)
    picks = []
    for q_range, e_range, label in [((1, 3), (0, 0.05), 'low q, low e'),
                                     ((1, 3), (0.3, 0.5), 'low q, high e'),
                                     ((7, 10), (0, 0.05), 'high q, low e'),
                                     ((7, 10), (0.3, 0.5), 'high q, high e')]:
        candidates = sorted(
            [d for d in train_data
             if q_range[0] <= d['q'] <= q_range[1]
             and e_range[0] <= d['e0'] <= e_range[1]
             and abs(d['chi_eff']) < 0.15],
            key=lambda d: abs(d['chi_eff']))
        if not candidates:
            candidates = [d for d in train_data
                          if q_range[0] <= d['q'] <= q_range[1]
                          and e_range[0] <= d['e0'] <= e_range[1]]
        if candidates:
            picks.append((candidates[0], label))

    if picks:
        fig, axes = plt.subplots(len(picks), 3, figsize=(10, 2.2 * len(picks)), squeeze=False)
        for i, (d, label) in enumerate(picks):
            nw = len(d['xi_omega']); mask = d['t'][:nw] <= 50
            t_plot = d['t'][:nw][mask] / 1e3
            info = f'q={d["q"]:.1f}, $e_0$={d["e0"]:.3f}, $\\chi_{{\\rm eff}}$={d["chi_eff"]:.2f}'

            # xi_amp: SEOB + ansatz
            axes[i, 0].plot(t_plot, d['xi_amp'][:nw][mask], 'k', lw=0.7, label='SEOB')
            axes[i, 0].plot(t_plot, d['xi_amp_ansatz'][:nw][mask], 'b--', lw=0.5, label='Ansatz')
            axes[i, 0].set_ylabel(r'$\xi_A$')
            axes[i, 0].text(0.02, 0.92, info, transform=axes[i, 0].transAxes, fontsize=5.5, va='top',
                            bbox=dict(facecolor='w', edgecolor='0.7', boxstyle='round,pad=0.2'))
            if i == 0: axes[i, 0].legend(fontsize=5)

            # xi_omega: SEOB + ansatz
            axes[i, 1].plot(t_plot, d['xi_omega'][:nw][mask], 'k', lw=0.7)
            axes[i, 1].plot(t_plot, d['xi_omega_ansatz'][:nw][mask], 'b--', lw=0.5)
            axes[i, 1].set_ylabel(r'$\xi_\omega$')

            # Residuals
            axes[i, 2].plot(t_plot, d['delta_xi_amp'][:nw][mask], 'r', lw=0.5, label=r'$\delta\xi_A$')
            axes[i, 2].plot(t_plot, d['delta_xi_omega'][:nw][mask], 'C1', lw=0.5, label=r'$\delta\xi_\omega$')
            axes[i, 2].set_ylabel('Residual')
            if i == 0: axes[i, 2].legend(fontsize=5)

        axes[-1, 0].set_xlabel(r'$t\;[10^3\,M]$')
        axes[-1, 1].set_xlabel(r'$t\;[10^3\,M]$')
        axes[-1, 2].set_xlabel(r'$t\;[10^3\,M]$')
        axes[0, 0].set_title(r'$\xi_A$: SEOB vs Ansatz', fontsize=9, fontweight='bold')
        axes[0, 1].set_title(r'$\xi_\omega$: SEOB vs Ansatz', fontsize=9, fontweight='bold')
        axes[0, 2].set_title('Residuals', fontsize=9, fontweight='bold')
        plt.tight_layout()
        for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'modulation_examples.{ext}'))
        plt.close(fig)

    # 7. Spin effect examples: same (q, e0) with different spins
    spin_picks = []
    target_pairs = [(2.0, 0.2), (5.0, 0.3), (8.0, 0.1)]
    for qt, et in target_pairs:
        cands = sorted(train_data, key=lambda d: abs(d['q'] - qt) + abs(d['e0'] - et) * 5)
        if len(cands) < 2:
            continue
        ref = cands[0]
        # Find cases with similar (q, e0) but very different chi_eff
        nearby = sorted([d for d in train_data
                         if abs(d['q'] - ref['q']) < 1.5 and abs(d['e0'] - ref['e0']) < 0.08],
                        key=lambda d: d['chi_eff'])
        if len(nearby) >= 2:
            spin_picks.append((nearby[0], nearby[-1]))

    if spin_picks:
        fig, axes = plt.subplots(len(spin_picks), 2, figsize=(8, 2.5 * len(spin_picks)), squeeze=False)
        for i, (d_neg, d_pos) in enumerate(spin_picks):
            nw_neg = min(len(d_neg['xi_amp']), len(d_neg['xi_omega']))
            nw_pos = min(len(d_pos['xi_amp']), len(d_pos['xi_omega']))
            mask_neg = d_neg['t'][:nw_neg] <= 50
            mask_pos = d_pos['t'][:nw_pos] <= 50

            lbl_neg = f'$\\chi_{{\\rm eff}}$={d_neg["chi_eff"]:.2f}'
            lbl_pos = f'$\\chi_{{\\rm eff}}$={d_pos["chi_eff"]:.2f}'
            info = f'q~{d_neg["q"]:.1f}, $e_0$~{d_neg["e0"]:.2f}'

            axes[i, 0].plot(d_neg['t'][:nw_neg][mask_neg] / 1e3,
                            d_neg['xi_amp'][:nw_neg][mask_neg], 'b', lw=0.6, label=lbl_neg)
            axes[i, 0].plot(d_pos['t'][:nw_pos][mask_pos] / 1e3,
                            d_pos['xi_amp'][:nw_pos][mask_pos], 'r', lw=0.6, label=lbl_pos)
            axes[i, 0].set_ylabel(r'$\xi_A$')
            axes[i, 0].legend(fontsize=5)
            axes[i, 0].text(0.02, 0.08, info, transform=axes[i, 0].transAxes, fontsize=5.5,
                            bbox=dict(facecolor='w', edgecolor='0.7', boxstyle='round,pad=0.2'))

            axes[i, 1].plot(d_neg['t'][:nw_neg][mask_neg] / 1e3,
                            d_neg['xi_omega'][:nw_neg][mask_neg], 'b', lw=0.6, label=lbl_neg)
            axes[i, 1].plot(d_pos['t'][:nw_pos][mask_pos] / 1e3,
                            d_pos['xi_omega'][:nw_pos][mask_pos], 'r', lw=0.6, label=lbl_pos)
            axes[i, 1].set_ylabel(r'$\xi_\omega$')
            axes[i, 1].legend(fontsize=5)

        axes[-1, 0].set_xlabel(r'$t\;[10^3\,M]$')
        axes[-1, 1].set_xlabel(r'$t\;[10^3\,M]$')
        axes[0, 0].set_title(r'Spin effect on $\xi_A$', fontsize=9, fontweight='bold')
        axes[0, 1].set_title(r'Spin effect on $\xi_\omega$', fontsize=9, fontweight='bold')
        plt.tight_layout()
        for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'spin_effect_examples.{ext}'))
        plt.close(fig)


def plot_feature_importance(train_data):
    from sklearn.ensemble import RandomForestRegressor

    # Downsample for feature importance — include spin features
    all_e, all_x, all_z, all_nu, all_chiS, all_chiA = [], [], [], [], [], []
    all_dya, all_dyw = [], []
    for d in train_data:
        n = min(len(d['delta_xi_amp']), len(d['delta_xi_omega']))
        e = np.clip(d['e'][:n], 1e-6, 0.95); x = np.clip(d['x'][:n], 1e-6, 0.5)
        z = d['zeta'][:n]; mask = d['t'][:n] <= 0; idx = np.where(mask)[0]
        if len(idx) > 150: idx = idx[np.linspace(0, len(idx) - 1, 150, dtype=int)]
        if len(idx) < 10: continue
        all_e.append(e[idx]); all_x.append(x[idx]); all_z.append(z[idx])
        all_nu.append(np.full(len(idx), d['nu']))
        all_chiS.append(np.full(len(idx), d['chi_S']))
        all_chiA.append(np.full(len(idx), d['chi_A']))
        all_dya.append(d['delta_xi_amp'][idx])
        all_dyw.append(d['delta_xi_omega'][idx])

    ds_e = np.concatenate(all_e); ds_x = np.concatenate(all_x)
    ds_z = np.concatenate(all_z); ds_nu = np.concatenate(all_nu)
    ds_chiS = np.concatenate(all_chiS); ds_chiA = np.concatenate(all_chiA)
    ds_dya = np.concatenate(all_dya); ds_dyw = np.concatenate(all_dyw)

    raw = np.column_stack([ds_e, ds_x, ds_nu, ds_chiS, ds_chiA,
                           np.cos(ds_z), np.sin(ds_z),
                           np.cos(2 * ds_z), np.sin(2 * ds_z)])
    feat_names = ['e', 'x', r'$\nu$', r'$\chi_S$', r'$\chi_A$',
                  r'cos($\zeta$)', r'sin($\zeta$)', r'cos(2$\zeta$)', r'sin(2$\zeta$)']
    feat_names_plain = ['e', 'x', 'nu', 'chi_S', 'chi_A',
                        'cos(zeta)', 'sin(zeta)', 'cos(2*zeta)', 'sin(2*zeta)']

    results = {}
    for target, tname, tname_plain in [(ds_dya, r'$\delta\xi_A$', 'delta_xi_amp'),
                                        (ds_dyw, r'$\delta\xi_\omega$', 'delta_xi_omega')]:
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(raw, target)
        imp = rf.feature_importances_
        results[tname_plain] = dict(zip(feat_names_plain, imp.tolist()))
        log(f'  Feature importances for {tname_plain}:')
        for name, val in sorted(zip(feat_names_plain, imp), key=lambda x: -x[1]):
            log(f'    {name:>15s}: {val:.4f}')

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax_i, (target, tname, tname_plain) in enumerate(
            [(ds_dya, r'$\delta\xi_A$', 'delta_xi_amp'),
             (ds_dyw, r'$\delta\xi_\omega$', 'delta_xi_omega')]):
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(raw, target)
        imp = rf.feature_importances_
        order = np.argsort(imp)[::-1]
        axes[ax_i].barh(range(len(feat_names)), imp[order], color='#4c72b0',
                        edgecolor='0.3', lw=0.4)
        axes[ax_i].set_yticks(range(len(feat_names)))
        axes[ax_i].set_yticklabels([feat_names[i] for i in order], fontsize=7)
        axes[ax_i].set_xlabel('Importance')
        axes[ax_i].invert_yaxis()
        axes[ax_i].set_title(f'RF importance: {tname}', fontsize=9, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(os.path.join(COMMON, f'feature_importance.{ext}'))
    plt.close(fig)

    return results


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    log(f'# Small-Spin Eccentric Modulation Learning — Progress Log\n')
    log(f'## [Step 1: Data Generation] — {now}\n')
    log(f'### Setup')
    log(f'  q in [1, 10], chi1/chi2 in [-0.5, 0.5], e0 in [0.001, 0.5], omega0 = 0.0085')
    log(f'  Ansatz decomposition: h22_ecc(x, e, zeta, nu) baseline + learned residuals')
    log(f'  300 training + 150 validation (4D LHC)')

    # Warmup JIT
    log('\nWarming up JIT...')
    _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                            t_end=100, rtol=1e-4, max_steps=50)
    log('  Done.')

    for tag, n, seed in [('training', 300, 42), ('validation', 150, 123)]:
        log(f'\n### Generating {tag} data ({n} points, seed={seed})')
        qs, chi1s, chi2s, e0s = sample_params(n, seed)
        data = []; n_fail = 0
        t_total = time.perf_counter()
        for i in range(n):
            try:
                res = process_one(qs[i], chi1s[i], chi2s[i], e0s[i], i)
                if res is not None:
                    data.append(res)
                else:
                    n_fail += 1
            except Exception as ex:
                n_fail += 1
                if (i + 1) % 50 == 0:
                    log(f'  Warning: point {i} failed: {str(ex)[:80]}')
            if (i + 1) % 30 == 0:
                log(f'  {i + 1}/{n}: {len(data)} ok, {n_fail} fail')
        elapsed = time.perf_counter() - t_total
        log(f'  {tag}: {len(data)}/{n} successful, {n_fail} failed, {elapsed:.0f}s')

        outfile = os.path.join(RESULTS, f'{tag}_data.pkl')
        with open(outfile, 'wb') as f:
            pickle.dump(data, f)
        log(f'  Saved {outfile}')

        if data:
            wls = [d['wf_length_M'] for d in data]
            ts = [d['t_seob_ms'] for d in data]
            to = [d['t_ode_ms'] for d in data]
            log(f'  Wf lengths: median={np.median(wls):.0f}M, range=[{np.min(wls):.0f}, {np.max(wls):.0f}]M')
            log(f'  SEOB timing: median={np.median(ts):.0f}ms | ODE: median={np.median(to):.0f}ms')

            # Residual stats
            dxa = np.array([np.std(d['delta_xi_amp'][d['t'] < 0]) for d in data])
            dxw = np.array([np.std(d['delta_xi_omega'][d['t'] < 0]) for d in data])
            log(f'  Residual std: delta_xi_amp median={np.median(dxa):.4e}, '
                f'delta_xi_omega median={np.median(dxw):.4e}')

    # Load data for plots
    log('\n### Generating common plots')
    with open(os.path.join(RESULTS, 'training_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(RESULTS, 'validation_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    plot_common(train_data, val_data)
    log('  Saved parameter_space.pdf, parameter_space_spin.pdf, wf_length_histogram.pdf,')
    log('  wf_length_vs_q.pdf, wf_length_vs_chieff.pdf, modulation_examples.pdf,')
    log('  spin_effect_examples.pdf')

    # Feature importance on residuals
    log('\n### Feature importance (on residuals)')
    fi_results = plot_feature_importance(train_data)
    log('  Saved feature_importance.pdf')

    # Training summary
    summary = {
        'n_train': len(train_data), 'n_val': len(val_data),
        'q_range': [float(min(d['q'] for d in train_data)),
                    float(max(d['q'] for d in train_data))],
        'e0_range': [float(min(d['e0'] for d in train_data)),
                     float(max(d['e0'] for d in train_data))],
        'chi1_range': [float(min(d['chi1'] for d in train_data)),
                       float(max(d['chi1'] for d in train_data))],
        'chi2_range': [float(min(d['chi2'] for d in train_data)),
                       float(max(d['chi2'] for d in train_data))],
        'omega0': 0.0085,
        'feature_importance': fi_results,
    }
    with open(os.path.join(COMMON, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Update CHANGELOG
    changelog = os.path.join(TRACKING, 'CHANGELOG.md')
    with open(changelog, 'r') as f:
        existing = f.read()
    entry = f"""## [{now}] DATA GENERATION

**What**: Generated training + validation data for small-spin eccentric workflow
**Details**:
- {len(train_data)} training + {len(val_data)} validation waveforms
- 4D LHC: q in [1,10], chi1/chi2 in [-0.5,0.5], e0 in [0.001,0.5]
- Ansatz decomposition computed (h22_ecc baseline + residuals)
- Feature importance on residuals computed
- All common plots generated
**Status**: DONE

---

"""
    # Insert after the first ---
    parts = existing.split('---', 2)
    if len(parts) >= 3:
        updated = parts[0] + '---\n\n' + entry + parts[2]
    else:
        updated = existing + '\n' + entry
    with open(changelog, 'w') as f:
        f.write(updated)

    log(f'\n### Summary')
    log(f'  {len(train_data)} train + {len(val_data)} val waveforms generated')
    log(f'\nStep 1 complete.')
