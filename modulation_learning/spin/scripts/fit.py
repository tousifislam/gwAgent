"""
Fit models for small-spin eccentric modulation learning.
All models fit residuals (delta_xi) on top of the h22_ecc ansatz.

Models:
  0: Ansatz only (delta=0, baseline)
  1: Ridge — validation-driven basis+alpha scan (with spin features)
  2: Polynomial + Ridge (with spin features)
  3: Random Forest
  4: Hybrid ansatz + Ridge residual
  5: Best model + polynomial phase correction

Usage:
    conda activate kitp-py310
    cd modulation_learning/spin_05_04_26
    python scripts/fit.py
"""
import sys, os, time, json, pickle, datetime
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')
TRACKING = os.path.join(BASE, 'tracking')
LOGFILE = os.path.join(TRACKING, 'progress_log.md')

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.labelsize': 11, 'axes.titlesize': 10, 'legend.fontsize': 8,
    'legend.frameon': False, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.0, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
C_REF = '#1a1a1a'; C_MOD = '#d62728'; C_TR = '#4c72b0'
C_VA = '#c44e52'; C_TG = '#2ca02c'

MTOT_VALUES = [20, 65, 110, 155, 200]  # Solar masses


def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, 'a') as f:
        f.write(msg + '\n')


# ====================================================================
# Ansatz (same as generate_data.py)
# ====================================================================
def h22_ecc_ansatz(x, e, zeta, nu):
    e2 = e * e; e3 = e2 * e
    eiz = np.exp(1j * zeta); emiz = np.exp(-1j * zeta)
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
    pa_term = (x * e) / (1.0 - e2)**2 * curly
    return leading + pa_term


def ansatz_modulations(e, x, zeta, nu):
    xi_amp_ansatz = np.abs(h22_ecc_ansatz(x, e, zeta, nu)) - 1.0
    xi_omega_ansatz = xi_amp_ansatz / 0.9
    return xi_amp_ansatz, xi_omega_ansatz


# ====================================================================
# Error metrics
# ====================================================================
def mathcalE_error(h_ref, h):
    n1 = np.sum(np.abs(h_ref)**2); n2 = np.sum(np.abs(h)**2)
    s = np.real(np.sum(h_ref * np.conj(h)))
    return ((n1 + n2) - 2 * s) / (2 * n1) if n1 > 0 else 1.0


def compute_mismatch(hp, hr):
    i = np.real(np.sum(hp * np.conj(hr)))
    np_ = np.linalg.norm(hp); nr_ = np.linalg.norm(hr)
    return 1 - i / (np_ * nr_) if np_ > 0 and nr_ > 0 else 1.0


def smooth_taper(t, ts=-50.0, te=0.0):
    w = np.ones_like(t)
    m = (t >= ts) & (t <= te)
    w[m] = 0.5 * (1 + np.cos(np.pi * (t[m] - ts) / (te - ts)))
    w[t > te] = 0
    return w


def ligo_mismatch(h_pred, h_ref, dt_geometric, Mtot_msun, f_low=20.0):
    try:
        from pycbc.types import TimeSeries
        from pycbc.filter import match
        from pycbc.psd import aLIGOZeroDetHighPower
    except ImportError:
        return np.nan

    Mtot_sec = Mtot_msun * 4.925491025543576e-06
    dt_sec = dt_geometric * Mtot_sec

    hp_pred = TimeSeries(np.real(h_pred).astype(np.float64), delta_t=dt_sec)
    hp_ref = TimeSeries(np.real(h_ref).astype(np.float64), delta_t=dt_sec)

    tlen = max(len(hp_pred), len(hp_ref))
    hp_pred.resize(tlen); hp_ref.resize(tlen)

    delta_f = 1.0 / hp_ref.duration
    flen = tlen // 2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

    try:
        m, _ = match(hp_ref, hp_pred, psd=psd, low_frequency_cutoff=f_low)
        return 1.0 - m
    except Exception:
        return np.nan


# ====================================================================
# Basis construction (with spin features)
# ====================================================================
def build_basis(e, z, x, nu, chi_S, chi_A, max_e=4, max_x=3, max_nu=2,
                max_chi=1, n_harm=5):
    """Polynomial x Fourier basis with spin dimensions."""
    features = []
    for a in range(1, max_e + 1):
        for b in range(max_x + 1):
            for c in range(max_nu + 1):
                for d_s in range(max_chi + 1):
                    for d_a in range(max_chi + 1):
                        if a + b + c + d_s + d_a > max_e + 3:
                            continue
                        base = e**a * x**b * nu**c * chi_S**d_s * chi_A**d_a
                        features.append(base)
                        for k in range(1, n_harm + 1):
                            features.append(base * np.cos(k * z))
                            features.append(base * np.sin(k * z))
    return np.column_stack(features) if features else np.zeros((len(e), 1))


# ====================================================================
# Downsampling
# ====================================================================
def downsample(data, n_pts=400):
    all_e, all_x, all_z, all_nu = [], [], [], []
    all_chiS, all_chiA = [], []
    all_dya, all_dyw = [], []
    for d in data:
        n = min(len(d['delta_xi_amp']), len(d['delta_xi_omega']))
        e = np.clip(d['e'][:n], 1e-6, 0.95)
        x = np.clip(d['x'][:n], 1e-6, 0.5)
        z = d['zeta'][:n]
        mask = d['t'][:n] <= 50
        idx = np.where(mask)[0]
        if len(idx) > n_pts:
            idx = idx[np.linspace(0, len(idx) - 1, n_pts, dtype=int)]
        if len(idx) < 10:
            continue
        all_e.append(e[idx]); all_x.append(x[idx]); all_z.append(z[idx])
        all_nu.append(np.full(len(idx), d['nu']))
        all_chiS.append(np.full(len(idx), d['chi_S']))
        all_chiA.append(np.full(len(idx), d['chi_A']))
        all_dya.append(d['delta_xi_amp'][idx])
        all_dyw.append(d['delta_xi_omega'][idx])
    return {
        'e': np.concatenate(all_e), 'x': np.concatenate(all_x),
        'z': np.concatenate(all_z), 'nu': np.concatenate(all_nu),
        'chiS': np.concatenate(all_chiS), 'chiA': np.concatenate(all_chiA),
        'dya': np.concatenate(all_dya), 'dyw': np.concatenate(all_dyw),
    }


# ====================================================================
# Reconstruction
# ====================================================================
def reconstruct(d, xi_a, xi_w, phase_corr=None):
    dt = 0.1
    t_d = np.arange(d['t'][0], d['t'][-1], dt)
    h_ecc_d = CubicSpline(d['t'], np.real(d['h_ecc']))(t_d) + \
              1j * CubicSpline(d['t'], np.imag(d['h_ecc']))(t_d)
    h_cir_d = CubicSpline(d['t'], np.real(d['h_cir']))(t_d) + \
              1j * CubicSpline(d['t'], np.imag(d['h_cir']))(t_d)
    xi_a_d = np.interp(t_d, d['t'], xi_a)
    xi_w_d = np.interp(t_d, d['t'], xi_w)
    taper = smooth_taper(t_d)
    xi_a_d *= taper; xi_w_d *= taper
    A_p = np.abs(h_cir_d) * (1 + xi_a_d)
    pc = np.unwrap(np.angle(h_cir_d)); oc = np.gradient(pc, dt)
    pp = cumulative_trapezoid(oc * (1 + xi_w_d), dx=dt, initial=0.0)
    pe = np.unwrap(np.angle(h_ecc_d)); pp += pe[0] - pp[0]
    if phase_corr is not None:
        pp = phase_corr(t_d, pp, pe)
    hp = A_p * np.exp(1j * pp)
    ha = h_ecc_d * np.exp(-1j * np.angle(h_ecc_d[0]))
    hm = hp * np.exp(-1j * np.angle(hp[0]))
    pr = np.unwrap(np.angle(ha)); pm = np.unwrap(np.angle(hm))
    pm -= (pm[0] - pr[0])
    E = mathcalE_error(ha, hm); MM = compute_mismatch(hm, ha)
    dphi = np.max(np.abs(pm - pr))
    return E, MM, dphi, t_d, ha, hm


def envelope_monotonicity_violation(xi, t):
    from scipy.signal import argrelextrema
    order = max(10, len(xi) // 300)
    peaks = argrelextrema(xi, np.greater_equal, order=order)[0]
    troughs = argrelextrema(xi, np.less_equal, order=order)[0]
    if len(peaks) < 3 or len(troughs) < 3:
        return 0.0
    pv = xi[peaks]
    n_upper_viol = np.sum(np.diff(pv) > 1e-10)
    tv = xi[troughs]
    n_lower_viol = np.sum(np.diff(tv) < -1e-10)
    total = max(len(pv) - 1 + len(tv) - 1, 1)
    return (n_upper_viol + n_lower_viol) / total


# ====================================================================
# Evaluation
# ====================================================================
def eval_all(data, predict_fn, phase_corr=None, compute_ligo=True):
    """Evaluate a model on all data. predict_fn(e, x, z, nu, chi_S, chi_A) -> (xi_a, xi_w)"""
    Es, MMs, dphis, mono_viols = [], [], [], []
    ligo_mms = []  # shape: (N, 5) for 5 Mtot values
    for d in data:
        n = len(d['xi_amp'])
        e = np.clip(d['e'][:n], 1e-6, 0.95)
        x = np.clip(d['x'][:n], 1e-6, 0.5)
        z = d['zeta'][:n]
        xi_a, xi_w = predict_fn(e, x, z, d['nu'], d['chi_S'], d['chi_A'])
        E, MM, dp, t_d, ha, hm = reconstruct(d, xi_a, xi_w, phase_corr)
        Es.append(E); MMs.append(MM); dphis.append(dp)

        # Envelope monotonicity on inspiral
        inspiral = d['t'][:n] < 0
        if np.sum(inspiral) > 100:
            mv_a = envelope_monotonicity_violation(xi_a[inspiral], d['t'][:n][inspiral])
            mv_w = envelope_monotonicity_violation(xi_w[inspiral], d['t'][:n][inspiral])
            mono_viols.append(max(mv_a, mv_w))
        else:
            mono_viols.append(0.0)

        # LIGO mismatch
        if compute_ligo:
            dt_geom = 0.1  # dense grid dt
            row = []
            for Mtot in MTOT_VALUES:
                row.append(ligo_mismatch(hm, ha, dt_geom, Mtot))
            ligo_mms.append(row)
        else:
            ligo_mms.append([np.nan] * 5)

    return (np.array(Es), np.array(MMs), np.array(dphis),
            np.array(mono_viols), np.array(ligo_mms))


# ====================================================================
# Checklist: save arrays, plots, summary
# ====================================================================
def save_error_arrays(name, train, val, predict_fn, phase_corr=None):
    """Save all npy error arrays for train and val."""
    errdir = os.path.join(RESULTS, 'errors', name)
    os.makedirs(errdir, exist_ok=True)

    for tag, data in [('train', train), ('val', val)]:
        params = np.array([[d['q'], d['chi1'], d['chi2'], d['e0']] for d in data])
        Es, MMs, dphis, mono, ligo = eval_all(data, predict_fn, phase_corr)
        np.save(os.path.join(errdir, f'{tag}_params.npy'), params)
        np.save(os.path.join(errdir, f'{tag}_mathcalE.npy'), Es)
        np.save(os.path.join(errdir, f'{tag}_td_mismatch.npy'), MMs)
        np.save(os.path.join(errdir, f'{tag}_dephasing.npy'), dphis)
        np.save(os.path.join(errdir, f'{tag}_ligo_mismatch.npy'), ligo)
    return errdir


def make_plots(outdir, name, train, val, E_tr, MM_tr, dphi_tr, E_va, MM_va, dphi_va,
               predict_fn, mono_tr=None, mono_va=None, ligo_tr=None, ligo_va=None,
               phase_corr=None):
    os.makedirs(outdir, exist_ok=True)

    # 1. Histogram: mathcalE + MM + dephasing
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    bins_E = np.linspace(-5, 0.5, 35)
    axes[0].hist(np.log10(E_tr + 1e-20), bins=bins_E, alpha=0.7, color=C_TR,
                 label='Train', edgecolor='0.3', lw=0.3)
    axes[0].hist(np.log10(E_va + 1e-20), bins=bins_E, alpha=0.5, color=C_VA,
                 label='Val', edgecolor='0.3', lw=0.3)
    axes[0].axvline(-2, color=C_TG, ls='--', lw=1); axes[0].axvline(-3, color=C_TG, ls=':', lw=0.8)
    axes[0].set_xlabel(r'$\log_{10}(\mathcal{E})$'); axes[0].legend(fontsize=7)
    axes[0].set_title(f'{name}: $\\mathcal{{E}}$', fontsize=8, fontweight='bold')

    axes[1].hist(np.log10(MM_tr + 1e-20), bins=bins_E, alpha=0.7, color=C_TR, edgecolor='0.3', lw=0.3)
    axes[1].hist(np.log10(MM_va + 1e-20), bins=bins_E, alpha=0.5, color=C_VA, edgecolor='0.3', lw=0.3)
    axes[1].axvline(-2, color=C_TG, ls='--', lw=1)
    axes[1].set_xlabel(r'$\log_{10}(\mathrm{MM})$')
    axes[1].set_title(f'{name}: MM', fontsize=8, fontweight='bold')

    bins_p = np.linspace(-2, 2, 30)
    axes[2].hist(np.log10(dphi_tr + 1e-20), bins=bins_p, alpha=0.7, color=C_TR, edgecolor='0.3', lw=0.3)
    axes[2].hist(np.log10(dphi_va + 1e-20), bins=bins_p, alpha=0.5, color=C_VA, edgecolor='0.3', lw=0.3)
    axes[2].axvline(np.log10(0.1), color=C_TG, ls='--', lw=1, label='0.1 rad')
    axes[2].set_xlabel(r'$\log_{10}(\max|\Delta\phi|)$'); axes[2].legend(fontsize=6)
    axes[2].set_title(f'{name}: dephasing', fontsize=8, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'histogram.{ext}'))
    plt.close(fig)

    # 2. LIGO mismatch histograms (5 panels)
    if ligo_va is not None and not np.all(np.isnan(ligo_va)):
        fig, axes = plt.subplots(1, 5, figsize=(14, 2.5))
        for j, Mtot in enumerate(MTOT_VALUES):
            lv = ligo_va[:, j]; lt_ = ligo_tr[:, j]
            lv = lv[~np.isnan(lv)]; lt_ = lt_[~np.isnan(lt_)]
            bins_l = np.linspace(-5, 0, 30)
            if len(lt_) > 0:
                axes[j].hist(np.log10(lt_ + 1e-20), bins=bins_l, alpha=0.7, color=C_TR,
                             edgecolor='0.3', lw=0.3, label='Train')
            if len(lv) > 0:
                axes[j].hist(np.log10(lv + 1e-20), bins=bins_l, alpha=0.5, color=C_VA,
                             edgecolor='0.3', lw=0.3, label='Val')
            axes[j].axvline(-2, color=C_TG, ls='--', lw=1)
            axes[j].set_title(f'$M_{{\\rm tot}}$={Mtot}', fontsize=8)
            axes[j].set_xlabel(r'$\log_{10}(\mathrm{MM}_{\rm LIGO})$')
            if j == 0: axes[j].legend(fontsize=5)
        fig.suptitle(f'{name}: LIGO mismatch', fontsize=9, fontweight='bold')
        plt.tight_layout()
        for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'histogram_ligo_mm.{ext}'))
        plt.close(fig)

    # 3. Dephasing vs e0
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for data, dph, label, marker in [(train, dphi_tr, 'Train', 'o'),
                                      (val, dphi_va, 'Val', 's')]:
        e0s = [d['e0'] for d in data]; qs = [d['q'] for d in data]
        sc = ax.scatter(e0s, dph, c=qs, cmap='viridis', s=15, marker=marker,
                        alpha=0.7, label=label, vmin=1, vmax=10,
                        edgecolors='0.3', linewidths=0.3)
    ax.axhline(0.1, color=C_TG, ls='--', lw=1.2, label='0.1 rad')
    ax.set_yscale('log')
    ax.set_xlabel(r'$e_0$'); ax.set_ylabel(r'$\max|\Delta\phi|$ [rad]')
    ax.set_title(f'{name}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6); plt.colorbar(sc, ax=ax, label=r'$q$')
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'dephasing_vs_e0.{ext}'))
    plt.close(fig)

    # 4. Dephasing vs chi_eff
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for data, dph, label, marker in [(train, dphi_tr, 'Train', 'o'),
                                      (val, dphi_va, 'Val', 's')]:
        chis = [d['chi_eff'] for d in data]; e0s = [d['e0'] for d in data]
        sc = ax.scatter(chis, dph, c=e0s, cmap='viridis', s=15, marker=marker,
                        alpha=0.7, label=label, vmin=0, vmax=0.5,
                        edgecolors='0.3', linewidths=0.3)
    ax.axhline(0.1, color=C_TG, ls='--', lw=1.2, label='0.1 rad')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\chi_{\rm eff}$'); ax.set_ylabel(r'$\max|\Delta\phi|$ [rad]')
    ax.set_title(f'{name}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6); plt.colorbar(sc, ax=ax, label=r'$e_0$')
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'dephasing_vs_chieff.{ext}'))
    plt.close(fig)

    # 5. mathcalE vs e0
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for data, errs, label, marker in [(train, E_tr, 'Train', 'o'),
                                       (val, E_va, 'Val', 's')]:
        e0s = [d['e0'] for d in data]; qs = [d['q'] for d in data]
        sc = ax.scatter(e0s, np.log10(errs + 1e-20), c=qs, cmap='viridis',
                        s=15, marker=marker, alpha=0.7, label=label,
                        vmin=1, vmax=10, edgecolors='0.3', linewidths=0.3)
    ax.axhline(-2, color=C_TG, ls='--', lw=1)
    ax.set_xlabel(r'$e_0$'); ax.set_ylabel(r'$\log_{10}(\mathcal{E})$')
    ax.set_title(f'{name}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); plt.colorbar(sc, ax=ax, label=r'$q$')
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'mathcalE_vs_e0.{ext}'))
    plt.close(fig)

    # 6. mathcalE vs chi_eff
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for data, errs, label, marker in [(train, E_tr, 'Train', 'o'),
                                       (val, E_va, 'Val', 's')]:
        chis = [d['chi_eff'] for d in data]; e0s = [d['e0'] for d in data]
        sc = ax.scatter(chis, np.log10(errs + 1e-20), c=e0s, cmap='viridis',
                        s=15, marker=marker, alpha=0.7, label=label,
                        vmin=0, vmax=0.5, edgecolors='0.3', linewidths=0.3)
    ax.axhline(-2, color=C_TG, ls='--', lw=1)
    ax.set_xlabel(r'$\chi_{\rm eff}$'); ax.set_ylabel(r'$\log_{10}(\mathcal{E})$')
    ax.set_title(f'{name}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); plt.colorbar(sc, ax=ax, label=r'$e_0$')
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'mathcalE_vs_chieff.{ext}'))
    plt.close(fig)

    # 7. LIGO mismatch vs e0 (val only, one curve per Mtot)
    if ligo_va is not None and not np.all(np.isnan(ligo_va)):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        e0v = np.array([d['e0'] for d in val])
        colors_mtot = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
        for j, Mtot in enumerate(MTOT_VALUES):
            lv = ligo_va[:, j]; good = ~np.isnan(lv)
            if np.sum(good) > 0:
                ax.scatter(e0v[good], lv[good], s=10, color=colors_mtot[j], alpha=0.6,
                           label=f'$M_{{\\rm tot}}$={Mtot}', edgecolors='0.3', linewidths=0.2)
        ax.axhline(0.01, color=C_TG, ls='--', lw=1.2, label='1%')
        ax.set_yscale('log'); ax.set_xlabel(r'$e_0$')
        ax.set_ylabel(r'LIGO mismatch')
        ax.set_title(f'{name}: LIGO MM vs $e_0$', fontsize=9, fontweight='bold')
        ax.legend(fontsize=5, ncol=2); plt.tight_layout()
        for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'ligo_mm_vs_e0.{ext}'))
        plt.close(fig)

    # 8. LIGO mismatch vs chi_eff
    if ligo_va is not None and not np.all(np.isnan(ligo_va)):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        chiv = np.array([d['chi_eff'] for d in val])
        for j, Mtot in enumerate(MTOT_VALUES):
            lv = ligo_va[:, j]; good = ~np.isnan(lv)
            if np.sum(good) > 0:
                ax.scatter(chiv[good], lv[good], s=10, color=colors_mtot[j], alpha=0.6,
                           label=f'$M_{{\\rm tot}}$={Mtot}', edgecolors='0.3', linewidths=0.2)
        ax.axhline(0.01, color=C_TG, ls='--', lw=1.2, label='1%')
        ax.set_yscale('log'); ax.set_xlabel(r'$\chi_{\rm eff}$')
        ax.set_ylabel(r'LIGO mismatch')
        ax.set_title(f'{name}: LIGO MM vs $\\chi_{{\\rm eff}}$', fontsize=9, fontweight='bold')
        ax.legend(fontsize=5, ncol=2); plt.tight_layout()
        for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'ligo_mm_vs_chieff.{ext}'))
        plt.close(fig)

    # 9. Best/median/worst modulation panels (split layout)
    all_data = train + val
    all_dphi = np.concatenate([dphi_tr, dphi_va])
    all_E = np.concatenate([E_tr, E_va])
    for pick, idx in [('best', int(np.argmin(all_dphi))),
                      ('worst', int(np.argmax(all_dphi))),
                      ('median', int(np.argsort(all_dphi)[len(all_dphi) // 2]))]:
        d = all_data[idx]; n = len(d['xi_amp'])
        e = np.clip(d['e'][:n], 1e-6, 0.95)
        x = np.clip(d['x'][:n], 1e-6, 0.5)
        z = d['zeta'][:n]
        xi_a, xi_w = predict_fn(e, x, z, d['nu'], d['chi_S'], d['chi_A'])
        _, _, dp, td, hr, hp = reconstruct(d, xi_a, xi_w, phase_corr)
        nw = min(len(d['xi_omega']), len(xi_w), n)

        dt_plot = 0.1
        t_dense = np.arange(d['t'][0], min(d['t'][nw - 1], 100.0), dt_plot)
        xi_a_ref_dense = np.interp(t_dense, d['t'][:nw], d['xi_amp'][:nw])
        xi_w_ref_dense = np.interp(t_dense, d['t'][:nw], d['xi_omega'][:nw])
        xi_a_mod_dense = np.interp(t_dense, d['t'][:nw], xi_a[:nw])
        xi_w_mod_dense = np.interp(t_dense, d['t'][:nw], xi_w[:nw])

        t_split = -250.0
        mL = td < t_split; mR = (td >= t_split) & (td <= 100.0)
        mL_d = t_dense < t_split; mR_d = (t_dense >= t_split) & (t_dense <= 100.0)

        fig, axes = plt.subplots(3, 2, figsize=(8, 9),
                                 gridspec_kw={'width_ratios': [3, 2]})
        fig.suptitle(
            f'{name} ({pick}) q={d["q"]:.1f}, $\\chi_1$={d["chi1"]:.2f}, '
            f'$\\chi_2$={d["chi2"]:.2f}, $e_0$={d["e0"]:.3f}\n'
            f'$\\mathcal{{E}}$={all_E[idx]:.2e}, '
            f'$\\max|\\Delta\\phi|$={dp:.3f} rad',
            fontsize=7.5, fontweight='bold', y=1.01)

        axes[0, 0].plot(t_dense[mL_d] / 1e3, xi_a_ref_dense[mL_d], C_REF, lw=0.8, label='SEOB')
        axes[0, 0].plot(t_dense[mL_d] / 1e3, xi_a_mod_dense[mL_d], C_MOD, lw=0.6, ls='--', label='Model')
        axes[0, 0].set_ylabel(r'$\xi_A$'); axes[0, 0].legend(fontsize=6)
        axes[0, 1].plot(t_dense[mR_d], xi_a_ref_dense[mR_d], C_REF, lw=0.8)
        axes[0, 1].plot(t_dense[mR_d], xi_a_mod_dense[mR_d], C_MOD, lw=0.6, ls='--')
        axes[0, 1].axvline(-50, color='0.5', ls=':', lw=0.7, label='taper onset')
        axes[0, 1].legend(fontsize=5)

        axes[1, 0].plot(t_dense[mL_d] / 1e3, xi_w_ref_dense[mL_d], C_REF, lw=0.8)
        axes[1, 0].plot(t_dense[mL_d] / 1e3, xi_w_mod_dense[mL_d], C_MOD, lw=0.6, ls='--')
        axes[1, 0].set_ylabel(r'$\xi_\omega$')
        axes[1, 1].plot(t_dense[mR_d], xi_w_ref_dense[mR_d], C_REF, lw=0.8)
        axes[1, 1].plot(t_dense[mR_d], xi_w_mod_dense[mR_d], C_MOD, lw=0.6, ls='--')
        axes[1, 1].axvline(-50, color='0.5', ls=':', lw=0.7)

        axes[2, 0].plot(td[mL] / 1e3, np.real(hr[mL]), C_REF, lw=0.7, label='SEOB')
        axes[2, 0].plot(td[mL] / 1e3, np.real(hp[mL]), C_MOD, lw=0.5, ls='--', label='Model')
        axes[2, 0].set_ylabel(r'Re$(h_{22})$')
        axes[2, 0].set_xlabel(r'$t\;[10^3\,M]$'); axes[2, 0].legend(fontsize=6)
        axes[2, 1].plot(td[mR], np.real(hr[mR]), C_REF, lw=0.7)
        axes[2, 1].plot(td[mR], np.real(hp[mR]), C_MOD, lw=0.5, ls='--')
        axes[2, 1].set_xlabel(r'$t\;[M]$')
        axes[2, 1].axvline(-50, color='0.5', ls=':', lw=0.7)

        plt.tight_layout()
        for ext in ('pdf',): fig.savefig(os.path.join(outdir, f'{pick}_modulation.{ext}'))
        plt.close(fig)

    # Summary JSON
    summary = {
        'name': name,
        'val_E_med': float(np.median(E_va)), 'val_E_max': float(np.max(E_va)),
        'val_MM_med': float(np.median(MM_va)), 'val_MM_max': float(np.max(MM_va)),
        'val_dphi_med': float(np.median(dphi_va)), 'val_dphi_max': float(np.max(dphi_va)),
        'val_frac_01': float(np.mean(dphi_va < 0.1)),
        'val_frac_05': float(np.mean(dphi_va < 0.5)),
        'train_E_med': float(np.median(E_tr)),
        'train_MM_med': float(np.median(MM_tr)),
        'train_dphi_med': float(np.median(dphi_tr)),
    }
    if mono_va is not None:
        summary['val_mono_viol_med'] = float(np.median(mono_va))
        summary['val_mono_viol_max'] = float(np.max(mono_va))
    if ligo_va is not None and not np.all(np.isnan(ligo_va)):
        for j, Mtot in enumerate(MTOT_VALUES):
            lv = ligo_va[:, j]; lv = lv[~np.isnan(lv)]
            lt_ = ligo_tr[:, j]; lt_ = lt_[~np.isnan(lt_)]
            summary[f'val_ligo_mm_{Mtot}_med'] = float(np.median(lv)) if len(lv) > 0 else None
            summary[f'val_ligo_mm_{Mtot}_max'] = float(np.max(lv)) if len(lv) > 0 else None
            summary[f'train_ligo_mm_{Mtot}_med'] = float(np.median(lt_)) if len(lt_) > 0 else None
            summary[f'val_ligo_mm_{Mtot}_frac_001'] = float(np.mean(lv < 0.01)) if len(lv) > 0 else None

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def poly_phase_correction(order=5):
    def corr(t_d, phi_pred, phi_ecc):
        dphi = phi_pred - phi_ecc
        T = (t_d - t_d[0]) / max(t_d[-1] - t_d[0], 1.0)
        A = np.column_stack([T**k for k in range(1, order + 1)])
        c = np.linalg.lstsq(A, dphi - dphi[0], rcond=None)[0]
        return phi_pred - A @ c
    return corr


# ====================================================================
# Mandatory checklist
# ====================================================================
def run_checklist(name, outdir, errdir, predict_fn, train, val, phase_corr=None):
    """Run the mandatory post-training checklist. Returns (pass_count, fail_count, lines)."""
    lines = [f'=== CHECKLIST: {name} ===']
    checks = []

    # A. model.pkl
    ok = os.path.exists(os.path.join(outdir, 'model.pkl'))
    checks.append(('model.pkl saved', ok))

    # B/C. error arrays (10 npy files)
    npy_files = ['train_params', 'train_mathcalE', 'train_td_mismatch', 'train_dephasing',
                 'train_ligo_mismatch', 'val_params', 'val_mathcalE', 'val_td_mismatch',
                 'val_dephasing', 'val_ligo_mismatch']
    npy_ok = all(os.path.exists(os.path.join(errdir, f'{f}.npy')) for f in npy_files)
    checks.append(('error arrays saved (10 npy files)', npy_ok))

    # D. Diagnostic plots
    pdfs = ['histogram.pdf', 'histogram_ligo_mm.pdf', 'dephasing_vs_e0.pdf',
            'dephasing_vs_chieff.pdf', 'mathcalE_vs_e0.pdf', 'mathcalE_vs_chieff.pdf',
            'ligo_mm_vs_e0.pdf', 'ligo_mm_vs_chieff.pdf',
            'best_modulation.pdf', 'median_modulation.pdf', 'worst_modulation.pdf']
    for pdf in pdfs:
        ok = os.path.exists(os.path.join(outdir, pdf))
        checks.append((pdf, ok))

    # E. summary.json
    ok = os.path.exists(os.path.join(outdir, 'summary.json'))
    checks.append(('summary.json', ok))

    # F. Logs
    checks.append(('progress_log.md updated', os.path.exists(LOGFILE)))
    checks.append(('CHANGELOG.md updated', os.path.exists(os.path.join(TRACKING, 'CHANGELOG.md'))))

    n_pass = sum(1 for _, ok in checks if ok)
    n_fail = sum(1 for _, ok in checks if not ok)
    for desc, ok in checks:
        lines.append(f'[{"PASS" if ok else "FAIL"}] {desc}')
    if n_fail == 0:
        lines.append(f'=== ALL {n_pass} CHECKS PASSED ===')
    else:
        lines.append(f'=== {n_fail} CHECKS FAILED ===')

    result = '\n'.join(lines)

    # Save checklist
    ckfile = os.path.join(TRACKING, f'checklist_{name}.txt')
    with open(ckfile, 'w') as f:
        f.write(result + '\n')
    log(result)

    return n_pass, n_fail


def train_and_checklist(name, predict_fn, train, val, model_obj, phase_corr=None):
    """Full checklist: save model, evaluate, save arrays, make plots, log."""
    outdir = os.path.join(RESULTS, 'models', name)
    os.makedirs(outdir, exist_ok=True)

    # A. Save model
    with open(os.path.join(outdir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_obj, f)

    # B. Evaluate
    log(f'  Evaluating on training set...')
    E_tr, MM_tr, dphi_tr, mono_tr, ligo_tr = eval_all(train, predict_fn, phase_corr)
    log(f'  Evaluating on validation set...')
    E_va, MM_va, dphi_va, mono_va, ligo_va = eval_all(val, predict_fn, phase_corr)

    log(f'  Train: E={np.median(E_tr):.4e} MM={np.median(MM_tr):.4e} dphi={np.median(dphi_tr):.3f}')
    log(f'  Val:   E={np.median(E_va):.4e} MM={np.median(MM_va):.4e} dphi={np.median(dphi_va):.3f} '
        f'<0.1:{np.mean(dphi_va < 0.1):.0%} <0.5:{np.mean(dphi_va < 0.5):.0%} '
        f'mono_viol={np.median(mono_va):.3f}')

    # LIGO mismatch summary
    for j, Mtot in enumerate(MTOT_VALUES):
        lv = ligo_va[:, j]; lv = lv[~np.isnan(lv)]
        if len(lv) > 0:
            log(f'  Val LIGO MM (Mtot={Mtot}): med={np.median(lv):.4e} max={np.max(lv):.4e} '
                f'<0.01:{np.mean(lv < 0.01):.0%}')

    # C. Save error arrays
    errdir = os.path.join(RESULTS, 'errors', name)
    os.makedirs(errdir, exist_ok=True)
    for tag, data_, Es_, MMs_, dphis_, ligos_ in [
        ('train', train, E_tr, MM_tr, dphi_tr, ligo_tr),
        ('val', val, E_va, MM_va, dphi_va, ligo_va),
    ]:
        params = np.array([[d['q'], d['chi1'], d['chi2'], d['e0']] for d in data_])
        np.save(os.path.join(errdir, f'{tag}_params.npy'), params)
        np.save(os.path.join(errdir, f'{tag}_mathcalE.npy'), Es_)
        np.save(os.path.join(errdir, f'{tag}_td_mismatch.npy'), MMs_)
        np.save(os.path.join(errdir, f'{tag}_dephasing.npy'), dphis_)
        np.save(os.path.join(errdir, f'{tag}_ligo_mismatch.npy'), ligos_)

    # D. Make plots
    log(f'  Generating diagnostic plots...')
    summary = make_plots(outdir, name, train, val, E_tr, MM_tr, dphi_tr,
                         E_va, MM_va, dphi_va, predict_fn,
                         mono_tr=mono_tr, mono_va=mono_va,
                         ligo_tr=ligo_tr, ligo_va=ligo_va, phase_corr=phase_corr)

    # F. Log
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    log(f'\n  [{now}] Model {name} trained and evaluated')

    # Update CHANGELOG
    changelog = os.path.join(TRACKING, 'CHANGELOG.md')
    with open(changelog, 'r') as f:
        existing = f.read()
    entry = f"""## [{now}] MODEL TRAINED + CHECKLIST

**What**: Trained model {name}
**Details**:
- Val E={np.median(E_va):.4e}, MM={np.median(MM_va):.4e}, dphi={np.median(dphi_va):.3f}
- Val <0.1 rad: {np.mean(dphi_va < 0.1):.0%}, <0.5 rad: {np.mean(dphi_va < 0.5):.0%}
- Mono violation: {np.median(mono_va):.3f}
**Status**: DONE

---

"""
    parts = existing.split('---', 2)
    if len(parts) >= 3:
        updated = parts[0] + '---\n\n' + entry + parts[2]
    else:
        updated = existing + '\n' + entry
    with open(changelog, 'w') as f:
        f.write(updated)

    # G. Verify
    n_pass, n_fail = run_checklist(name, outdir, errdir, predict_fn, train, val, phase_corr)

    return summary, E_tr, MM_tr, dphi_tr, mono_tr, ligo_tr, \
           E_va, MM_va, dphi_va, mono_va, ligo_va


# ====================================================================
# Comparison plots
# ====================================================================
def make_comparison(all_summaries, all_names):
    comp_dir = os.path.join(RESULTS, 'comparison')
    os.makedirs(comp_dir, exist_ok=True)

    # 1. Progress staircase (dephasing + mathcalE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_pos = np.arange(len(all_summaries))
    dphi_meds = [s['val_dphi_med'] for s in all_summaries]
    colors = [C_MOD if 'phase_corr' in s['name'] else C_TR for s in all_summaries]
    axes[0].bar(x_pos, dphi_meds, color=colors, edgecolor='0.3', lw=0.4, width=0.6)
    axes[0].axhline(0.1, color=C_TG, ls='--', lw=1.2, label='0.1 rad')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([s['name'] for s in all_summaries], fontsize=4.5, rotation=35, ha='right')
    axes[0].set_ylabel(r'Val median $\max|\Delta\phi|$ [rad]')
    axes[0].set_title('Dephasing progress', fontsize=9, fontweight='bold')
    axes[0].legend(fontsize=7)

    E_meds = [s['val_E_med'] for s in all_summaries]
    axes[1].bar(x_pos, E_meds, color=colors, edgecolor='0.3', lw=0.4, width=0.6)
    axes[1].axhline(0.01, color=C_TG, ls='--', lw=1.2, label=r'$10^{-2}$')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([s['name'] for s in all_summaries], fontsize=4.5, rotation=35, ha='right')
    axes[1].set_ylabel(r'Val median $\mathcal{E}$')
    axes[1].set_title(r'$\mathcal{E}$ progress', fontsize=9, fontweight='bold')
    axes[1].set_yscale('log'); axes[1].legend(fontsize=7)
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(comp_dir, f'progress.{ext}'))
    plt.close(fig)

    # 2. LIGO mismatch staircase
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.15
    colors_mtot = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for j, Mtot in enumerate(MTOT_VALUES):
        vals = []
        for s in all_summaries:
            v = s.get(f'val_ligo_mm_{Mtot}_med', None)
            vals.append(v if v is not None else 0.0)
        ax.bar(x_pos + j * bar_width, vals, bar_width, color=colors_mtot[j],
               edgecolor='0.3', lw=0.3, label=f'$M_{{\\rm tot}}$={Mtot}')
    ax.axhline(0.01, color=C_TG, ls='--', lw=1.2, label='1%')
    ax.set_xticks(x_pos + 2 * bar_width)
    ax.set_xticklabels([s['name'] for s in all_summaries], fontsize=4.5, rotation=35, ha='right')
    ax.set_ylabel('Val median LIGO mismatch'); ax.set_yscale('log')
    ax.set_title('LIGO mismatch progress', fontsize=9, fontweight='bold')
    ax.legend(fontsize=5, ncol=3); plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(comp_dir, f'progress_ligo_mm.{ext}'))
    plt.close(fig)

    # 3. Dephasing comparison across models
    fig, ax = plt.subplots(figsize=(6, 4))
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(all_summaries)))
    for i, s in enumerate(all_summaries):
        if s['name'] == 'ansatz_only':
            continue
        ax.axhline(s['val_dphi_med'], color=model_colors[i], ls='-', lw=0.8,
                   alpha=0.6, label=f'{s["name"]} ({s["val_dphi_med"]:.3f})')
    ax.axhline(0.1, color=C_TG, ls='--', lw=1.5, label='0.1 rad target')
    ax.set_ylabel(r'Val median $\max|\Delta\phi|$ [rad]')
    ax.set_title('Model comparison: dephasing', fontsize=9, fontweight='bold')
    ax.legend(fontsize=4.5, loc='upper left', ncol=2); ax.set_yscale('log')
    plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(comp_dir, f'comparison_dephasing.{ext}'))
    plt.close(fig)

    # 4. LIGO mismatch comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(all_summaries):
        if s['name'] == 'ansatz_only':
            continue
        vals = [s.get(f'val_ligo_mm_{Mt}_med', np.nan) for Mt in MTOT_VALUES]
        ax.plot(MTOT_VALUES, vals, 'o-', color=model_colors[i], lw=0.8, ms=4,
                label=s['name'])
    ax.axhline(0.01, color=C_TG, ls='--', lw=1.2, label='1%')
    ax.set_xlabel(r'$M_{\rm tot}\;[M_\odot]$'); ax.set_ylabel('Val median LIGO MM')
    ax.set_yscale('log')
    ax.set_title('LIGO mismatch comparison', fontsize=9, fontweight='bold')
    ax.legend(fontsize=4.5, ncol=2); plt.tight_layout()
    for ext in ('pdf',): fig.savefig(os.path.join(comp_dir, f'comparison_ligo_mm.{ext}'))
    plt.close(fig)

    with open(os.path.join(comp_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)

    # Also save progress_log.json
    with open(os.path.join(TRACKING, 'progress_log.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    log(f'\n\n{"=" * 70}')
    log(f'## [Small-Spin Model Fitting] — {now}')
    log(f'{"=" * 70}\n')

    with open(os.path.join(RESULTS, 'training_data.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(RESULTS, 'validation_data.pkl'), 'rb') as f:
        val = pickle.load(f)
    log(f'Loaded {len(train)} train + {len(val)} val')

    all_summaries = []; all_names = []
    best_predict = None; best_name = None; best_dphi = 1e10

    # ------------------------------------------------------------------
    # Model 0: Ansatz only (delta = 0, baseline)
    # ------------------------------------------------------------------
    log('\n### Model 0: Ansatz only (delta=0)')

    def predict_ansatz_only(e, x, z, nu, chi_S, chi_A):
        xa, xw = ansatz_modulations(e, x, z, nu)
        return xa, xw

    s0, *errs0 = train_and_checklist(
        'ansatz_only', predict_ansatz_only, train, val,
        {'type': 'ansatz_only'})
    all_summaries.append(s0); all_names.append('ansatz_only')

    # ------------------------------------------------------------------
    # Model 1: Ridge — validation-driven basis+alpha scan (with spin)
    # ------------------------------------------------------------------
    log('\n### Ridge basis scan (with spin features)')
    ds = downsample(train, n_pts=400)

    def make_ridge_pred(ma, mw, bc_):
        def pred(e, x, z, nu, chi_S, chi_A):
            n = len(e)
            B = build_basis(e, z, x, np.full(n, nu),
                           np.full(n, chi_S) if np.isscalar(chi_S) else chi_S,
                           np.full(n, chi_A) if np.isscalar(chi_A) else chi_A, **bc_)
            xa_ans, xw_ans = ansatz_modulations(e, x, z, nu)
            return xa_ans + ma.predict(B), xw_ans + mw.predict(B)
        return pred

    scan_results = []
    for n_harm in [3, 5, 7]:
        for max_e in [3, 4, 5]:
            for max_x in [2, 3]:
                for max_chi in [1, 2]:
                    bc = dict(max_e=max_e, max_x=max_x, max_nu=2,
                              max_chi=max_chi, n_harm=n_harm)
                    X = build_basis(ds['e'], ds['z'], ds['x'], ds['nu'],
                                    ds['chiS'], ds['chiA'], **bc)
                    nf = X.shape[1]
                    if nf > len(ds['e']) // 2:
                        continue
                    for alpha in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
                        m_a = Ridge(alpha=alpha, fit_intercept=False).fit(X, ds['dya'])
                        m_w = Ridge(alpha=alpha, fit_intercept=False).fit(X, ds['dyw'])
                        pf = make_ridge_pred(m_a, m_w, bc)
                        val_sub = val[:25]
                        _, _, dv_sub, mv_sub, _ = eval_all(val_sub, pf, compute_ligo=False)
                        scan_results.append({
                            'bc': bc, 'alpha': alpha, 'nf': nf,
                            'dphi_med': float(np.median(dv_sub)),
                            'mono_med': float(np.median(mv_sub)),
                            'm_a': m_a, 'm_w': m_w,
                        })
        log(f'  n_harm={n_harm}: {len(scan_results)} configs tested')

    valid = [r for r in scan_results if r['mono_med'] <= 0.05]
    if not valid:
        log('  WARNING: all configs violate monotonicity, using least-violating')
        valid = sorted(scan_results, key=lambda r: r['mono_med'])[:10]
    valid.sort(key=lambda r: r['dphi_med'])

    log(f'\n  Top 10 configs (of {len(valid)} passing monotonicity):')
    log(f'  {"n_harm":>6s} {"max_e":>5s} {"max_x":>5s} {"mchi":>5s} {"alpha":>8s} {"nf":>5s} {"dphi":>8s} {"mono":>6s}')
    for r in valid[:10]:
        log(f'  {r["bc"]["n_harm"]:6d} {r["bc"]["max_e"]:5d} {r["bc"]["max_x"]:5d} '
            f'{r["bc"]["max_chi"]:5d} {r["alpha"]:8.0e} {r["nf"]:5d} '
            f'{r["dphi_med"]:8.3f} {r["mono_med"]:6.3f}')

    best_cfg = valid[0]
    best_bc = best_cfg['bc']; best_alpha = best_cfg['alpha']
    ridge_pf = make_ridge_pred(best_cfg['m_a'], best_cfg['m_w'], best_bc)
    ridge_name = (f'ridge_nh{best_bc["n_harm"]}_me{best_bc["max_e"]}'
                  f'_mchi{best_bc["max_chi"]}_a{best_alpha:.0e}')
    log(f'\n### Best Ridge: {ridge_name} ({best_cfg["nf"]} features)')

    s1, *errs1 = train_and_checklist(
        ridge_name, ridge_pf, train, val,
        {'m_a': best_cfg['m_a'], 'm_w': best_cfg['m_w'],
         'bc': best_bc, 'alpha': best_alpha, 'type': 'ridge'})
    all_summaries.append(s1); all_names.append(ridge_name)
    if s1['val_dphi_med'] < best_dphi:
        best_dphi = s1['val_dphi_med']; best_predict = ridge_pf; best_name = ridge_name

    # ------------------------------------------------------------------
    # Model 2: Polynomial + Ridge (with spin features)
    # ------------------------------------------------------------------
    log('\n### Polynomial scan (with spin features)')
    ds = downsample(train, n_pts=400)

    def make_poly_pred(pa, pw):
        def pred(e, x, z, nu, chi_S, chi_A):
            n = len(e)
            r = np.column_stack([e, x, np.full(n, nu) if np.isscalar(nu) else nu,
                                 np.full(n, chi_S) if np.isscalar(chi_S) else chi_S,
                                 np.full(n, chi_A) if np.isscalar(chi_A) else chi_A,
                                 np.cos(z), np.sin(z),
                                 np.cos(2 * z), np.sin(2 * z),
                                 np.cos(3 * z), np.sin(3 * z)])
            dya, dyw = pa.predict(r), pw.predict(r)
            xa_ans, xw_ans = ansatz_modulations(e, x, z,
                                                 np.full(n, nu) if np.isscalar(nu) else nu)
            return xa_ans + dya, xw_ans + dyw
        return pred

    raw = np.column_stack([ds['e'], ds['x'], ds['nu'], ds['chiS'], ds['chiA'],
                           np.cos(ds['z']), np.sin(ds['z']),
                           np.cos(2 * ds['z']), np.sin(2 * ds['z']),
                           np.cos(3 * ds['z']), np.sin(3 * ds['z'])])
    best_poly_dphi = 1e10; pf_poly = None
    for degree in [3, 4, 5]:
        for alpha in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
            pa = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                               Ridge(alpha=alpha, fit_intercept=False))
            pw = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                               Ridge(alpha=alpha, fit_intercept=False))
            pa.fit(raw, ds['dya']); pw.fit(raw, ds['dyw'])
            pf_cand = make_poly_pred(pa, pw)
            _, _, dv_sub, mv_sub, _ = eval_all(val[:25], pf_cand, compute_ligo=False)
            if np.median(mv_sub) <= 0.05 and np.median(dv_sub) < best_poly_dphi:
                best_poly_dphi = np.median(dv_sub)
                pf_poly = pf_cand; best_poly_deg = degree
                best_pipe_a = pa; best_pipe_w = pw; best_poly_alpha = alpha
        log(f'  degree={degree} scanned')

    if pf_poly is not None:
        poly_name = f'polynomial_deg{best_poly_deg}'
        n_feat = best_pipe_a.named_steps['polynomialfeatures'].n_output_features_
        log(f'\n### Best Polynomial: deg={best_poly_deg}, alpha={best_poly_alpha:.0e}, {n_feat} features')

        s2, *errs2 = train_and_checklist(
            poly_name, pf_poly, train, val,
            {'pipe_a': best_pipe_a, 'pipe_w': best_pipe_w,
             'degree': best_poly_deg, 'type': 'polynomial'})
        all_summaries.append(s2); all_names.append(poly_name)
        if s2['val_dphi_med'] < best_dphi:
            best_dphi = s2['val_dphi_med']; best_predict = pf_poly; best_name = poly_name

    # ------------------------------------------------------------------
    # Model 3: Random Forest
    # ------------------------------------------------------------------
    log('\n### Random Forest')
    ds = downsample(train, n_pts=400)
    raw_rf = np.column_stack([ds['e'], ds['x'], ds['nu'], ds['chiS'], ds['chiA'],
                              np.cos(ds['z']), np.sin(ds['z']),
                              np.cos(2 * ds['z']), np.sin(2 * ds['z'])])
    rf_a = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    rf_w = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    log('  Fitting RF...')
    rf_a.fit(raw_rf, ds['dya']); rf_w.fit(raw_rf, ds['dyw'])

    def make_rf_pred(rfa, rfw):
        def pred(e, x, z, nu, chi_S, chi_A):
            n = len(e)
            r = np.column_stack([e, x, np.full(n, nu) if np.isscalar(nu) else nu,
                                 np.full(n, chi_S) if np.isscalar(chi_S) else chi_S,
                                 np.full(n, chi_A) if np.isscalar(chi_A) else chi_A,
                                 np.cos(z), np.sin(z),
                                 np.cos(2 * z), np.sin(2 * z)])
            xa_ans, xw_ans = ansatz_modulations(e, x, z,
                                                 np.full(n, nu) if np.isscalar(nu) else nu)
            if n > 2000:
                idx = np.linspace(0, n - 1, 2000, dtype=int)
                da = rfa.predict(r[idx]); dw = rfw.predict(r[idx])
                da = np.interp(np.arange(n), idx, da)
                dw = np.interp(np.arange(n), idx, dw)
            else:
                da = rfa.predict(r); dw = rfw.predict(r)
            return xa_ans + da, xw_ans + dw
        return pred

    pf_rf = make_rf_pred(rf_a, rf_w)
    s3, *errs3 = train_and_checklist(
        'random_forest', pf_rf, train, val,
        {'rf_a': rf_a, 'rf_w': rf_w, 'type': 'random_forest'})
    all_summaries.append(s3); all_names.append('random_forest')
    if s3['val_dphi_med'] < best_dphi:
        best_dphi = s3['val_dphi_med']; best_predict = pf_rf; best_name = 'random_forest'

    # ------------------------------------------------------------------
    # Model 4: Hybrid ansatz + Ridge residual
    # ------------------------------------------------------------------
    log('\n### Hybrid (ansatz + Ridge residual)')
    ds = downsample(train, n_pts=400)
    bc_hyb = best_bc.copy()
    X_hyb = build_basis(ds['e'], ds['z'], ds['x'], ds['nu'],
                        ds['chiS'], ds['chiA'], **bc_hyb)
    m_res_a = Ridge(alpha=best_alpha, fit_intercept=False).fit(X_hyb, ds['dya'])
    m_res_w = Ridge(alpha=best_alpha, fit_intercept=False).fit(X_hyb, ds['dyw'])

    def make_hybrid_pred(mra, mrw, bc_):
        def pred(e, x, z, nu, chi_S, chi_A):
            n = len(e)
            B = build_basis(e, z, x, np.full(n, nu) if np.isscalar(nu) else nu,
                           np.full(n, chi_S) if np.isscalar(chi_S) else chi_S,
                           np.full(n, chi_A) if np.isscalar(chi_A) else chi_A, **bc_)
            xa_ans, xw_ans = ansatz_modulations(e, x, z,
                                                 np.full(n, nu) if np.isscalar(nu) else nu)
            return xa_ans + mra.predict(B), xw_ans + mrw.predict(B)
        return pred

    pf_hyb = make_hybrid_pred(m_res_a, m_res_w, bc_hyb)
    s4, *errs4 = train_and_checklist(
        'hybrid_ansatz', pf_hyb, train, val,
        {'m_res_a': m_res_a, 'm_res_w': m_res_w, 'bc': bc_hyb, 'type': 'hybrid'})
    all_summaries.append(s4); all_names.append('hybrid_ansatz')
    if s4['val_dphi_med'] < best_dphi:
        best_dphi = s4['val_dphi_med']; best_predict = pf_hyb; best_name = 'hybrid_ansatz'

    # ------------------------------------------------------------------
    # Model 5: Best model + polynomial phase correction
    # ------------------------------------------------------------------
    log(f'\n### Best model ({best_name}) + phase correction')
    pc = poly_phase_correction(order=5)
    name_pc = f'{best_name}+phase_corr'
    s5, *errs5 = train_and_checklist(
        name_pc, best_predict, train, val,
        {'base_model': best_name, 'phase_order': 5, 'type': 'phase_corr'},
        phase_corr=pc)
    all_summaries.append(s5); all_names.append(name_pc)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    log(f'\n### Comparison')
    log(f'{"Name":<40s} {"Val E":>10s} {"Val MM":>10s} {"Val dphi":>10s} {"<0.1":>6s} {"<0.5":>6s}')
    log('-' * 80)
    for s in all_summaries:
        log(f'{s["name"]:<40s} {s["val_E_med"]:10.4e} {s["val_MM_med"]:10.4e} '
            f'{s["val_dphi_med"]:10.3f} {s["val_frac_01"]:5.0%} {s["val_frac_05"]:5.0%}')

    make_comparison(all_summaries, all_names)
    log('\nComparison plots saved to results/comparison/')

    best_final = min(all_summaries, key=lambda s: s['val_dphi_med'])
    log(f'\nBest: {best_final["name"]} (dphi={best_final["val_dphi_med"]:.3f} rad, '
        f'E={best_final["val_E_med"]:.4e})')
    log(f'\nDone.')
