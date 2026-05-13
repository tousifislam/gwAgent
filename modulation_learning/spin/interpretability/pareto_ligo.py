"""
Pareto plot: complexity (number of terms) vs LIGO mismatch
for the best Ridge model at various skeleton truncation levels.

Uses the importance-ranked skeleton approach from compactify.py:
  N = [10, 20, 30, 50, 75, 100, 150, 200, 300, 1379(k<=3), 2955(full)]

For each N, refits Ridge on the top-N features, reconstructs all 150
validation waveforms, and computes LIGO mismatch at Mtot=65 Msun.
"""
import os, sys, pickle, time, warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 10,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.7,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.abspath(__file__))
SPIN_DIR = os.path.dirname(BASE)
RESULTS = os.path.join(SPIN_DIR, 'results')
OUTDIR = BASE

MTOT_VALUES = [20, 65, 110, 155, 200]

# ====================================================================
# Load model and data
# ====================================================================
print("Loading model and data...")
model_path = os.path.join(RESULTS, 'models', 'ridge_nh7_me5_mchi1_a1e-06', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

bc = model['bc']
max_e, max_x, max_nu, max_chi, n_harm = (
    bc['max_e'], bc['max_x'], bc['max_nu'], bc['max_chi'], bc['n_harm'])
coef_a_full = model['m_a'].coef_.copy()
coef_w_full = model['m_w'].coef_.copy()

with open(os.path.join(RESULTS, 'validation_data.pkl'), 'rb') as f:
    val_data = pickle.load(f)
with open(os.path.join(RESULTS, 'training_data.pkl'), 'rb') as f:
    train_data = pickle.load(f)

print(f"  {len(train_data)} train + {len(val_data)} val waveforms")

# ====================================================================
# Ansatz + reconstruction (same as fit.py)
# ====================================================================
def h22_ecc_ansatz(x, e, zeta, nu):
    e2 = e*e; e3 = e2*e
    eiz = np.exp(1j*zeta); emiz = np.exp(-1j*zeta)
    leading = (4.0 + 2.0*e2*eiz**2 + e*emiz + 5.0*e*eiz) / (4.0*(1.0-e2))
    tc = e*(26.0*nu/7.0 - 559.0/84.0)
    tem2 = e*np.exp(-2j*zeta)*(15.0*nu/14.0 - 95.0/168.0)
    tem3 = e2*np.exp(-3j*zeta)*(9.0*nu/56.0 + 1.0/112.0)
    te3 = e2*np.exp(3j*zeta)*(nu/8.0 - 49.0/48.0)
    te2 = np.exp(2j*zeta)*(e3*(6.0*nu/7.0-41.0/21.0)+e*(nu/14.0-153.0/56.0))
    tem = emiz*(e2*(7.0*nu/8.0-59.0/48.0)+27.0*nu/14.0-23.0/14.0)
    tei = eiz*(e2*(143.0*nu/56.0-2071.0/336.0)+nu/14.0-13.0/7.0)
    curly = tc+tem3+te3+tem2+te2+tem+tei
    return leading + (x*e)/(1.0-e2)**2 * curly

def ansatz_modulations(e, x, zeta, nu):
    return np.abs(h22_ecc_ansatz(x, e, zeta, nu)) - 1.0, \
           (np.abs(h22_ecc_ansatz(x, e, zeta, nu)) - 1.0) / 0.9

def smooth_taper(t, ts=-50.0, te=0.0):
    w = np.ones_like(t)
    m = (t >= ts) & (t <= te)
    w[m] = 0.5*(1+np.cos(np.pi*(t[m]-ts)/(te-ts)))
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
    delta_f = 1.0 / hp_ref.duration; flen = tlen // 2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
    try:
        m, _ = match(hp_ref, hp_pred, psd=psd, low_frequency_cutoff=f_low)
        return 1.0 - m
    except Exception:
        return np.nan

def reconstruct_and_mismatch(d, xi_a, xi_w):
    """Reconstruct waveform and return LIGO mismatches at all Mtot."""
    dt = 0.1
    t_d = np.arange(d['t'][0], d['t'][-1], dt)
    h_ecc_d = CubicSpline(d['t'], np.real(d['h_ecc']))(t_d) + \
              1j*CubicSpline(d['t'], np.imag(d['h_ecc']))(t_d)
    h_cir_d = CubicSpline(d['t'], np.real(d['h_cir']))(t_d) + \
              1j*CubicSpline(d['t'], np.imag(d['h_cir']))(t_d)
    xi_a_d = np.interp(t_d, d['t'], xi_a)
    xi_w_d = np.interp(t_d, d['t'], xi_w)
    taper = smooth_taper(t_d)
    xi_a_d *= taper; xi_w_d *= taper
    A_p = np.abs(h_cir_d)*(1.0+xi_a_d)
    phi_cir = np.unwrap(np.angle(h_cir_d))
    omega_cir = np.gradient(phi_cir, dt)
    pp = cumulative_trapezoid(omega_cir*(1.0+xi_w_d), dx=dt, initial=0)
    pe = np.unwrap(np.angle(h_ecc_d))
    pp += pe[0] - pp[0]
    hp = A_p * np.exp(1j*pp)
    hp *= np.exp(-1j*np.angle(hp[0]))
    ha = h_ecc_d * np.exp(-1j*np.angle(h_ecc_d[0]))
    lms = [ligo_mismatch(hp, ha, 0.1, Mt) for Mt in MTOT_VALUES]
    return np.array(lms)

# ====================================================================
# Build basis and feature importance ranking
# ====================================================================
print("Building basis and ranking features...")

groups = []
idx = 0
for a in range(1, max_e+1):
    for b in range(max_x+1):
        for c in range(max_nu+1):
            for d_s in range(max_chi+1):
                for d_a in range(max_chi+1):
                    if a+b+c+d_s+d_a > max_e+3:
                        continue
                    groups.append({'a':a,'b':b,'c':c,'ds':d_s,'da':d_a,
                                   'start':idx,'n_feat':1+2*n_harm})
                    idx += 1+2*n_harm

K_MAX = 3

def build_full_basis(e, z, x, nu, chiS, chiA):
    features = []
    for g in groups:
        base = e**g['a'] * x**g['b'] * nu**g['c'] * chiS**g['ds'] * chiA**g['da']
        features.append(base)
        for k in range(1, n_harm+1):
            features.append(base*np.cos(k*z))
            features.append(base*np.sin(k*z))
    return np.column_stack(features)

def downsample(data, n_pts=400):
    out = {k: [] for k in ['e','x','z','nu','chiS','chiA','dya','dyw']}
    for d in data:
        n = min(len(d['delta_xi_amp']), len(d['delta_xi_omega']))
        e_ = np.clip(d['e'][:n], 1e-6, 0.95)
        x_ = np.clip(d['x'][:n], 1e-6, 0.5)
        z_ = d['zeta'][:n]
        mask = d['t'][:n] <= 50
        idx = np.where(mask)[0]
        if len(idx) > n_pts:
            idx = idx[np.linspace(0, len(idx)-1, n_pts, dtype=int)]
        if len(idx) < 10:
            continue
        out['e'].append(e_[idx]); out['x'].append(x_[idx])
        out['z'].append(z_[idx]); out['nu'].append(np.full(len(idx), d['nu']))
        out['chiS'].append(np.full(len(idx), d['chi_S']))
        out['chiA'].append(np.full(len(idx), d['chi_A']))
        out['dya'].append(d['delta_xi_amp'][idx])
        out['dyw'].append(d['delta_xi_omega'][idx])
    return {k: np.concatenate(v) for k, v in out.items()}

td = downsample(train_data, n_pts=400)
X_train = build_full_basis(td['e'], td['z'], td['x'], td['nu'], td['chiS'], td['chiA'])

# k<=3 truncated coefficients
coef_a_k3 = coef_a_full.copy()
coef_w_k3 = coef_w_full.copy()
for g in groups:
    s = g['start']
    for k in range(K_MAX+1, n_harm+1):
        coef_a_k3[s+2*k-1] = 0; coef_a_k3[s+2*k] = 0
        coef_w_k3[s+2*k-1] = 0; coef_w_k3[s+2*k] = 0

# Feature importance ranking
feat_stds = np.std(X_train, axis=0)
importance = (np.abs(coef_a_k3) + np.abs(coef_w_k3)) * feat_stds

ranked_indices = np.argsort(importance)[::-1]
# Only include k<=3 features (non-zero in coef_a_k3)
ranked_indices = [i for i in ranked_indices if coef_a_k3[i] != 0 or coef_w_k3[i] != 0]

print(f"  {len(ranked_indices)} non-zero features in k<=3 model")

# ====================================================================
# Evaluate LIGO mismatch for each skeleton level
# ====================================================================
N_values = [10, 20, 30, 50, 75, 100, 150, 200, 300]

# Add the full model points
pareto_points = []

# For each N, refit and evaluate on validation
for N in N_values:
    top_idx = ranked_indices[:N]
    X_tr_sub = X_train[:, top_idx]

    ridge_a = Ridge(alpha=1e-6, fit_intercept=False)
    ridge_a.fit(X_tr_sub, td['dya'])
    ridge_w = Ridge(alpha=1e-6, fit_intercept=False)
    ridge_w.fit(X_tr_sub, td['dyw'])

    coef_a_skel = np.zeros_like(coef_a_full)
    coef_w_skel = np.zeros_like(coef_w_full)
    for j, fi in enumerate(top_idx):
        coef_a_skel[fi] = ridge_a.coef_[j]
        coef_w_skel[fi] = ridge_w.coef_[j]

    # Evaluate LIGO mismatch on validation
    all_ligo = []
    t0 = time.time()
    for d in val_data:
        n = min(len(d['delta_xi_amp']), len(d['delta_xi_omega']))
        e = np.clip(d['e'][:n], 1e-6, 0.95)
        x = np.clip(d['x'][:n], 1e-6, 0.5)
        z = d['zeta'][:n]
        X_wf = build_full_basis(e, z, x,
                                np.full(n, d['nu']),
                                np.full(n, d['chi_S']),
                                np.full(n, d['chi_A']))
        dxi_a = X_wf @ coef_a_skel
        dxi_w = X_wf @ coef_w_skel
        xi_amp_ans, xi_omg_ans = ansatz_modulations(e, x, z, d['nu'])
        xi_a = xi_amp_ans + dxi_a
        xi_w = xi_omg_ans + dxi_w
        lms = reconstruct_and_mismatch(d, xi_a, xi_w)
        all_ligo.append(lms)

    all_ligo = np.array(all_ligo)  # (150, 5)
    flat = all_ligo.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    med = np.median(flat)
    p90 = np.percentile(flat, 90)
    elapsed = time.time() - t0

    pareto_points.append({
        'N': N, 'median_ligo': med, 'p90_ligo': p90,
        'all_ligo': all_ligo, 'time': elapsed
    })
    print(f"  N={N:4d}: median_LIGO={med:.2e}, 90th={p90:.2e} ({elapsed:.1f}s)")

# Full model (2955 params) — load precomputed
full_ligo_path = os.path.join(RESULTS, 'errors',
                               'ridge_nh7_me5_mchi1_a1e-06+phase_corr',
                               'val_ligo_mismatch.npy')
if os.path.exists(full_ligo_path):
    full_ligo = np.load(full_ligo_path)
    flat = full_ligo.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    pareto_points.append({
        'N': 2955, 'median_ligo': np.median(flat),
        'p90_ligo': np.percentile(flat, 90),
        'all_ligo': full_ligo, 'time': 0
    })
    print(f"  N=2955: median_LIGO={np.median(flat):.2e} (precomputed, +phase_corr)")

# k<=3 model (1379 params) — evaluate
print("  Evaluating k<=3 model (1379 params)...")
all_ligo_k3 = []
for d in val_data:
    n = min(len(d['delta_xi_amp']), len(d['delta_xi_omega']))
    e = np.clip(d['e'][:n], 1e-6, 0.95)
    x = np.clip(d['x'][:n], 1e-6, 0.5)
    z = d['zeta'][:n]
    X_wf = build_full_basis(e, z, x,
                            np.full(n, d['nu']),
                            np.full(n, d['chi_S']),
                            np.full(n, d['chi_A']))
    dxi_a = X_wf @ coef_a_k3
    dxi_w = X_wf @ coef_w_k3
    xi_amp_ans, xi_omg_ans = ansatz_modulations(e, x, z, d['nu'])
    xi_a = xi_amp_ans + dxi_a
    xi_w = xi_omg_ans + dxi_w
    lms = reconstruct_and_mismatch(d, xi_a, xi_w)
    all_ligo_k3.append(lms)

all_ligo_k3 = np.array(all_ligo_k3)
flat_k3 = all_ligo_k3.flatten()
flat_k3 = flat_k3[np.isfinite(flat_k3) & (flat_k3 > 0)]
pareto_points.append({
    'N': 1379, 'median_ligo': np.median(flat_k3),
    'p90_ligo': np.percentile(flat_k3, 90),
    'all_ligo': all_ligo_k3, 'time': 0
})
print(f"  N=1379: median_LIGO={np.median(flat_k3):.2e}")

# Sort by N
pareto_points.sort(key=lambda p: p['N'])

# ====================================================================
# Plot
# ====================================================================
print("\nGenerating Pareto plot...")

fig, ax = plt.subplots(figsize=(6, 6))

Ns = [p['N'] for p in pareto_points]
meds = [p['median_ligo'] for p in pareto_points]
p90s = [p['p90_ligo'] for p in pareto_points]

# Median line
ax.plot(Ns, meds, 'o-', ms=7, lw=1.5, color='#0072B2', label='Median', zorder=5)

# 90th percentile line
ax.plot(Ns, p90s, 's--', ms=5, lw=1.0, color='#D55E00', alpha=0.7,
        label='90th percentile', zorder=4)

# Fill between
ax.fill_between(Ns, meds, p90s, alpha=0.08, color='#0072B2', zorder=2)

# Reference lines
ax.axhline(1e-2, color='0.5', ls='--', lw=0.8, zorder=1)
ax.axhline(1e-3, color='0.5', ls=':', lw=0.8, zorder=1)
ax.text(Ns[0]*0.85, 1e-2*1.3, '$10^{-2}$', fontsize=8, color='0.4', va='bottom')
ax.text(Ns[0]*0.85, 1e-3*1.3, '$10^{-3}$', fontsize=8, color='0.4', va='bottom')

# Highlight key points
highlights = {10: 'Skeleton-10', 50: 'Skeleton-50', 75: 'Skeleton-75',
              1379: '$k{\\leq}3$', 2955: 'Full+phase'}
for p in pareto_points:
    if p['N'] in highlights:
        label = highlights[p['N']]
        offset = (12, -8) if p['N'] < 500 else (-15, 12)
        ax.annotate(f"{label}\n({p['N']} terms)",
                    xy=(p['N'], p['median_ligo']),
                    xytext=offset, textcoords='offset points',
                    fontsize=7, color='0.3', ha='center',
                    arrowprops=dict(arrowstyle='-', color='0.6', lw=0.5)
                    if p['N'] in [75, 2955] else None)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of basis functions (complexity)')
ax.set_ylabel('LIGO mismatch (all $M_{\\rm tot}$ combined)')
ax.set_title('Pareto Front: Complexity vs LIGO Mismatch')
ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=False,
          edgecolor='0.7', framealpha=0.9)

ax.text(0.03, 0.03,
        'Validation set, 150 waveforms\n'
        '$M_{\\rm tot} \\in \\{20, 65, 110, 155, 200\\}\\,M_\\odot$',
        transform=ax.transAxes, fontsize=7, va='bottom', ha='left', color='0.4')

ax.set_xlim(7, 5000)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pareto_complexity_vs_ligo_mm.pdf'))
fig.savefig(os.path.join(OUTDIR, 'pareto_complexity_vs_ligo_mm.png'))
plt.close(fig)
print(f"  Saved pareto_complexity_vs_ligo_mm.pdf")

# Print summary table
print("\n  Pareto summary:")
print(f"  {'N':>5s}  {'Median LIGO':>12s}  {'90th LIGO':>12s}")
for p in pareto_points:
    print(f"  {p['N']:5d}  {p['median_ligo']:12.2e}  {p['p90_ligo']:12.2e}")
