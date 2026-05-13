"""
Compactify the 2955-term Ridge residual into a physically interpretable
closed-form expression.

Strategy:
  1. Truncate to k<=3 harmonics (2955 -> 1379, ~0% accuracy loss — validated)
  2. Within the k<=3 model, identify the dominant individual basis functions
  3. Group the top terms by physics sector and present the compact structure
  4. Refit a "skeleton" model: top N terms re-fitted on training data
  5. Factor (1-e^2)^{-p} for interpretive display of eccentricity structure
  6. Evaluate all compact models on validation data

Outputs:
  - compact_vs_full.pdf          : residual comparison plots
  - compact_sectors.pdf          : sector-by-sector breakdown
  - compact_accuracy.pdf         : accuracy scatter
  - compact_skeleton_accuracy.pdf: skeleton vs full accuracy by e, x
  - compact_ecc_resum.pdf        : (1-e^2)^{-p} interpretive fits
  - compact_formula.txt          : human-readable formula
  - compact_coefficients.json    : machine-readable coefficients
"""
import os, sys, pickle, json, warnings
import numpy as np
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.labelsize': 11, 'axes.titlesize': 10, 'legend.fontsize': 7,
    'legend.frameon': False, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.0, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.abspath(__file__))
SPIN_DIR = os.path.dirname(BASE)
RESULTS = os.path.join(SPIN_DIR, 'results')
OUTDIR = BASE

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
# Build group catalog
# ====================================================================
groups = []
idx = 0
for a in range(1, max_e + 1):
    for b in range(max_x + 1):
        for c in range(max_nu + 1):
            for d_s in range(max_chi + 1):
                for d_a in range(max_chi + 1):
                    if a + b + c + d_s + d_a > max_e + 3:
                        continue
                    groups.append({
                        'a': a, 'b': b, 'c': c, 'ds': d_s, 'da': d_a,
                        'start': idx, 'n_feat': 1 + 2 * n_harm
                    })
                    idx += 1 + 2 * n_harm

print(f"  {len(groups)} groups, {idx} total features")

K_MAX = 3  # truncation level


def group_label(g):
    parts = []
    if g['a'] > 0: parts.append(f"e^{g['a']}")
    if g['b'] > 0: parts.append(f"x^{g['b']}")
    if g['c'] > 0: parts.append(f"nu^{g['c']}")
    if g['ds'] > 0: parts.append("chiS")
    if g['da'] > 0: parts.append("chiA")
    return ' '.join(parts) if parts else '1'


def sector_label(g):
    if g['ds'] == 0 and g['da'] == 0:
        return 'nonspin'
    elif g['ds'] == 1 and g['da'] == 0:
        return 'chiS'
    elif g['ds'] == 0 and g['da'] == 1:
        return 'chiA'
    else:
        return 'chiS*chiA'


HARM_NAMES_K3 = ['base', 'cos1', 'sin1', 'cos2', 'sin2', 'cos3', 'sin3']


# ====================================================================
# Downsample data
# ====================================================================
def downsample_data(data, n_pts=200):
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


vd = downsample_data(val_data)
td = downsample_data(train_data, n_pts=400)
print(f"  Val points: {len(vd['e'])}, Train points: {len(td['e'])}")


# ====================================================================
# Build feature matrices
# ====================================================================
def build_full_basis(e, z, x, nu, chiS, chiA):
    """Full 2955-feature basis."""
    features = []
    for g in groups:
        base = e**g['a'] * x**g['b'] * nu**g['c'] * chiS**g['ds'] * chiA**g['da']
        features.append(base)
        for k in range(1, n_harm + 1):
            features.append(base * np.cos(k * z))
            features.append(base * np.sin(k * z))
    return np.column_stack(features)


# Build an index mapping: for each group and harmonic, what is the flat index?
feat_info = []  # list of (group_idx, group, harm_name, flat_idx)
for gi, g in enumerate(groups):
    s = g['start']
    feat_info.append((gi, g, 'base', s))
    for k in range(1, n_harm + 1):
        feat_info.append((gi, g, f'cos{k}', s + 2*k - 1))
        feat_info.append((gi, g, f'sin{k}', s + 2*k))


# ====================================================================
# Step 1: k<=3 truncated coefficients
# ====================================================================
print("\n" + "="*70)
print("STEP 1: k<=3 TRUNCATION")
print("="*70)

coef_a_k3 = coef_a_full.copy()
coef_w_k3 = coef_w_full.copy()
for g in groups:
    s = g['start']
    for k in range(K_MAX + 1, n_harm + 1):
        coef_a_k3[s + 2*k - 1] = 0
        coef_a_k3[s + 2*k] = 0
        coef_w_k3[s + 2*k - 1] = 0
        coef_w_k3[s + 2*k] = 0

# Count non-zero
n_nonzero_k3 = np.count_nonzero(coef_a_k3)
print(f"  k<=3 truncated: {n_nonzero_k3} non-zero coefficients "
      f"(vs {len(groups) * (1 + 2*K_MAX)} = {len(groups) * (1 + 2*K_MAX)} possible)")

X_val = build_full_basis(vd['e'], vd['z'], vd['x'], vd['nu'], vd['chiS'], vd['chiA'])

pred_a_full = X_val @ coef_a_full
pred_w_full = X_val @ coef_w_full
pred_a_k3 = X_val @ coef_a_k3
pred_w_k3 = X_val @ coef_w_k3

r2_full_a = r2_score(vd['dya'], pred_a_full)
r2_full_w = r2_score(vd['dyw'], pred_w_full)
r2_k3_a = r2_score(vd['dya'], pred_a_k3)
r2_k3_w = r2_score(vd['dyw'], pred_w_k3)
print(f"  Full:  R2_amp={r2_full_a:.6f}, R2_omega={r2_full_w:.6f}")
print(f"  k<=3:  R2_amp={r2_k3_a:.6f}, R2_omega={r2_k3_w:.6f}")


# ====================================================================
# Step 2: Identify dominant individual basis functions
# ====================================================================
print("\n" + "="*70)
print("STEP 2: RANKING INDIVIDUAL BASIS FUNCTIONS")
print("="*70)

# For each group g and harmonic channel h (within k<=3), compute
# the "importance" = |c_a| * std(feature) on training data
# This weights both the coefficient magnitude and how much the
# feature varies across the data.

# Compute feature stds on training data
X_train = build_full_basis(td['e'], td['z'], td['x'], td['nu'], td['chiS'], td['chiA'])
feat_stds = np.std(X_train, axis=0)

# Importance = |coef| * std
importance_a = np.abs(coef_a_k3) * feat_stds
importance_w = np.abs(coef_w_k3) * feat_stds
importance_combined = importance_a + importance_w

# Build a ranked list of (importance, flat_idx, group, harm_name)
ranked = []
for gi, g in enumerate(groups):
    s = g['start']
    # base
    ranked.append((importance_combined[s], s, g, 'base'))
    for k in range(1, K_MAX + 1):
        ranked.append((importance_combined[s + 2*k-1], s + 2*k-1, g, f'cos{k}'))
        ranked.append((importance_combined[s + 2*k], s + 2*k, g, f'sin{k}'))

ranked.sort(reverse=True)

print(f"\n  Top 30 basis functions by importance (|c| * std):")
print(f"  {'Rank':>4s} {'Basis':35s} {'|c_a|':>8s} {'|c_w|':>8s} {'std':>8s} "
      f"{'Import':>8s} {'Sector':10s}")
for i, (imp, fi, g, hn) in enumerate(ranked[:30]):
    label = f"{group_label(g)} * {hn}"
    sec = sector_label(g)
    print(f"  {i+1:4d} {label:35s} {abs(coef_a_k3[fi]):8.3f} {abs(coef_w_k3[fi]):8.3f} "
          f"{feat_stds[fi]:8.4f} {imp:8.3f} {sec:10s}")


# ====================================================================
# Step 3: Refit skeleton models with top N terms
# ====================================================================
print("\n" + "="*70)
print("STEP 3: SKELETON MODELS (REFIT TOP-N TERMS)")
print("="*70)

# For each N, select the top-N feature indices, build a sub-basis,
# and refit Ridge on training data.

skeleton_results = []

for N in [20, 30, 50, 75, 100, 150, 200, 300]:
    top_indices = [fi for _, fi, _, _ in ranked[:N]]

    X_tr_sub = X_train[:, top_indices]
    X_va_sub = X_val[:, top_indices]

    # Refit with small regularization
    ridge = Ridge(alpha=1e-6, fit_intercept=False)
    ridge.fit(X_tr_sub, td['dya'])
    pred_a_skel = ridge.predict(X_va_sub)
    r2a = r2_score(vd['dya'], pred_a_skel)

    ridge_w = Ridge(alpha=1e-6, fit_intercept=False)
    ridge_w.fit(X_tr_sub, td['dyw'])
    pred_w_skel = ridge_w.predict(X_va_sub)
    r2w = r2_score(vd['dyw'], pred_w_skel)

    skeleton_results.append({
        'N': N, 'r2_a': r2a, 'r2_w': r2w,
        'coef_a': ridge.coef_.copy(),
        'coef_w': ridge_w.coef_.copy(),
        'indices': top_indices,
        'pred_a': pred_a_skel,
        'pred_w': pred_w_skel,
    })

    print(f"  N={N:3d}: R2_amp={r2a:.6f}, R2_omega={r2w:.6f}")

# Select the N=50 skeleton as our "compact interpretable" model
skel_50 = [s for s in skeleton_results if s['N'] == 50][0]
skel_75 = [s for s in skeleton_results if s['N'] == 75][0]

# And also the N=100 as a "full compact" that's still much smaller
skel_100 = [s for s in skeleton_results if s['N'] == 100][0]

print(f"\n  Selected compact models:")
print(f"    Skeleton-50:  {skel_50['N']:3d} terms, R2_amp={skel_50['r2_a']:.6f}")
print(f"    Skeleton-75:  {skel_75['N']:3d} terms, R2_amp={skel_75['r2_a']:.6f}")
print(f"    Skeleton-100: {skel_100['N']:3d} terms, R2_amp={skel_100['r2_a']:.6f}")


# ====================================================================
# Step 4: Write out the skeleton-75 compact formula
# ====================================================================
print("\n" + "="*70)
print("STEP 4: COMPACT FORMULA (SKELETON-75)")
print("="*70)

skel = skel_75
top_N = skel['N']

# Map indices back to group/harmonic info
term_table = []
for rank_i, (imp, fi, g, hn) in enumerate(ranked[:top_N]):
    # Get the re-fitted coefficients
    local_idx = skel['indices'].index(fi)
    ca = skel['coef_a'][local_idx]
    cw = skel['coef_w'][local_idx]

    term_table.append({
        'rank': rank_i + 1,
        'label': f"{group_label(g)} * {hn}",
        'a': g['a'], 'b': g['b'], 'c': g['c'], 'ds': g['ds'], 'da': g['da'],
        'harmonic': hn,
        'sector': sector_label(g),
        'c_amp': float(ca), 'c_omega': float(cw),
        'importance': float(imp),
        'c_omega_over_c_amp': float(cw / ca) if abs(ca) > 0.01 else float('nan'),
    })

# Print grouped by sector
sectors_display = {}
for t in term_table:
    sec = t['sector']
    if sec not in sectors_display:
        sectors_display[sec] = []
    sectors_display[sec].append(t)

formula_lines = []
formula_lines.append("=" * 78)
formula_lines.append("COMPACT ECCENTRIC MODULATION FORMULA (Skeleton-75)")
formula_lines.append("=" * 78)
formula_lines.append("")
formula_lines.append(f"delta_xi_amp = SUM_i  c_i^(amp) * e^a_i * x^b_i * nu^c_i * [spin] * [harm(zeta)]")
formula_lines.append(f"delta_xi_omega = SUM_i  c_i^(omg) * e^a_i * x^b_i * nu^c_i * [spin] * [harm(zeta)]")
formula_lines.append("")
formula_lines.append(f"Terms: {top_N}")
formula_lines.append(f"R2_amp = {skel['r2_a']:.6f}  (full model: {r2_full_a:.6f})")
formula_lines.append(f"R2_omega = {skel['r2_w']:.6f}  (full model: {r2_full_w:.6f})")
formula_lines.append("")

for sec_name in ['nonspin', 'chiS', 'chiA', 'chiS*chiA']:
    if sec_name not in sectors_display:
        continue
    terms = sectors_display[sec_name]
    formula_lines.append(f"--- {sec_name.upper()} SECTOR ({len(terms)} terms) ---")
    formula_lines.append(f"  {'#':>3s} {'Basis function':35s} {'c_amp':>10s} {'c_omega':>10s} "
                         f"{'c_w/c_a':>8s}")

    for t in sorted(terms, key=lambda x: -abs(x['c_amp'])):
        ratio_str = f"{t['c_omega_over_c_amp']:.2f}" if not np.isnan(t['c_omega_over_c_amp']) else "---"
        formula_lines.append(
            f"  {t['rank']:3d} {t['label']:35s} {t['c_amp']:+10.3f} {t['c_omega']:+10.3f} "
            f"{ratio_str:>8s}"
        )
    formula_lines.append("")

formula_text = '\n'.join(formula_lines)
print(formula_text)

with open(os.path.join(OUTDIR, 'compact_formula.txt'), 'w') as f:
    f.write(formula_text)

# Save JSON
coeff_output = {
    'description': 'Compact eccentric modulation residual (skeleton-75, refitted)',
    'n_terms': top_N,
    'R2_amp': skel['r2_a'],
    'R2_omega': skel['r2_w'],
    'R2_amp_full': r2_full_a,
    'R2_omega_full': r2_full_w,
    'formula': 'sum_i c_i * e^a_i * x^b_i * nu^c_i * chiS^ds_i * chiA^da_i * harmonic_i(zeta)',
    'terms': term_table,
    'skeleton_scan': [{'N': s['N'], 'R2_amp': s['r2_a'], 'R2_omega': s['r2_w']}
                      for s in skeleton_results],
}
with open(os.path.join(OUTDIR, 'compact_coefficients.json'), 'w') as f:
    json.dump(coeff_output, f, indent=2)
print(f"\n  Saved compact_formula.txt, compact_coefficients.json")


# ====================================================================
# Step 5: (1-e^2)^{-p} interpretive analysis of skeleton terms
# ====================================================================
print("\n" + "="*70)
print("STEP 5: (1-e^2)^{-p} INTERPRETIVE ANALYSIS")
print("="*70)

# For each (b, c, sector, harmonic), collect the skeleton coefficients
# across eccentricity powers and check if they follow (1-e^2)^{-p} ratios.

# Group skeleton terms by (b, c, sector, harmonic)
interp_groups = {}
for t in term_table:
    key = (t['b'], t['c'], t['sector'], t['harmonic'])
    if key not in interp_groups:
        interp_groups[key] = {}
    interp_groups[key][t['a']] = t['c_amp']

print(f"\n  Interpretive eccentricity structure:")
print(f"  {'Group':40s} {'e^1':>8s} {'e^2':>8s} {'e^3':>8s} {'p_est':>7s} {'PN interp':15s}")

resum_display = []
for key in sorted(interp_groups.keys()):
    b, c, sec, harm = key
    coeffs = interp_groups[key]

    if len(coeffs) < 2:
        continue

    c1 = coeffs.get(1, 0)
    c2 = coeffs.get(2, 0)
    c3 = coeffs.get(3, 0)

    # From e*(1-e^2)^{-p} = e + p*e^3 + ...
    # So c3/c1 ≈ p for odd-power ratio
    # From e^2*(1-e^2)^{-p} = e^2 + p*e^4 + ...
    # Different interpretation for even-leading terms

    p_est = c3 / c1 if abs(c1) > 0.1 else float('nan')
    pn_label = ['0PN', '1PN', '2PN', '3PN'][b]

    parts = [f"x^{b}"]
    if c > 0: parts.append(f"nu^{c}")
    if sec != 'nonspin': parts.append(sec)
    parts.append(harm)
    label = ' '.join(parts)

    p_str = f"{p_est:.2f}" if not np.isnan(p_est) else "---"
    pn_expected = {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.5}
    interp = f"expect p={pn_expected.get(b, '?')}"

    print(f"  {label:40s} {c1:8.3f} {c2:8.3f} {c3:8.3f} {p_str:>7s} {interp:15s}")

    resum_display.append({
        'label': label, 'b': b, 'c': c, 'sector': sec, 'harmonic': harm,
        'coeffs': coeffs, 'p_est': p_est
    })


# ====================================================================
# Step 6: Generate all plots
# ====================================================================
print("\n" + "="*70)
print("STEP 6: GENERATING PLOTS")
print("="*70)

# --- Plot 1: Skeleton accuracy scan ---
fig, ax = plt.subplots(figsize=(5.5, 3.5))
Ns = [s['N'] for s in skeleton_results]
r2as = [s['r2_a'] for s in skeleton_results]
r2ws = [s['r2_w'] for s in skeleton_results]
ax.plot(Ns, r2as, 'o-', ms=5, lw=1.2, color='steelblue', label='$R^2$ amp')
ax.plot(Ns, r2ws, 's-', ms=5, lw=1.2, color='firebrick', label='$R^2$ omega')
ax.axhline(r2_full_a, color='steelblue', ls='--', lw=0.8, alpha=0.5,
           label=f'Full model amp ({r2_full_a:.4f})')
ax.axhline(r2_full_w, color='firebrick', ls='--', lw=0.8, alpha=0.5,
           label=f'Full model $\\omega$ ({r2_full_w:.4f})')
ax.set_xlabel('Number of basis functions (refitted)')
ax.set_ylabel('$R^2$ on validation set')
ax.set_title('Skeleton Model Accuracy vs Number of Terms', fontweight='bold')
ax.legend(fontsize=6)
ax.set_ylim(min(min(r2as), min(r2ws)) - 0.005, 1.001)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_skeleton_scan.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_skeleton_scan.png'))
plt.close(fig)
print("  Saved compact_skeleton_scan.pdf")


# --- Plot 2: Compact vs Full model residual comparison ---
fig, axes = plt.subplots(2, 3, figsize=(11, 6))

z_arr = np.linspace(0, 2*np.pi, 500)
test_cases = [
    (0.15, 0.03, 0.25, 0.0, 0.0,  '$e{=}0.15, x{=}0.03, \\nu{=}0.25$'),
    (0.30, 0.05, 0.25, 0.0, 0.0,  '$e{=}0.3, x{=}0.05, \\nu{=}0.25$'),
    (0.45, 0.07, 0.20, 0.0, 0.0,  '$e{=}0.45, x{=}0.07, \\nu{=}0.2$'),
    (0.30, 0.05, 0.25, 0.3, 0.0,  '$\\chi_S{=}0.3$'),
    (0.30, 0.05, 0.25, 0.0, 0.3,  '$\\chi_A{=}0.3$'),
    (0.30, 0.05, 0.25, 0.3, 0.2,  '$\\chi_S{=}0.3, \\chi_A{=}0.2$'),
]

for i, (e_val, x_val, nu_val, chiS_val, chiA_val, title) in enumerate(test_cases):
    ax = axes.ravel()[i]
    n = len(z_arr)
    ea = np.full(n, e_val); xa = np.full(n, x_val); nua = np.full(n, nu_val)
    chSa = np.full(n, chiS_val); chAa = np.full(n, chiA_val)

    X_test = build_full_basis(ea, z_arr, xa, nua, chSa, chAa)
    y_full = X_test @ coef_a_full
    y_k3 = X_test @ coef_a_k3

    # Skeleton-75
    X_skel = X_test[:, skel_75['indices']]
    y_skel = X_skel @ skel_75['coef_a']

    ax.plot(z_arr / np.pi, y_full, 'k-', lw=1.0, label='Full (2955)')
    ax.plot(z_arr / np.pi, y_skel, 'r--', lw=0.8, label=f'Skeleton-{top_N}')
    ax.fill_between(z_arr / np.pi, y_full, y_skel, alpha=0.1, color='red')
    ax.set_title(title, fontsize=7)
    if i >= 3: ax.set_xlabel('$\\zeta / \\pi$')
    if i % 3 == 0: ax.set_ylabel('$\\delta\\xi_{\\rm amp}$')
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)

fig.suptitle(f'Compact (Skeleton-{top_N}) vs Full Model: $\\delta\\xi_{{\\rm amp}}(\\zeta)$',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_vs_full.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_vs_full.png'))
plt.close(fig)
print("  Saved compact_vs_full.pdf")


# --- Plot 3: Sector breakdown ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Count terms per sector
sec_counts = {}
sec_importance = {}
for t in term_table:
    sec = t['sector']
    sec_counts[sec] = sec_counts.get(sec, 0) + 1
    sec_importance[sec] = sec_importance.get(sec, 0) + t['importance']

sec_order = ['nonspin', 'chiS', 'chiA', 'chiS*chiA']
sec_colors = {'nonspin': 'steelblue', 'chiS': '#2ca02c', 'chiA': '#d62728', 'chiS*chiA': '#9467bd'}

# Bar chart: term count
bars1 = ax1.bar([s for s in sec_order if s in sec_counts],
                [sec_counts.get(s, 0) for s in sec_order if s in sec_counts],
                color=[sec_colors[s] for s in sec_order if s in sec_counts])
ax1.set_ylabel('Number of terms')
ax1.set_title(f'Sector Distribution (Skeleton-{top_N})', fontweight='bold')

# Bar chart: importance
bars2 = ax2.bar([s for s in sec_order if s in sec_importance],
                [sec_importance.get(s, 0) for s in sec_order if s in sec_importance],
                color=[sec_colors[s] for s in sec_order if s in sec_importance])
ax2.set_ylabel('Total importance (|c| $\\times$ std)')
ax2.set_title(f'Sector Importance', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_sectors.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_sectors.png'))
plt.close(fig)
print("  Saved compact_sectors.pdf")


# --- Plot 4: Accuracy scatter (compact vs full) ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, (pf, pc, label) in zip(axes, [
    (pred_a_full, skel_75['pred_a'], '$\\delta\\xi_{\\rm amp}$'),
    (pred_w_full, skel_75['pred_w'], '$\\delta\\xi_\\omega$'),
]):
    idx = np.random.RandomState(42).choice(len(pf), min(5000, len(pf)), replace=False)
    ax.scatter(pf[idx], pc[idx], s=0.5, alpha=0.3, c='steelblue', rasterized=True)
    lims = [min(pf[idx].min(), pc[idx].min()), max(pf[idx].max(), pc[idx].max())]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Full model')
    ax.set_ylabel(f'Skeleton-{top_N}')
    ax.set_title(label, fontweight='bold')
    ax.set_aspect('equal')
    r2 = r2_score(pf, pc)
    ax.text(0.05, 0.95, f'$R^2 = {r2:.5f}$', transform=ax.transAxes, fontsize=8, va='top')

fig.suptitle(f'Skeleton-{top_N} vs Full: Prediction Correlation', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_accuracy.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_accuracy.png'))
plt.close(fig)
print("  Saved compact_accuracy.pdf")


# --- Plot 5: Skeleton accuracy vs parameter regions ---
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

rel_err = np.abs(skel_75['pred_a'] - pred_a_full) / (np.abs(pred_a_full) + 1e-6)

for ax, (xvar, xlabel, color) in zip(axes, [
    (vd['e'], 'Eccentricity $e$', 'steelblue'),
    (vd['x'], 'Frequency $x$', 'firebrick'),
    (vd['chiS'], '$\\chi_S$', 'forestgreen'),
]):
    ax.scatter(xvar, rel_err, s=0.3, alpha=0.2, c=color, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 10)
    if ax == axes[0]:
        ax.set_ylabel('Relative error vs full model')
    ax.axhline(0.01, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(0.1, color='gray', ls=':', lw=0.5, alpha=0.5)

fig.suptitle(f'Skeleton-{top_N} Relative Error vs Full Model', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_error_vs_params.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_error_vs_params.png'))
plt.close(fig)
print("  Saved compact_error_vs_params.pdf")


# --- Plot 6: (1-e^2)^{-p} eccentricity resummation display ---
e_grid = np.linspace(0.01, 0.5, 200)

# Show how the polynomial in e maps to (1-e^2)^{-p} for a few key groups
fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
axes = axes.ravel()

plot_groups = [rd for rd in resum_display if len(rd['coeffs']) >= 2]
plot_groups.sort(key=lambda x: sum(abs(v) for v in x['coeffs'].values()), reverse=True)

for i, rd in enumerate(plot_groups[:8]):
    ax = axes[i]
    coeffs = rd['coeffs']

    # Polynomial
    y_poly = sum(c * e_grid**a for a, c in coeffs.items())

    # Best (1-e^2)^{-p} fit
    def model(e, c_eff, p):
        return c_eff * e / (1.0 - e**2)**p

    c1 = coeffs.get(1, coeffs.get(2, 0))
    p0 = rd['p_est'] if not np.isnan(rd['p_est']) else 1.0
    p0 = np.clip(p0, -2, 8)
    try:
        popt, _ = curve_fit(model, e_grid, y_poly, p0=[c1, p0],
                            maxfev=10000, bounds=([-500, -3], [500, 10]))
        y_resum = model(e_grid, *popt)
        p_fit = popt[1]
    except Exception:
        y_resum = np.zeros_like(e_grid)
        p_fit = float('nan')

    ax.plot(e_grid, y_poly, 'k-', lw=1.0, label='Polynomial')
    if not np.isnan(p_fit):
        ax.plot(e_grid, y_resum, 'r--', lw=0.8,
                label=f'$e/(1{{-}}e^2)^{{{p_fit:.2f}}}$')
    ax.set_title(rd['label'], fontsize=6)
    if i >= 4: ax.set_xlabel('$e$')
    if i % 4 == 0: ax.set_ylabel('Value')
    ax.legend(fontsize=4)
    ax.tick_params(labelsize=5)

fig.suptitle('Eccentricity Resummation: Polynomial $\\to$ $(1-e^2)^{-p}$ (Interpretive)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_ecc_resum.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_ecc_resum.png'))
plt.close(fig)
print("  Saved compact_ecc_resum.pdf")


# --- Plot 7: Coefficient heatmap by (PN order, harmonic) ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (sec_name, sec_title) in zip(axes, [('nonspin', 'Non-spin'), ('chiS', '$\\chi_S$-linear')]):
    # Build matrix: rows = harmonics, cols = PN order, value = sum of |c_amp|
    harm_list = ['base', 'cos1', 'sin1', 'cos2', 'sin2', 'cos3', 'sin3']
    pn_list = [0, 1, 2, 3]
    matrix = np.zeros((len(harm_list), len(pn_list)))

    for t in term_table:
        if t['sector'] != sec_name:
            continue
        if t['harmonic'] not in harm_list:
            continue
        hi = harm_list.index(t['harmonic'])
        pi = t['b']
        if pi < len(pn_list):
            matrix[hi, pi] += abs(t['c_amp'])

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xticks(range(len(pn_list)))
    ax.set_xticklabels(['0PN', '1PN', '2PN', '3PN'])
    ax.set_yticks(range(len(harm_list)))
    ax.set_yticklabels(harm_list, fontsize=7)
    ax.set_xlabel('PN order')
    ax.set_ylabel('Harmonic channel')
    ax.set_title(f'{sec_title} Sector', fontweight='bold')
    plt.colorbar(im, ax=ax, label='$|c_{\\rm amp}|$', shrink=0.8)

    # Annotate values
    for hi in range(len(harm_list)):
        for pi in range(len(pn_list)):
            if matrix[hi, pi] > 0.5:
                ax.text(pi, hi, f'{matrix[hi,pi]:.0f}', ha='center', va='center',
                        fontsize=5, color='white' if matrix[hi,pi] > matrix.max()/2 else 'black')

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'compact_heatmap.pdf'))
fig.savefig(os.path.join(OUTDIR, 'compact_heatmap.png'))
plt.close(fig)
print("  Saved compact_heatmap.pdf")


# ====================================================================
# Final summary
# ====================================================================
print("\n" + "="*70)
print("COMPACTIFICATION COMPLETE")
print("="*70)

print(f"""
  Compactification Summary
  ========================

  Full model:        2955 params, R2_amp = {r2_full_a:.6f}
  k<=3 truncated:    1379 params, R2_amp = {r2_k3_a:.6f}  (53% reduction, ~0% loss)
  Skeleton-50:         50 terms,  R2_amp = {skel_50['r2_a']:.6f}  (98% reduction)
  Skeleton-75:         75 terms,  R2_amp = {skel_75['r2_a']:.6f}  (97% reduction)
  Skeleton-100:       100 terms,  R2_amp = {skel_100['r2_a']:.6f}  (97% reduction)

  Sector breakdown (Skeleton-75):
    Non-spin:    {sec_counts.get('nonspin', 0):2d} terms
    chiS-linear: {sec_counts.get('chiS', 0):2d} terms
    chiA-linear: {sec_counts.get('chiA', 0):2d} terms
    chiS*chiA:   {sec_counts.get('chiS*chiA', 0):2d} terms

  Generated files:
""")
for f in sorted(os.listdir(OUTDIR)):
    if f.startswith('compact'):
        print(f"    {f}")
