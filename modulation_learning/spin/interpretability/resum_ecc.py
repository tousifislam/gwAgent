"""
Refit the Ridge residual with explicit (1-e^2)^{-p} eccentricity
resummation factors.

PN theory predicts denominators (1-e^2)^{-p} at each PN order:
  0PN: p=1,   1PN: p=2,   1.5PN: p=7/2,   2PN: p=3

Strategy:
  1. For each PN order (x^b), build a new basis where the eccentricity
     polynomial e^a is replaced by e^a / (1-e^2)^{p(b)}.
  2. Refit Ridge on the resummed basis.
  3. Compare accuracy vs the raw polynomial basis.
  4. Scan over p values to find the best fit.

This produces a COMPACT resummed model where the eccentricity
dependence is captured by a few resummed terms rather than
5 polynomial powers.
"""
import os, sys, pickle, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
SPIN_DIR = os.path.dirname(BASE)
RESULTS = os.path.join(SPIN_DIR, 'results')

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 7,
    'legend.frameon': False, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})

print("Loading model and data...")
model = pickle.load(open(os.path.join(
    RESULTS, 'models/ridge_nh7_me5_mchi1_a1e-06/model.pkl'), 'rb'))
bc = model['bc']
max_e = bc['max_e']; max_x = bc['max_x']; max_nu = bc['max_nu']
max_chi = bc['max_chi']; n_harm = bc['n_harm']

train = pickle.load(open(os.path.join(RESULTS, 'training_data.pkl'), 'rb'))
val = pickle.load(open(os.path.join(RESULTS, 'validation_data.pkl'), 'rb'))


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
        out['z'].append(z_[idx])
        out['nu'].append(np.full(len(idx), d['nu']))
        out['chiS'].append(np.full(len(idx), d['chi_S']))
        out['chiA'].append(np.full(len(idx), d['chi_A']))
        out['dya'].append(d['delta_xi_amp'][idx])
        out['dyw'].append(d['delta_xi_omega'][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


td = downsample(train); vd = downsample(val, n_pts=200)
print(f"  Train: {len(td['e'])} pts, Val: {len(vd['e'])} pts")

K_MAX = 3  # harmonics up to k=3 (validated: no accuracy loss)


# ======================================================================
# 1. Build raw polynomial basis (baseline)
# ======================================================================
def build_poly_basis(e, z, x, nu, chiS, chiA):
    """Standard polynomial × Fourier basis (same as fit.py)."""
    features = []
    for a in range(1, max_e + 1):
        for b in range(max_x + 1):
            for c in range(max_nu + 1):
                for ds in range(max_chi + 1):
                    for da in range(max_chi + 1):
                        if a + b + c + ds + da > max_e + 3:
                            continue
                        base = e**a * x**b * nu**c * chiS**ds * chiA**da
                        features.append(base)
                        for k in range(1, K_MAX + 1):
                            features.append(base * np.cos(k * z))
                            features.append(base * np.sin(k * z))
    return np.column_stack(features)


# ======================================================================
# 2. Build resummed basis: e^a → e^a / (1-e^2)^{p(b)}
# ======================================================================
def build_resum_basis(e, z, x, nu, chiS, chiA, p_dict):
    """Resummed basis: each PN order x^b gets its own (1-e^2)^{-p(b)}.

    p_dict maps b → p value.  E.g. {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.5}
    """
    e2 = e * e
    one_m_e2 = np.clip(1.0 - e2, 1e-6, None)
    features = []
    for a in range(1, max_e + 1):
        for b in range(max_x + 1):
            p = p_dict.get(b, 1.0)
            for c in range(max_nu + 1):
                for ds in range(max_chi + 1):
                    for da in range(max_chi + 1):
                        if a + b + c + ds + da > max_e + 3:
                            continue
                        # Resummed: e^a / (1-e^2)^p instead of plain e^a
                        base = (e**a / one_m_e2**p) * x**b * nu**c * chiS**ds * chiA**da
                        features.append(base)
                        for k in range(1, K_MAX + 1):
                            features.append(base * np.cos(k * z))
                            features.append(base * np.sin(k * z))
    return np.column_stack(features)


# ======================================================================
# 3. Baseline: raw polynomial fit
# ======================================================================
print("\n=== Baseline: raw polynomial basis (k<=3) ===")
X_tr_poly = build_poly_basis(td['e'], td['z'], td['x'], td['nu'], td['chiS'], td['chiA'])
X_va_poly = build_poly_basis(vd['e'], vd['z'], vd['x'], vd['nu'], vd['chiS'], vd['chiA'])

ra_poly = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr_poly, td['dya'])
rw_poly = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr_poly, td['dyw'])
r2a_poly = r2_score(vd['dya'], ra_poly.predict(X_va_poly))
r2w_poly = r2_score(vd['dyw'], rw_poly.predict(X_va_poly))
print(f"  R2_amp = {r2a_poly:.6f}, R2_omega = {r2w_poly:.6f}")
print(f"  n_features = {X_tr_poly.shape[1]}")


# ======================================================================
# 4. Scan p values for each PN order
# ======================================================================
print("\n=== Scanning (1-e^2)^{-p} values ===")

# PN-motivated priors
p_grid = np.arange(0.5, 5.5, 0.5)

# Fix all PN orders to the same p first, then scan individually
print("\n  Uniform p for all PN orders:")
results_uniform = []
for p in p_grid:
    p_dict = {b: p for b in range(max_x + 1)}
    X_tr = build_resum_basis(td['e'], td['z'], td['x'], td['nu'], td['chiS'], td['chiA'], p_dict)
    X_va = build_resum_basis(vd['e'], vd['z'], vd['x'], vd['nu'], vd['chiS'], vd['chiA'], p_dict)
    ra = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr, td['dya'])
    rw = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr, td['dyw'])
    r2a = r2_score(vd['dya'], ra.predict(X_va))
    r2w = r2_score(vd['dyw'], rw.predict(X_va))
    results_uniform.append({'p': p, 'r2a': r2a, 'r2w': r2w})
    print(f"    p={p:.1f}: R2_amp={r2a:.6f}, R2_omega={r2w:.6f}")

best_uniform = max(results_uniform, key=lambda r: r['r2a'] + r['r2w'])
print(f"  Best uniform: p={best_uniform['p']:.1f}")

# Now scan each PN order independently
print("\n  Independent p per PN order:")
best_p = {}
for b_target in range(max_x + 1):
    best_r2 = -1e10
    for p in p_grid:
        p_dict = {b: best_uniform['p'] for b in range(max_x + 1)}
        p_dict[b_target] = p
        X_tr = build_resum_basis(td['e'], td['z'], td['x'], td['nu'],
                                 td['chiS'], td['chiA'], p_dict)
        X_va = build_resum_basis(vd['e'], vd['z'], vd['x'], vd['nu'],
                                 vd['chiS'], vd['chiA'], p_dict)
        ra = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr, td['dya'])
        rw = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr, td['dyw'])
        r2_total = r2_score(vd['dya'], ra.predict(X_va)) + \
                   r2_score(vd['dyw'], rw.predict(X_va))
        if r2_total > best_r2:
            best_r2 = r2_total; best_p[b_target] = p
    pn_label = ['0PN', '1PN', '2PN', '3PN'][b_target]
    pn_expected = [1.0, 2.0, 3.0, 3.5][b_target]
    print(f"    x^{b_target} ({pn_label}): best p={best_p[b_target]:.1f} "
          f"(PN expects p={pn_expected:.1f})")


# ======================================================================
# 5. Final refit with best p values
# ======================================================================
print(f"\n=== Final refit with best p values: {best_p} ===")
X_tr_resum = build_resum_basis(td['e'], td['z'], td['x'], td['nu'],
                                td['chiS'], td['chiA'], best_p)
X_va_resum = build_resum_basis(vd['e'], vd['z'], vd['x'], vd['nu'],
                                vd['chiS'], vd['chiA'], best_p)

ra_resum = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr_resum, td['dya'])
rw_resum = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr_resum, td['dyw'])
r2a_resum = r2_score(vd['dya'], ra_resum.predict(X_va_resum))
r2w_resum = r2_score(vd['dyw'], rw_resum.predict(X_va_resum))
print(f"  Resummed R2_amp = {r2a_resum:.6f} (poly: {r2a_poly:.6f})")
print(f"  Resummed R2_omega = {r2w_resum:.6f} (poly: {r2w_poly:.6f})")
print(f"  Improvement: R2_amp {(r2a_resum-r2a_poly)*100:+.4f}%, "
      f"R2_omega {(r2w_resum-r2w_poly)*100:+.4f}%")

# Now do the skeleton refit with the resummed basis
print("\n  Skeleton scan on resummed basis:")
feat_stds = np.std(X_tr_resum, axis=0)
importance = (np.abs(ra_resum.coef_) + np.abs(rw_resum.coef_)) * feat_stds
idx_sorted = np.argsort(importance)[::-1]

for N in [20, 30, 50, 75, 100]:
    top_idx = idx_sorted[:N]
    ra_s = Ridge(alpha=1e-6, fit_intercept=False).fit(
        X_tr_resum[:, top_idx], td['dya'])
    rw_s = Ridge(alpha=1e-6, fit_intercept=False).fit(
        X_tr_resum[:, top_idx], td['dyw'])
    r2a = r2_score(vd['dya'], ra_s.predict(X_va_resum[:, top_idx]))
    r2w = r2_score(vd['dyw'], rw_s.predict(X_va_resum[:, top_idx]))
    print(f"    N={N:3d}: R2_amp={r2a:.6f}, R2_omega={r2w:.6f}")


# ======================================================================
# 6. Plot
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Panel 1: R2 vs uniform p
ax = axes[0]
ps = [r['p'] for r in results_uniform]
r2as = [r['r2a'] for r in results_uniform]
r2ws = [r['r2w'] for r in results_uniform]
ax.plot(ps, r2as, 'o-', ms=5, color='steelblue', label='$R^2$ amp')
ax.plot(ps, r2ws, 's-', ms=5, color='firebrick', label='$R^2$ omega')
ax.axhline(r2a_poly, ls='--', color='steelblue', lw=0.7, alpha=0.5,
           label=f'Poly baseline ({r2a_poly:.4f})')
ax.set_xlabel('$p$ in $(1-e^2)^{-p}$')
ax.set_ylabel('$R^2$ (validation)')
ax.set_title('Uniform $p$ scan', fontweight='bold')
ax.legend(fontsize=6)

# Panel 2: Best p per PN order vs PN expectation
ax = axes[1]
b_vals = sorted(best_p.keys())
p_fitted = [best_p[b] for b in b_vals]
p_expected = [1.0, 2.0, 3.0, 3.5]
pn_labels = ['0PN', '1PN', '2PN', '3PN']
ax.bar(np.array(b_vals) - 0.15, p_fitted, width=0.3, color='steelblue',
       label='Fitted $p$', edgecolor='0.3', lw=0.5)
ax.bar(np.array(b_vals) + 0.15, p_expected, width=0.3, color='#d62728',
       alpha=0.5, label='PN prediction', edgecolor='0.3', lw=0.5)
ax.set_xticks(b_vals)
ax.set_xticklabels(pn_labels)
ax.set_xlabel('PN order')
ax.set_ylabel('$p$ exponent')
ax.set_title('$(1-e^2)^{-p}$: fitted vs PN', fontweight='bold')
ax.legend()

# Panel 3: Skeleton scan comparison (poly vs resummed)
ax = axes[2]
# Re-do poly skeleton scan
imp_poly = (np.abs(ra_poly.coef_) + np.abs(rw_poly.coef_)) * np.std(X_tr_poly, axis=0)
idx_poly = np.argsort(imp_poly)[::-1]
Ns = [20, 30, 50, 75, 100, 150]
r2_poly_skel = []; r2_resum_skel = []
for N in Ns:
    # Poly skeleton
    ti = idx_poly[:N]
    ra_s = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr_poly[:, ti], td['dya'])
    r2_poly_skel.append(r2_score(vd['dya'], ra_s.predict(X_va_poly[:, ti])))
    # Resummed skeleton
    ti = idx_sorted[:N]
    ra_s = Ridge(alpha=1e-6, fit_intercept=False).fit(X_tr_resum[:, ti], td['dya'])
    r2_resum_skel.append(r2_score(vd['dya'], ra_s.predict(X_va_resum[:, ti])))

ax.plot(Ns, r2_poly_skel, 'o-', ms=5, color='0.5', label='Polynomial basis')
ax.plot(Ns, r2_resum_skel, 's-', ms=5, color='steelblue',
        label='$(1-e^2)^{-p}$ resummed')
ax.set_xlabel('Number of skeleton terms')
ax.set_ylabel('$R^2$ amp (validation)')
ax.set_title('Skeleton: poly vs resummed', fontweight='bold')
ax.legend()

fig.suptitle('Eccentricity resummation: $(1-e^2)^{-p}$ refitting',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'resum_ecc_refit.pdf'))
fig.savefig(os.path.join(BASE, 'resum_ecc_refit.png'), dpi=300)
plt.close(fig)

# Save results
summary = {
    'best_p': {str(k): v for k, v in best_p.items()},
    'pn_expected_p': {'0': 1.0, '1': 2.0, '2': 3.0, '3': 3.5},
    'r2_poly': {'amp': r2a_poly, 'omega': r2w_poly},
    'r2_resummed': {'amp': r2a_resum, 'omega': r2w_resum},
    'best_uniform_p': best_uniform['p'],
}
json.dump(summary, open(os.path.join(BASE, 'resum_ecc_summary.json'), 'w'), indent=2)
print(f"\nSaved resum_ecc_refit.{{pdf,png}} and resum_ecc_summary.json")
