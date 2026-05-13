"""
Generate all standard diagnostic plots for the compact (top-75)
Ridge + phase correction model and save under
results/models/final_ridge_compactified_phase_corr/.

This is the production-speed model used in all comparison plots.
Reuses fit.py's make_plots + save_error_arrays machinery.
"""
import os, sys, pickle
import numpy as np
from sklearn.linear_model import Ridge

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE, 'timing_optimization'))

from fit import (build_basis, h22_ecc_ansatz, poly_phase_correction,
                 reconstruct, eval_all, make_plots, MTOT_VALUES,
                 RESULTS)
from optimize import (precompute_column_specs, build_basis_numba,
                      predict_numba)

NAME = 'final_ridge_compactified_phase_corr'
OUTDIR = os.path.join(RESULTS, 'models', NAME)
ERRDIR = os.path.join(RESULTS, 'errors', NAME)
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(ERRDIR, exist_ok=True)

# ======================================================================
# 1. Load data + full Ridge model
# ======================================================================
print("Loading data and full Ridge model...")
train = pickle.load(open(os.path.join(RESULTS, 'training_data.pkl'), 'rb'))
val = pickle.load(open(os.path.join(RESULTS, 'validation_data.pkl'), 'rb'))

opus = pickle.load(open(os.path.join(
    RESULTS, 'models', 'ridge_nh7_me5_mchi1_a1e-06', 'model.pkl'), 'rb'))
bc = opus['bc']
full_specs = precompute_column_specs(
    max_e=bc['max_e'], max_x=bc['max_x'], max_nu=bc['max_nu'],
    max_chi=bc['max_chi'], n_harm=bc['n_harm'])
ca_full = opus['m_a'].coef_.astype(np.float64)
cw_full = opus['m_w'].coef_.astype(np.float64)

# ======================================================================
# 2. Build compact 75-term skeleton refit
# ======================================================================
print("Building compact 75-term skeleton refit...")


def downsample_points(data, n_pts=400):
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


td = downsample_points(train, n_pts=400)
X_train = build_basis_numba(td['e'], td['z'], td['x'], td['nu'],
                            td['chiS'], td['chiA'], full_specs)
feat_stds = np.std(X_train, axis=0)
importance = (np.abs(ca_full) + np.abs(cw_full)) * feat_stds
top75 = np.argsort(importance)[::-1][:75]
specs75 = full_specs[top75]
X_sub = X_train[:, top75]
m_a = Ridge(alpha=1e-6, fit_intercept=False).fit(X_sub, td['dya'])
m_w = Ridge(alpha=1e-6, fit_intercept=False).fit(X_sub, td['dyw'])
ca75 = m_a.coef_.astype(np.float64)
cw75 = m_w.coef_.astype(np.float64)
print(f"  75 features selected and refit")

# Save model.pkl
pickle.dump({
    'm_a': m_a, 'm_w': m_w,
    'bc': bc, 'alpha': 1e-6, 'nf': 75,
    'parent_model': 'ridge_nh7_me5_mchi1_a1e-06',
    'compactification': 'top-75 by |coef|*std importance',
    'type': 'ridge_compactified+phase_corr',
    'specs75_indices': top75.tolist(),
}, open(os.path.join(OUTDIR, 'model.pkl'), 'wb'))

# ======================================================================
# 3. Define predict function
# ======================================================================
def predict_compact(e, x, z, nu, chi_S, chi_A):
    n = len(e)
    xi_a_ans = np.abs(h22_ecc_ansatz(x, e, z, nu)) - 1.0
    xi_w_ans = xi_a_ans / 0.9
    nu_a = np.full(n, nu) if np.isscalar(nu) else nu
    cS_a = np.full(n, chi_S) if np.isscalar(chi_S) else chi_S
    cA_a = np.full(n, chi_A) if np.isscalar(chi_A) else chi_A
    B = build_basis_numba(e, z, x, nu_a, cS_a, cA_a, specs75)
    da, dw = predict_numba(B, ca75, cw75)
    return xi_a_ans + da, xi_w_ans + dw


pc = poly_phase_correction(order=5)

# ======================================================================
# 4. Evaluate on train and val
# ======================================================================
print("Evaluating on training set (300 waveforms)...")
E_tr, MM_tr, dphi_tr, mono_tr, ligo_tr = eval_all(
    train, predict_compact, pc, compute_ligo=True)
print(f"  train: E_med={np.median(E_tr):.2e}, dphi_med={np.median(dphi_tr):.3f}")

print("Evaluating on validation set (150 waveforms)...")
E_va, MM_va, dphi_va, mono_va, ligo_va = eval_all(
    val, predict_compact, pc, compute_ligo=True)
print(f"  val: E_med={np.median(E_va):.2e}, dphi_med={np.median(dphi_va):.3f}")

# ======================================================================
# 5. Save error arrays
# ======================================================================
print("Saving error arrays...")
for tag, data, Es, MMs, dphis, ligos in [
    ('train', train, E_tr, MM_tr, dphi_tr, ligo_tr),
    ('val', val, E_va, MM_va, dphi_va, ligo_va),
]:
    params = np.array([[d['q'], d['chi1'], d['chi2'], d['e0']] for d in data])
    np.save(os.path.join(ERRDIR, f'{tag}_params.npy'), params)
    np.save(os.path.join(ERRDIR, f'{tag}_mathcalE.npy'), Es)
    np.save(os.path.join(ERRDIR, f'{tag}_td_mismatch.npy'), MMs)
    np.save(os.path.join(ERRDIR, f'{tag}_dephasing.npy'), dphis)
    np.save(os.path.join(ERRDIR, f'{tag}_ligo_mismatch.npy'), ligos)

# ======================================================================
# 6. Make all diagnostic plots + summary.json
# ======================================================================
print("Making plots...")
summary = make_plots(
    OUTDIR, NAME, train, val,
    E_tr, MM_tr, dphi_tr, E_va, MM_va, dphi_va,
    predict_compact,
    mono_tr=mono_tr, mono_va=mono_va,
    ligo_tr=ligo_tr, ligo_va=ligo_va,
    phase_corr=pc)

print(f"\nSummary:")
for k in ['val_E_med', 'val_dphi_med', 'val_dphi_max',
          'val_ligo_mm_20_med', 'val_ligo_mm_20_max',
          'val_ligo_mm_65_med', 'val_ligo_mm_110_med',
          'val_ligo_mm_155_med', 'val_ligo_mm_200_med']:
    if k in summary:
        print(f"  {k}: {summary[k]:.3e}")

print(f"\nSaved to:")
print(f"  {OUTDIR}/  (model.pkl, summary.json, 12 PDF plots)")
print(f"  {ERRDIR}/  (10 npy arrays)")
