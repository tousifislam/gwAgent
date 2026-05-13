"""
Interpretability analysis for the Ridge modulation model.
Produces numerical results and plots for the interpret.pdf document.

Steps:
  1. Fourier diagnostic (A_k vs k, Bessel/Hansen check)
  2. Padé approximation in w = exp(iz)
  3. (1-e^2)^{-p} extraction
  4. Direct PN comparison
"""
import os, sys, pickle, warnings
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import jv as besselJ
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
OUTDIR = BASE  # interpretability/

# ====================================================================
# Load model
# ====================================================================
model_path = os.path.join(RESULTS, 'models', 'ridge_nh7_me5_mchi1_a1e-06', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

bc = model['bc']
max_e = bc['max_e']   # 5
max_x = bc['max_x']   # 3
max_nu = bc['max_nu']  # 2
max_chi = bc['max_chi']  # 1
n_harm = bc['n_harm']  # 7

coef_a = model['m_a'].coef_
coef_w = model['m_w'].coef_

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

print(f"Total groups: {len(groups)}, total features: {idx}")

def group_label(g):
    parts = []
    if g['a'] > 0: parts.append(f"e^{g['a']}")
    if g['b'] > 0: parts.append(f"x^{g['b']}")
    if g['c'] > 0: parts.append(f"nu^{g['c']}")
    if g['ds'] > 0: parts.append("chiS")
    if g['da'] > 0: parts.append("chiA")
    return ' '.join(parts)

def get_fourier_coeffs(g, coefs):
    """Extract (c0, [(alpha_k, beta_k) for k=1..n_harm]) for a group."""
    s = g['start']
    c0 = coefs[s]
    harmonics = []
    for k in range(1, n_harm + 1):
        alpha_k = coefs[s + 2*k - 1]
        beta_k = coefs[s + 2*k]
        harmonics.append((alpha_k, beta_k))
    return c0, harmonics

def group_weight(g, coefs):
    s = g['start']
    return np.sum(np.abs(coefs[s:s + g['n_feat']]))

# Sort groups by weight
gw = [(group_weight(g, coef_a), g) for g in groups]
gw.sort(reverse=True)

# ====================================================================
# STEP 1: Fourier diagnostic
# ====================================================================
print("\n" + "="*70)
print("STEP 1: FOURIER DIAGNOSTIC")
print("="*70)

fig, axes = plt.subplots(5, 6, figsize=(14, 10))
axes = axes.ravel()

bessel_fits = []

for rank in range(30):
    w, g = gw[rank]
    c0, harms = get_fourier_coeffs(g, coef_a)
    Ak = np.array([np.sqrt(a**2 + b**2) for a, b in harms])
    ks = np.arange(1, n_harm + 1)

    # Normalize
    Ak_norm = Ak / (Ak[0] + 1e-20)

    # Fit geometric decay: A_k ~ A_1 * r^(k-1)
    if Ak[0] > 1e-10:
        log_Ak = np.log(Ak + 1e-20)
        # Linear fit in log space for geometric decay
        try:
            p = np.polyfit(ks[Ak > 1e-10] - 1, log_Ak[Ak > 1e-10], 1)
            decay_rate = np.exp(p[0])
        except:
            decay_rate = 0.0
    else:
        decay_rate = 0.0

    # Fit Bessel envelope: A_k ~ C * |J_k(k*e_eff)|
    # For the eccentricity power a, the effective e argument scales
    def bessel_model(k, C, e_eff):
        return C * np.abs(besselJ(k, k * e_eff))

    try:
        popt, _ = curve_fit(bessel_model, ks, Ak, p0=[Ak[0]*2, 0.3],
                           bounds=([0, 0.01], [Ak[0]*100, 0.99]), maxfev=5000)
        bessel_C, bessel_e = popt
        bessel_resid = np.sqrt(np.mean((Ak - bessel_model(ks, *popt))**2)) / (np.mean(Ak) + 1e-10)
    except:
        bessel_C, bessel_e, bessel_resid = 0, 0, 999

    bessel_fits.append({
        'rank': rank, 'label': group_label(g), 'weight': w,
        'decay_rate': decay_rate, 'bessel_e': bessel_e,
        'bessel_resid': bessel_resid, 'Ak': Ak,
        'group': g
    })

    ax = axes[rank]
    ax.semilogy(ks, Ak, 'ko-', ms=3, lw=0.8, label='Data')

    # Plot geometric decay
    if decay_rate > 0 and decay_rate < 1:
        ax.semilogy(ks, Ak[0] * decay_rate**(ks - 1), 'b--', lw=0.6, alpha=0.7,
                    label=f'Geom r={decay_rate:.2f}')

    # Plot Bessel fit
    if bessel_resid < 5:
        k_fine = np.linspace(1, n_harm, 50)
        ax.semilogy(k_fine, bessel_model(k_fine, bessel_C, bessel_e), 'r-', lw=0.6,
                    alpha=0.7, label=f'Bessel e={bessel_e:.2f}')

    ax.set_title(f'{rank+1}. {group_label(g)}', fontsize=6)
    ax.set_ylim(bottom=max(1e-3, Ak.min()/3))
    if rank >= 24: ax.set_xlabel('$k$', fontsize=7)
    if rank % 6 == 0: ax.set_ylabel('$A_k$', fontsize=7)
    ax.legend(fontsize=4, loc='upper right')
    ax.tick_params(labelsize=5)

fig.suptitle('Fourier Amplitude Spectra $A_k = \\sqrt{\\alpha_k^2 + \\beta_k^2}$ for Top 30 Groups',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fourier_diagnostic.pdf'))
fig.savefig(os.path.join(OUTDIR, 'fourier_diagnostic.png'))
plt.close(fig)
print("  Saved fourier_diagnostic.pdf")

# Summary table
print("\n  Fourier diagnostic summary (top 30 groups):")
print(f"  {'Rank':>4s} {'Label':30s} {'|c|':>8s} {'r_geom':>7s} {'e_Bess':>7s} {'Bess_err':>8s} {'A2/A1':>6s} {'A3/A1':>6s}")
for bf in bessel_fits:
    Ak = bf['Ak']
    r21 = Ak[1]/Ak[0] if Ak[0] > 0 else 0
    r31 = Ak[2]/Ak[0] if Ak[0] > 0 else 0
    print(f"  {bf['rank']+1:4d} {bf['label']:30s} {bf['weight']:8.1f} {bf['decay_rate']:7.3f} "
          f"{bf['bessel_e']:7.3f} {bf['bessel_resid']:8.3f} {r21:6.3f} {r31:6.3f}")


# ====================================================================
# STEP 2: Padé approximation in w = exp(iz)
# ====================================================================
print("\n" + "="*70)
print("STEP 2: PADÉ APPROXIMATION")
print("="*70)

def fourier_eval(c0, harms, z):
    """Evaluate Fourier sum at array of z values."""
    val = np.full_like(z, c0, dtype=float)
    for k, (ak, bk) in enumerate(harms, 1):
        val += ak * np.cos(k * z) + bk * np.sin(k * z)
    return val

def pade_rational_form(z, A, B, C, D, E):
    """Rational function: (A + B*cos(z) + C*sin(z)) / (1 + D*cos(z) + E*sin(z))"""
    num = A + B * np.cos(z) + C * np.sin(z)
    den = 1.0 + D * np.cos(z) + E * np.sin(z)
    return num / den

def pade_21(z, A, B, C, D, E, F, G):
    """[2,1] Padé: (A + B*cos(z) + C*sin(z) + D*cos(2z) + E*sin(2z)) / (1 + F*cos(z) + G*sin(z))"""
    num = A + B * np.cos(z) + C * np.sin(z) + D * np.cos(2*z) + E * np.sin(2*z)
    den = 1.0 + F * np.cos(z) + G * np.sin(z)
    return num / den

z_test = np.linspace(0, 2*np.pi, 500)

pade_results = []
for rank in range(min(50, len(gw))):
    w_tot, g = gw[rank]
    c0, harms = get_fourier_coeffs(g, coef_a)
    f_exact = fourier_eval(c0, harms, z_test)

    # [1,1] Padé (5 params)
    try:
        p0 = [c0, harms[0][0], harms[0][1], 0, 0]
        popt11, _ = curve_fit(pade_rational_form, z_test, f_exact, p0=p0, maxfev=10000)
        f_11 = pade_rational_form(z_test, *popt11)
        err_11 = np.sqrt(np.mean((f_exact - f_11)**2)) / (np.std(f_exact) + 1e-20)
    except:
        err_11 = 999

    # [2,1] Padé (7 params)
    try:
        p0 = [c0, harms[0][0], harms[0][1],
              harms[1][0] if len(harms) > 1 else 0,
              harms[1][1] if len(harms) > 1 else 0, 0, 0]
        popt21, _ = curve_fit(pade_21, z_test, f_exact, p0=p0, maxfev=10000)
        f_21 = pade_21(z_test, *popt21)
        err_21 = np.sqrt(np.mean((f_exact - f_21)**2)) / (np.std(f_exact) + 1e-20)
    except:
        err_21 = 999

    # Truncated Fourier with only k=1,2 (5 params)
    f_trunc2 = c0
    for k in range(1, min(3, n_harm + 1)):
        f_trunc2 = f_trunc2 + harms[k-1][0] * np.cos(k * z_test) + harms[k-1][1] * np.sin(k * z_test)
    err_trunc2 = np.sqrt(np.mean((f_exact - f_trunc2)**2)) / (np.std(f_exact) + 1e-20)

    # Truncated Fourier with only k=1 (3 params)
    f_trunc1 = c0 + harms[0][0] * np.cos(z_test) + harms[0][1] * np.sin(z_test)
    err_trunc1 = np.sqrt(np.mean((f_exact - f_trunc1)**2)) / (np.std(f_exact) + 1e-20)

    pade_results.append({
        'rank': rank, 'label': group_label(g), 'weight': w_tot,
        'err_11': err_11, 'err_21': err_21,
        'err_trunc1': err_trunc1, 'err_trunc2': err_trunc2,
        'n_harm_full': n_harm
    })

# Print results
print(f"\n  {'Rank':>4s} {'Label':30s} {'|c|':>8s} {'Trunc1':>8s} {'Trunc2':>8s} {'[1,1]':>8s} {'[2,1]':>8s}")
for pr in pade_results[:30]:
    print(f"  {pr['rank']+1:4d} {pr['label']:30s} {pr['weight']:8.1f} "
          f"{pr['err_trunc1']:8.4f} {pr['err_trunc2']:8.4f} "
          f"{pr['err_11']:8.4f} {pr['err_21']:8.4f}")

# Aggregate: how many groups achieve <1%, <5%, <10% error with Padé
for threshold in [0.01, 0.05, 0.10]:
    n11 = sum(1 for pr in pade_results if pr['err_11'] < threshold)
    n21 = sum(1 for pr in pade_results if pr['err_21'] < threshold)
    nt1 = sum(1 for pr in pade_results if pr['err_trunc1'] < threshold)
    nt2 = sum(1 for pr in pade_results if pr['err_trunc2'] < threshold)
    print(f"  Groups with rel error < {threshold:.0%}: Trunc(k=1)={nt1}/{len(pade_results)}, "
          f"Trunc(k<=2)={nt2}/{len(pade_results)}, [1,1]={n11}/{len(pade_results)}, [2,1]={n21}/{len(pade_results)}")

# Plot: Padé accuracy comparison
fig, ax = plt.subplots(figsize=(6, 4))
ranks = [pr['rank']+1 for pr in pade_results[:30]]
ax.semilogy(ranks, [pr['err_trunc1'] for pr in pade_results[:30]], 's-', ms=3, lw=0.8, label='Trunc $k{=}1$ (3 params)')
ax.semilogy(ranks, [pr['err_trunc2'] for pr in pade_results[:30]], 'D-', ms=3, lw=0.8, label='Trunc $k{\\leq}2$ (5 params)')
ax.semilogy(ranks, [pr['err_11'] for pr in pade_results[:30]], 'o-', ms=3, lw=0.8, label='Pad\\\'e [1,1] (5 params)')
ax.semilogy(ranks, [pr['err_21'] for pr in pade_results[:30]], '^-', ms=3, lw=0.8, label='Pad\\\'e [2,1] (7 params)')
ax.axhline(0.01, color='green', ls='--', lw=0.8, alpha=0.5, label='1% error')
ax.axhline(0.05, color='orange', ls=':', lw=0.8, alpha=0.5, label='5% error')
ax.set_xlabel('Group rank (by weight)')
ax.set_ylabel('Relative RMS error')
ax.set_title('Fourier Compactification: Pad\\\'e vs Truncation', fontweight='bold')
ax.legend(fontsize=6, ncol=2)
ax.set_ylim(1e-5, 2)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pade_comparison.pdf'))
fig.savefig(os.path.join(OUTDIR, 'pade_comparison.png'))
plt.close(fig)
print("  Saved pade_comparison.pdf")

# Plot: example Padé fits for top 6 groups
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
axes = axes.ravel()
for i in range(6):
    w_tot, g = gw[i]
    c0, harms = get_fourier_coeffs(g, coef_a)
    f_exact = fourier_eval(c0, harms, z_test)

    try:
        p0 = [c0, harms[0][0], harms[0][1],
              harms[1][0] if len(harms)>1 else 0,
              harms[1][1] if len(harms)>1 else 0, 0, 0]
        popt21, _ = curve_fit(pade_21, z_test, f_exact, p0=p0, maxfev=10000)
        f_21 = pade_21(z_test, *popt21)
    except:
        f_21 = np.zeros_like(z_test)

    f_t1 = c0 + harms[0][0]*np.cos(z_test) + harms[0][1]*np.sin(z_test)

    ax = axes[i]
    ax.plot(z_test, f_exact, 'k-', lw=1.0, label='Full (15 par)')
    ax.plot(z_test, f_21, 'r--', lw=0.8, label='[2,1] (7 par)')
    ax.plot(z_test, f_t1, 'b:', lw=0.8, label='$k{=}1$ (3 par)')
    ax.set_title(f'{i+1}. {group_label(g)}', fontsize=7)
    if i >= 3: ax.set_xlabel('$\\zeta$', fontsize=8)
    if i % 3 == 0: ax.set_ylabel('$\\delta\\xi$', fontsize=8)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)

fig.suptitle('Pad\\\'e [2,1] Approximation of Fourier Sums (Top 6 Groups)', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pade_examples.pdf'))
fig.savefig(os.path.join(OUTDIR, 'pade_examples.png'))
plt.close(fig)
print("  Saved pade_examples.pdf")


# ====================================================================
# STEP 3: (1-e^2)^{-p} extraction
# ====================================================================
print("\n" + "="*70)
print("STEP 3: (1-e^2)^{-p} EXTRACTION")
print("="*70)

# For each (b, c, ds, da) sector, collect the k=0 (base) coefficients
# across eccentricity powers a=1..5.
# If (1-e^2)^{-p} is the source, the even-power pattern should show:
# ratio c(a=3)/c(a=1) ~ p  (from the e^2 term in the expansion)
# ratio c(a=5)/c(a=1) ~ p(p+1)/2  (from the e^4 term)

# But note: the base monomial is e^a, not e^{2n}. So we need to check
# the coefficients for a=1,3,5 (odd) separately from a=2,4 (even).
# The (1-e^2)^{-p} factor produces:
# e * (1-e^2)^{-p} = e + p*e^3 + p(p+1)/2 * e^5 + ...
# So for odd powers: c(a=3)/c(a=1) ~ p, c(a=5)/c(a=1) ~ p(p+1)/2

sectors = {}
for g in groups:
    key = (g['b'], g['c'], g['ds'], g['da'])
    if key not in sectors:
        sectors[key] = {}
    sectors[key][g['a']] = g

resum_results = []
print(f"\n  {'Sector':35s} {'c(e1)':>8s} {'c(e3)':>8s} {'c(e5)':>8s} {'p_fit':>7s} {'p_check':>8s} {'quality':>8s}")

for key in sorted(sectors.keys()):
    sg = sectors[key]
    if 1 not in sg or 3 not in sg:
        continue

    b, c, ds, da = key
    label = f"x^{b} nu^{c}" + (f" chiS" if ds else "") + (f" chiA" if da else "")

    # Check for each harmonic k separately (focus on k=1 which dominates)
    for k_check in [0, 1]:
        coeffs_by_a = {}
        for a_val, g_val in sg.items():
            c0, harms = get_fourier_coeffs(g_val, coef_a)
            if k_check == 0:
                coeffs_by_a[a_val] = c0
            else:
                coeffs_by_a[a_val] = np.sqrt(harms[k_check-1][0]**2 + harms[k_check-1][1]**2)

        c1 = coeffs_by_a.get(1, 0)
        c3 = coeffs_by_a.get(3, 0)
        c5 = coeffs_by_a.get(5, 0)

        if abs(c1) < 0.1:
            continue

        # From e*(1-e^2)^{-p}: c3/c1 = p, c5/c1 = p(p+1)/2
        p_from_31 = c3 / c1 if abs(c1) > 0 else 0
        p_from_51 = 0
        quality = 'N/A'

        if abs(c5) > 0.01 and abs(c1) > 0.1:
            # c5/c1 = p(p+1)/2, so p^2 + p - 2*c5/c1 = 0
            r51 = c5 / c1
            disc = 1 + 8 * r51
            if disc > 0:
                p_from_51 = (-1 + np.sqrt(disc)) / 2
                # Check consistency
                if abs(p_from_31) > 0:
                    quality = f"{abs(p_from_51 - p_from_31) / (abs(p_from_31) + 0.1):.2f}"

        if k_check <= 1 and abs(p_from_31) > 0.05:
            tag = f"k={k_check}"
            resum_results.append({
                'sector': label, 'k': k_check,
                'c1': c1, 'c3': c3, 'c5': c5,
                'p_31': p_from_31, 'p_51': p_from_51, 'quality': quality,
                'b': b
            })
            if k_check == 1:
                print(f"  {label+' '+tag:35s} {c1:8.3f} {c3:8.3f} {c5:8.3f} "
                      f"{p_from_31:7.3f} {p_from_51:8.3f} {quality:>8s}")

# Expected p values from PN theory
print("\n  Expected (1-e^2)^{-p} exponents from PN:")
print("    0PN: p = 1   (leading term denominator)")
print("    1PN: p = 2   (1PN correction)")
print("    1.5PN: p = 7/2 = 3.5 (tail terms)")
print("    2PN: p = 3   (2PN correction)")

# Plot the p values vs PN order (x power)
fig, ax = plt.subplots(figsize=(5, 3.5))
pn_expected = {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.5}
for rr in resum_results:
    if rr['k'] == 1 and abs(rr['p_31']) > 0.1 and abs(rr['p_31']) < 10:
        ax.plot(rr['b'], rr['p_31'], 'o', ms=5, color='steelblue', alpha=0.5)

# Expected line
xpn = list(pn_expected.keys())
ppn = list(pn_expected.values())
ax.plot(xpn, ppn, 'rs-', ms=8, lw=1.5, label='PN expected', zorder=5)
ax.set_xlabel('$x$ power ($b$)')
ax.set_ylabel('Fitted $p$ from $c(e^3)/c(e^1)$')
ax.set_title('$(1-e^2)^{-p}$ Exponent: Fitted vs PN Expected', fontweight='bold')
ax.legend()
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['0PN', '1PN', '2PN', '3PN'])
ax.set_ylim(-1, 8)
ax.axhline(0, color='gray', ls='-', lw=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'resum_p_values.pdf'))
fig.savefig(os.path.join(OUTDIR, 'resum_p_values.png'))
plt.close(fig)
print("  Saved resum_p_values.pdf")


# ====================================================================
# STEP 4: Direct PN comparison
# ====================================================================
print("\n" + "="*70)
print("STEP 4: DIRECT PN COMPARISON")
print("="*70)

# The ansatz in fit.py is:
# h22_ecc = leading(e, zeta) + x*e/(1-e^2)^2 * C(zeta, e, nu)
# xi_amp_ansatz = |h22_ecc| - 1

# The FULL expression in EOB_modes.dat.m goes to 2.5PN.
# We compare the Ridge residual against what we'd expect from the missing PN orders.

# First: compute the ansatz xi_amp Fourier content at several (e, x, nu) points
# and compare the Ridge residual prediction

def h22_ecc_ansatz(x, e, zeta, nu):
    e2 = e*e; e3 = e2*e
    eiz = np.exp(1j*zeta); emiz = np.exp(-1j*zeta)
    leading = (4 + 2*e2*eiz**2 + e*emiz + 5*e*eiz) / (4*(1-e2))
    tc = e*(26*nu/7 - 559/84)
    tem2 = e*np.exp(-2j*zeta)*(15*nu/14 - 95/168)
    tem3 = e2*np.exp(-3j*zeta)*(9*nu/56 + 1/112)
    te3 = e2*np.exp(3j*zeta)*(nu/8 - 49/48)
    te2 = np.exp(2j*zeta)*(e3*(6*nu/7-41/21)+e*(nu/14-153/56))
    tem = emiz*(e2*(7*nu/8-59/48)+27*nu/14-23/14)
    tei = eiz*(e2*(143*nu/56-2071/336)+nu/14-13/7)
    curly = tc+tem3+te3+tem2+te2+tem+tei
    pa = (x*e)/(1-e2)**2 * curly
    return leading + pa

# Compare: at fixed e, nu, what does the Ridge predict vs what the ansatz misses?
# The residual at chi_S=chi_A=0 should match the missing PN content

# Evaluate the Ridge basis at specific points
def ridge_predict_single(e_val, x_val, z_arr, nu_val, chiS_val, chiA_val):
    """Evaluate the Ridge residual at given parameter values."""
    n = len(z_arr)
    features = []
    for g in groups:
        base = e_val**g['a'] * x_val**g['b'] * nu_val**g['c'] * chiS_val**g['ds'] * chiA_val**g['da']
        features.append(np.full(n, base))
        for k in range(1, n_harm + 1):
            features.append(base * np.cos(k * z_arr))
            features.append(base * np.sin(k * z_arr))
    X = np.column_stack(features)
    dxi_a = X @ coef_a
    dxi_w = X @ coef_w
    return dxi_a, dxi_w

# Compare xi_total = xi_ansatz + delta_xi  at various eccentricities
z_arr = np.linspace(0, 2*np.pi, 500)
nu_val = 0.25  # equal mass
x_val = 0.05   # moderate frequency

fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
e_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

for i, e_val in enumerate(e_vals):
    ax = axes.ravel()[i]

    # Ansatz
    h_ecc = h22_ecc_ansatz(x_val, e_val, z_arr, nu_val)
    xi_ansatz = np.abs(h_ecc) - 1.0

    # Ridge residual (non-spin)
    dxi_a, dxi_w = ridge_predict_single(e_val, x_val, z_arr, nu_val, 0.0, 0.0)

    # Total
    xi_total = xi_ansatz + dxi_a

    ax.plot(z_arr / np.pi, xi_ansatz, 'b-', lw=1.0, label='Ansatz (0+1PN)')
    ax.plot(z_arr / np.pi, dxi_a, 'r--', lw=0.8, label='Ridge residual')
    ax.plot(z_arr / np.pi, xi_total, 'k-', lw=0.6, label='Total')
    ax.set_title(f'$e = {e_val}$, $x = {x_val}$, $\\nu = {nu_val}$', fontsize=7)
    if i >= 3: ax.set_xlabel('$\\zeta / \\pi$', fontsize=8)
    if i % 3 == 0: ax.set_ylabel('$\\xi_{\\rm amp}$', fontsize=8)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)

fig.suptitle('Ansatz vs Ridge Residual vs Total (non-spinning, $\\chi_S = \\chi_A = 0$)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pn_comparison_xi.pdf'))
fig.savefig(os.path.join(OUTDIR, 'pn_comparison_xi.png'))
plt.close(fig)
print("  Saved pn_comparison_xi.pdf")

# Spin effect comparison
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
e_val = 0.3
x_val = 0.05
nu_val = 0.25
chi_vals = [(-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5), (0.3, 0.3), (-0.3, 0.3)]

for i, (chiS, chiA) in enumerate(chi_vals):
    ax = axes.ravel()[i]
    dxi_nospin, _ = ridge_predict_single(e_val, x_val, z_arr, nu_val, 0.0, 0.0)
    dxi_spin, _ = ridge_predict_single(e_val, x_val, z_arr, nu_val, chiS, chiA)
    dxi_diff = dxi_spin - dxi_nospin

    ax.plot(z_arr / np.pi, dxi_nospin, 'b-', lw=0.8, label='Non-spin residual')
    ax.plot(z_arr / np.pi, dxi_spin, 'r--', lw=0.8, label=f'With spin')
    ax.plot(z_arr / np.pi, dxi_diff, 'g:', lw=1.0, label='Spin contribution')
    ax.set_title(f'$\\chi_S = {chiS}$, $\\chi_A = {chiA}$', fontsize=7)
    if i >= 3: ax.set_xlabel('$\\zeta / \\pi$', fontsize=8)
    if i % 3 == 0: ax.set_ylabel('$\\delta\\xi_{\\rm amp}$', fontsize=8)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)

fig.suptitle(f'Spin Contribution to Residual ($e={e_val}$, $x={x_val}$, $\\nu={nu_val}$)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'spin_contribution.pdf'))
fig.savefig(os.path.join(OUTDIR, 'spin_contribution.png'))
plt.close(fig)
print("  Saved spin_contribution.pdf")


# PN order decomposition of the residual
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
e_val = 0.3; nu_val = 0.25

for i, x_val in enumerate([0.01, 0.03, 0.06, 0.10]):
    ax = axes[i]

    # Decompose by x power
    for b_val in range(max_x + 1):
        dxi_b = np.zeros_like(z_arr)
        for g in groups:
            if g['b'] != b_val or g['ds'] != 0 or g['da'] != 0:
                continue
            s = g['start']
            base = e_val**g['a'] * x_val**g['b'] * nu_val**g['c']
            dxi_b += coef_a[s] * base
            for k in range(1, n_harm + 1):
                dxi_b += coef_a[s + 2*k-1] * base * np.cos(k * z_arr)
                dxi_b += coef_a[s + 2*k] * base * np.sin(k * z_arr)

        pn_label = ['0PN', '1PN', '2PN', '3PN'][b_val]
        ax.plot(z_arr / np.pi, dxi_b, lw=0.8, label=f'{pn_label} ($x^{b_val}$)')

    # Total non-spin
    dxi_tot, _ = ridge_predict_single(e_val, x_val, z_arr, nu_val, 0.0, 0.0)
    ax.plot(z_arr / np.pi, dxi_tot, 'k-', lw=1.2, label='Total')

    ax.set_title(f'$x = {x_val}$', fontsize=8)
    ax.set_xlabel('$\\zeta / \\pi$', fontsize=8)
    if i == 0: ax.set_ylabel('$\\delta\\xi_{\\rm amp}$', fontsize=8)
    ax.legend(fontsize=4, ncol=2)
    ax.tick_params(labelsize=5)

fig.suptitle(f'PN Order Decomposition of Non-Spin Residual ($e={e_val}$, $\\nu={nu_val}$)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pn_order_decomposition.pdf'))
fig.savefig(os.path.join(OUTDIR, 'pn_order_decomposition.png'))
plt.close(fig)
print("  Saved pn_order_decomposition.pdf")

# Ratio c_w/c_a as a function of basis terms (testing Relation III)
fig, ax = plt.subplots(figsize=(6, 4))
ratios_nonspin = []
ratios_spin = []
weights_ns = []
weights_sp = []

for g in groups:
    s = g['start']
    for local in range(g['n_feat']):
        ca = coef_a[s + local]
        cw = coef_w[s + local]
        if abs(ca) > 1.0:
            r = cw / ca
            if g['ds'] == 0 and g['da'] == 0:
                ratios_nonspin.append(r)
                weights_ns.append(abs(ca))
            else:
                ratios_spin.append(r)
                weights_sp.append(abs(ca))

bins = np.linspace(0, 3, 60)
ax.hist(ratios_nonspin, bins=bins, weights=weights_ns, alpha=0.6, color='steelblue',
        label=f'Non-spin (median={np.median(ratios_nonspin):.2f})', edgecolor='0.3', lw=0.3)
ax.hist(ratios_spin, bins=bins, weights=weights_sp, alpha=0.5, color='firebrick',
        label=f'Spin (median={np.median(ratios_spin):.2f})', edgecolor='0.3', lw=0.3)
ax.axvline(1/0.9, color='green', ls='--', lw=1.5, label='$1/B = 1/0.9 = 1.11$ (Rel. III)')
ax.set_xlabel('$c_\\omega / c_{\\rm amp}$')
ax.set_ylabel('Weighted count')
ax.set_title('Amplitude--Frequency Coupling Ratio Distribution', fontweight='bold')
ax.legend(fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'relation_iii_ratio.pdf'))
fig.savefig(os.path.join(OUTDIR, 'relation_iii_ratio.png'))
plt.close(fig)
print("  Saved relation_iii_ratio.pdf")


# ====================================================================
# STEP 5: Validation-set accuracy of compactified model
# ====================================================================
print("\n" + "="*70)
print("STEP 5: COMPACTIFIED MODEL ACCURACY")
print("="*70)

# Load validation data
val_path = os.path.join(RESULTS, 'validation_data.pkl')
with open(val_path, 'rb') as f:
    val_data = pickle.load(f)

train_path = os.path.join(RESULTS, 'training_data.pkl')
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

print(f"  Loaded {len(train_data)} train + {len(val_data)} val waveforms")

# Build basis at data points and evaluate with truncated harmonics
def build_basis_truncated(e, z, x, nu, chi_S, chi_A, k_max=7):
    """Build basis with at most k_max harmonics."""
    features = []
    for g in groups:
        a, b, c, ds, da = g['a'], g['b'], g['c'], g['ds'], g['da']
        base = e**a * x**b * nu**c * chi_S**ds * chi_A**da
        features.append(base)
        for k in range(1, min(k_max, n_harm) + 1):
            features.append(base * np.cos(k * z))
            features.append(base * np.sin(k * z))
        # Zero-pad remaining harmonics
        for k in range(min(k_max, n_harm) + 1, n_harm + 1):
            features.append(np.zeros_like(e))
            features.append(np.zeros_like(e))
    return np.column_stack(features)

# Compute residual R^2 for different truncation levels
from sklearn.metrics import r2_score

# Gather all validation points
all_e, all_x, all_z, all_nu, all_chiS, all_chiA = [], [], [], [], [], []
all_dya, all_dyw = [], []
n_pts = 200
for d in val_data:
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
    all_e.append(e_[idx]); all_x.append(x_[idx]); all_z.append(z_[idx])
    all_nu.append(np.full(len(idx), d['nu']))
    all_chiS.append(np.full(len(idx), d['chi_S']))
    all_chiA.append(np.full(len(idx), d['chi_A']))
    all_dya.append(d['delta_xi_amp'][idx])
    all_dyw.append(d['delta_xi_omega'][idx])

e_all = np.concatenate(all_e); x_all = np.concatenate(all_x)
z_all = np.concatenate(all_z); nu_all = np.concatenate(all_nu)
chiS_all = np.concatenate(all_chiS); chiA_all = np.concatenate(all_chiA)
dya_all = np.concatenate(all_dya); dyw_all = np.concatenate(all_dyw)

print(f"  Validation points: {len(e_all)}")

# Full model prediction
X_full = build_basis_truncated(e_all, z_all, x_all, nu_all, chiS_all, chiA_all, k_max=7)
pred_a_full = X_full @ coef_a
pred_w_full = X_full @ coef_w
r2_a_full = r2_score(dya_all, pred_a_full)
r2_w_full = r2_score(dyw_all, pred_w_full)
print(f"  Full model (k<=7): R2_amp={r2_a_full:.6f}, R2_omega={r2_w_full:.6f}")

# Truncated models
trunc_results = []
for k_max in [1, 2, 3, 4, 5, 6, 7]:
    # Zero out coefficients for k > k_max
    coef_a_trunc = coef_a.copy()
    coef_w_trunc = coef_w.copy()
    for g in groups:
        s = g['start']
        for k in range(k_max + 1, n_harm + 1):
            coef_a_trunc[s + 2*k - 1] = 0
            coef_a_trunc[s + 2*k] = 0
            coef_w_trunc[s + 2*k - 1] = 0
            coef_w_trunc[s + 2*k] = 0

    pred_a_t = X_full @ coef_a_trunc
    pred_w_t = X_full @ coef_w_trunc
    r2_a_t = r2_score(dya_all, pred_a_t)
    r2_w_t = r2_score(dyw_all, pred_w_t)
    n_params = len(groups) * (1 + 2 * k_max)
    trunc_results.append({
        'k_max': k_max, 'r2_a': r2_a_t, 'r2_w': r2_w_t, 'n_params': n_params
    })
    print(f"  Trunc k<={k_max}: R2_amp={r2_a_t:.6f}, R2_omega={r2_w_t:.6f}, params={n_params}")

# Plot R^2 vs number of harmonics
fig, ax = plt.subplots(figsize=(5, 3.5))
ks = [tr['k_max'] for tr in trunc_results]
r2s_a = [tr['r2_a'] for tr in trunc_results]
r2s_w = [tr['r2_w'] for tr in trunc_results]
ax.plot(ks, r2s_a, 'o-', ms=5, lw=1.2, color='steelblue', label='$R^2$ amp')
ax.plot(ks, r2s_w, 's-', ms=5, lw=1.2, color='firebrick', label='$R^2$ omega')
ax.set_xlabel('Max harmonic $k_{\\rm max}$')
ax.set_ylabel('$R^2$ on validation set')
ax.set_title('Accuracy vs Fourier Truncation Level', fontweight='bold')
ax.legend()
ax.set_ylim(min(r2s_a + r2s_w) - 0.005, 1.001)
npar = [tr['n_params'] for tr in trunc_results]
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ks)
ax2.set_xticklabels([str(n) for n in npar], fontsize=6)
ax2.set_xlabel('Number of parameters', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'truncation_accuracy.pdf'))
fig.savefig(os.path.join(OUTDIR, 'truncation_accuracy.png'))
plt.close(fig)
print("  Saved truncation_accuracy.pdf")

print("\n" + "="*70)
print("ALL ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated plots in {OUTDIR}:")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith('.pdf') and f != 'interpret.pdf':
        print(f"  {f}")
