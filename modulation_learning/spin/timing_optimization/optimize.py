"""
Timing optimization for the best modulation learning model.
Target: 10-20 ms end-to-end.

Benchmarks, applies successive optimizations (Numba basis, downsample,
fast reconstruction), verifies accuracy, generates plots/summaries.

Usage:
    conda activate kitp-py310
    cd modulation_learning/spin_05_04_26
    python timing_optimization/optimize.py
"""
import sys, os, time, json, pickle
import numpy as np
import numba as nb
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DYN_SRC = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/dyn_rewrite/src'
sys.path.insert(0, DYN_SRC)
from dynamics import setup_and_integrate

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_MOD = os.path.join(BASE, 'results')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OPTMODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimized_model')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(OPTMODEL, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.labelsize': 11, 'axes.titlesize': 10, 'legend.fontsize': 8,
    'legend.frameon': False, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.0, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# ====================================================================
# Ansatz — numpy (original) and numba
# ====================================================================
def h22_ecc_ansatz_np(x, e, zeta, nu):
    e2 = e * e; e3 = e2 * e
    eiz = np.exp(1j * zeta); emiz = np.exp(-1j * zeta)
    leading = (4.0 + 2.0 * e2 * eiz**2 + e * emiz + 5.0 * e * eiz) / (4.0 * (1.0 - e2))
    tc = e * (26.0 * nu / 7.0 - 559.0 / 84.0)
    tem2 = e * np.exp(-2j * zeta) * (15.0 * nu / 14.0 - 95.0 / 168.0)
    tem3 = e2 * np.exp(-3j * zeta) * (9.0 * nu / 56.0 + 1.0 / 112.0)
    te3 = e2 * np.exp(3j * zeta) * (nu / 8.0 - 49.0 / 48.0)
    te2 = np.exp(2j * zeta) * (e3 * (6.0 * nu / 7.0 - 41.0 / 21.0)
                                + e * (nu / 14.0 - 153.0 / 56.0))
    tem = emiz * (e2 * (7.0 * nu / 8.0 - 59.0 / 48.0) + 27.0 * nu / 14.0 - 23.0 / 14.0)
    tep = eiz * (e2 * (143.0 * nu / 56.0 - 2071.0 / 336.0) + nu / 14.0 - 13.0 / 7.0)
    curly = tc + tem3 + te3 + tem2 + te2 + tem + tep
    pa = (x * e) / (1.0 - e2)**2 * curly
    return leading + pa


# ====================================================================
# Original (slow) basis
# ====================================================================
def build_basis_original(e, z, x, nu, chi_S, chi_A, max_e=4, max_x=3,
                         max_nu=2, max_chi=1, n_harm=5):
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
# Numba-optimized: precompute specs, JIT basis, JIT predict, JIT ansatz
# ====================================================================
def precompute_column_specs(max_e=4, max_x=3, max_nu=2, max_chi=1, n_harm=5):
    specs = []
    for a in range(1, max_e + 1):
        for b in range(max_x + 1):
            for c in range(max_nu + 1):
                for d_s in range(max_chi + 1):
                    for d_a in range(max_chi + 1):
                        if a + b + c + d_s + d_a > max_e + 3:
                            continue
                        specs.append((a, b, c, d_s, d_a, 0, 0))
                        for k in range(1, n_harm + 1):
                            specs.append((a, b, c, d_s, d_a, k, 0))
                            specs.append((a, b, c, d_s, d_a, k, 1))
    return np.array(specs, dtype=np.int32)


@nb.njit(cache=True)
def _pow_int(base, exp):
    r = 1.0
    for _ in range(exp):
        r *= base
    return r


@nb.njit(cache=True)
def build_basis_numba(e, z, x, nu, chi_S, chi_A, specs):
    n = len(e); nc = specs.shape[0]
    B = np.empty((n, nc))
    for i in range(n):
        ei = e[i]; xi = x[i]; zi = z[i]
        nui = nu[i]; csi = chi_S[i]; cai = chi_A[i]
        # Precompute trig for all harmonics needed (max k = max in specs col 5)
        for j in range(nc):
            a = specs[j, 0]; b = specs[j, 1]; c = specs[j, 2]
            ds = specs[j, 3]; da = specs[j, 4]
            k = specs[j, 5]; is_sin = specs[j, 6]
            val = _pow_int(ei, a)
            if b > 0: val *= _pow_int(xi, b)
            if c > 0: val *= _pow_int(nui, c)
            if ds > 0: val *= _pow_int(csi, ds)
            if da > 0: val *= _pow_int(cai, da)
            if k > 0:
                angle = k * zi
                if is_sin:
                    val *= np.sin(angle)
                else:
                    val *= np.cos(angle)
            B[i, j] = val
    return B


@nb.njit(cache=True)
def predict_numba(B, coefs_a, coefs_w):
    n = B.shape[0]; m = B.shape[1]
    out_a = np.zeros(n); out_w = np.zeros(n)
    for i in range(n):
        sa = 0.0; sw = 0.0
        for j in range(m):
            bij = B[i, j]
            sa += bij * coefs_a[j]
            sw += bij * coefs_w[j]
        out_a[i] = sa; out_w[i] = sw
    return out_a, out_w


@nb.njit(cache=True)
def ansatz_numba(x, e, zeta, nu):
    """Returns (xi_amp_ansatz, xi_omega_ansatz) arrays."""
    n = len(x)
    xa = np.empty(n); xw = np.empty(n)
    for i in range(n):
        ei = e[i]; xi = x[i]; zi = zeta[i]; nui = nu[i]
        e2 = ei * ei; e3 = e2 * ei
        cz = np.cos(zi); sz = np.sin(zi)
        c2z = np.cos(2.0 * zi); s2z = np.sin(2.0 * zi)
        c3z = np.cos(3.0 * zi); s3z = np.sin(3.0 * zi)
        denom = 4.0 * (1.0 - e2)
        nr = 4.0 + 2.0*e2*c2z + ei*cz + 5.0*ei*cz
        ni = 2.0*e2*s2z + ei*(-sz) + 5.0*ei*sz
        lr = nr / denom; li = ni / denom
        tc_r = ei * (26.0*nui/7.0 - 559.0/84.0)
        tem2_c = 15.0*nui/14.0 - 95.0/168.0
        tem2_r = ei * c2z * tem2_c; tem2_i = ei * (-s2z) * tem2_c
        tem3_c = 9.0*nui/56.0 + 1.0/112.0
        tem3_r = e2 * c3z * tem3_c; tem3_i = e2 * (-s3z) * tem3_c
        te3_c = nui/8.0 - 49.0/48.0
        te3_r = e2 * c3z * te3_c; te3_i = e2 * s3z * te3_c
        c_te2 = e3*(6.0*nui/7.0 - 41.0/21.0) + ei*(nui/14.0 - 153.0/56.0)
        te2_r = c2z * c_te2; te2_i = s2z * c_te2
        c_tem = e2*(7.0*nui/8.0 - 59.0/48.0) + 27.0*nui/14.0 - 23.0/14.0
        tem_r = cz * c_tem; tem_i = (-sz) * c_tem
        c_tep = e2*(143.0*nui/56.0 - 2071.0/336.0) + nui/14.0 - 13.0/7.0
        tep_r = cz * c_tep; tep_i = sz * c_tep
        cur_r = tc_r + tem3_r + te3_r + tem2_r + te2_r + tem_r + tep_r
        cur_i = tem3_i + te3_i + tem2_i + te2_i + tem_i + tep_i
        pf = (xi * ei) / ((1.0 - e2) * (1.0 - e2))
        hr = lr + pf * cur_r; hi = li + pf * cur_i
        amp = np.sqrt(hr*hr + hi*hi)
        xa[i] = amp - 1.0
        xw[i] = (amp - 1.0) / 0.9
    return xa, xw


@nb.njit(cache=True)
def full_predict_ds(e_full, x_full, z_full, nu_val, chiS_val, chiA_val,
                    specs, coefs_a, coefs_w, n_ds):
    """Downsample, predict with Numba, interpolate back. Single function."""
    n = len(e_full)
    nds = min(n, n_ds)
    # Downsample indices
    idx = np.empty(nds, dtype=nb.int64)
    for i in range(nds):
        idx[i] = int(i * (n - 1) / (nds - 1) + 0.5) if nds > 1 else 0
    # Extract downsampled
    e_ds = np.empty(nds); x_ds = np.empty(nds); z_ds = np.empty(nds)
    nu_ds = np.empty(nds); cs_ds = np.empty(nds); ca_ds = np.empty(nds)
    for i in range(nds):
        e_ds[i] = e_full[idx[i]]; x_ds[i] = x_full[idx[i]]; z_ds[i] = z_full[idx[i]]
        nu_ds[i] = nu_val; cs_ds[i] = chiS_val; ca_ds[i] = chiA_val
    # Ansatz
    xa_ds, xw_ds = ansatz_numba(x_ds, e_ds, z_ds, nu_ds)
    # Basis + Ridge
    B = build_basis_numba(e_ds, z_ds, x_ds, nu_ds, cs_ds, ca_ds, specs)
    da, dw = predict_numba(B, coefs_a, coefs_w)
    # Total modulation at downsampled points
    ma = np.empty(nds); mw = np.empty(nds)
    for i in range(nds):
        ma[i] = xa_ds[i] + da[i]
        mw[i] = xw_ds[i] + dw[i]
    # Linear interpolation back to full grid
    xi_amp = np.empty(n); xi_omega = np.empty(n)
    j = 0
    for i in range(n):
        while j < nds - 2 and idx[j + 1] <= i:
            j += 1
        if j >= nds - 1:
            xi_amp[i] = ma[nds - 1]; xi_omega[i] = mw[nds - 1]
        else:
            i0 = idx[j]; i1 = idx[j + 1]
            if i1 == i0:
                xi_amp[i] = ma[j]; xi_omega[i] = mw[j]
            else:
                frac = float(i - i0) / float(i1 - i0)
                xi_amp[i] = ma[j] + frac * (ma[j + 1] - ma[j])
                xi_omega[i] = mw[j] + frac * (mw[j + 1] - mw[j])
    return xi_amp, xi_omega


# ====================================================================
# Reconstruction helpers
# ====================================================================
def smooth_taper(t, ts=-50.0, te=0.0):
    w = np.ones_like(t)
    m = (t >= ts) & (t <= te)
    w[m] = 0.5 * (1 + np.cos(np.pi * (t[m] - ts) / (te - ts)))
    w[t > te] = 0
    return w


def reconstruct_fast(d, xi_amp, xi_omega, dt=0.5):
    """Fast reconstruction using np.interp (no CubicSpline) and coarser dt."""
    t_d = np.arange(d['t'][0], d['t'][-1], dt)
    # Interp (much faster than CubicSpline)
    h_cir_r = np.interp(t_d, d['t'], np.real(d['h_cir']))
    h_cir_i = np.interp(t_d, d['t'], np.imag(d['h_cir']))
    h_ecc_r = np.interp(t_d, d['t'], np.real(d['h_ecc']))
    h_ecc_i = np.interp(t_d, d['t'], np.imag(d['h_ecc']))
    xi_a_d = np.interp(t_d, d['t'], xi_amp)
    xi_w_d = np.interp(t_d, d['t'], xi_omega)

    taper = smooth_taper(t_d)
    xi_a_d *= taper; xi_w_d *= taper

    A_cir = np.sqrt(h_cir_r**2 + h_cir_i**2)
    A_p = A_cir * (1 + xi_a_d)
    phi_cir = np.unwrap(np.arctan2(h_cir_i, h_cir_r))
    omega_cir = np.gradient(phi_cir, dt)
    phi_pred = cumulative_trapezoid(omega_cir * (1 + xi_w_d), dx=dt, initial=0.0)
    phi_ecc = np.unwrap(np.arctan2(h_ecc_i, h_ecc_r))
    phi_pred += phi_ecc[0] - phi_pred[0]

    # Phase correction
    dphi = phi_pred - phi_ecc
    T = (t_d - t_d[0]) / max(t_d[-1] - t_d[0], 1.0)
    A = np.column_stack([T**k for k in range(1, 6)])
    c = np.linalg.lstsq(A, dphi - dphi[0], rcond=None)[0]
    phi_pred -= A @ c

    h_pred = A_p * np.exp(1j * phi_pred)
    return h_pred, h_ecc_r + 1j * h_ecc_i, t_d


def reconstruct_original(d, xi_amp, xi_omega):
    """Original reconstruction (CubicSpline, dt=0.1) for accuracy reference."""
    dt = 0.1
    t_d = np.arange(d['t'][0], d['t'][-1], dt)
    h_cir_d = CubicSpline(d['t'], np.real(d['h_cir']))(t_d) + \
              1j * CubicSpline(d['t'], np.imag(d['h_cir']))(t_d)
    h_ecc_d = CubicSpline(d['t'], np.real(d['h_ecc']))(t_d) + \
              1j * CubicSpline(d['t'], np.imag(d['h_ecc']))(t_d)
    xi_a_d = np.interp(t_d, d['t'], xi_amp)
    xi_w_d = np.interp(t_d, d['t'], xi_omega)
    taper = smooth_taper(t_d)
    xi_a_d *= taper; xi_w_d *= taper
    A_p = np.abs(h_cir_d) * (1 + xi_a_d)
    pc = np.unwrap(np.angle(h_cir_d)); oc = np.gradient(pc, dt)
    pp = cumulative_trapezoid(oc * (1 + xi_w_d), dx=dt, initial=0.0)
    pe = np.unwrap(np.angle(h_ecc_d)); pp += pe[0] - pp[0]
    dphi = pp - pe; T = (t_d - t_d[0]) / max(t_d[-1] - t_d[0], 1.0)
    A = np.column_stack([T**k for k in range(1, 6)])
    c = np.linalg.lstsq(A, dphi - dphi[0], rcond=None)[0]
    pp -= A @ c
    h_pred = A_p * np.exp(1j * pp)
    h_ref = h_ecc_d
    return h_pred, h_ref, t_d


def mathcalE_error(h_ref, h):
    n1 = np.sum(np.abs(h_ref)**2); n2 = np.sum(np.abs(h)**2)
    s = np.real(np.sum(h_ref * np.conj(h)))
    return ((n1 + n2) - 2 * s) / (2 * n1) if n1 > 0 else 1.0


# ====================================================================
# End-to-end inference functions
# ====================================================================
def inference_baseline(d, model):
    """Original (unoptimized) end-to-end."""
    bc = model['bc']; m_a = model['m_a']; m_w = model['m_w']
    n = len(d['t'])

    t0 = time.perf_counter()
    ode = setup_and_integrate(d['q'], d['chi1'], d['chi2'], 20.0, 0.0, 3.5,
                              d['e0'], 0.0, rtol=1e-8)
    t_ode = time.perf_counter() - t0

    t1 = time.perf_counter()
    t_ode_al = ode['t'] + d['t_ecc_start'] - d['t_peak_ecc']
    valid = (d['t'] >= t_ode_al[0]) & (d['t'] <= t_ode_al[-1])
    e_o = np.zeros(n); x_o = np.zeros(n); z_o = np.zeros(n)
    if np.sum(valid) > 10:
        e_o[valid] = CubicSpline(t_ode_al, ode['e'])(d['t'][valid])
        x_o[valid] = CubicSpline(t_ode_al, ode['x'])(d['t'][valid])
        z_o[valid] = CubicSpline(t_ode_al, ode['zeta'])(d['t'][valid])
    e_o = np.clip(e_o, 1e-6, 0.95); x_o = np.clip(x_o, 1e-6, 0.5)
    t_interp = time.perf_counter() - t1

    t2 = time.perf_counter()
    xa = np.abs(h22_ecc_ansatz_np(x_o, e_o, z_o, d['nu'])) - 1.0
    xw = xa / 0.9
    B = build_basis_original(e_o, z_o, x_o, np.full(n, d['nu']),
                             np.full(n, d['chi_S']), np.full(n, d['chi_A']), **bc)
    da = m_a.predict(B); dw = m_w.predict(B)
    xi_amp = xa + da; xi_omega = xw + dw
    t_predict = time.perf_counter() - t2

    t3 = time.perf_counter()
    reconstruct_original(d, xi_amp, xi_omega)
    t_recon = time.perf_counter() - t3

    return {'t_ode': t_ode*1e3, 't_interp': t_interp*1e3,
            't_predict': t_predict*1e3, 't_recon': t_recon*1e3,
            't_total': (t_ode+t_interp+t_predict+t_recon)*1e3}


def inference_optimized(d, coefs_a, coefs_w, specs, n_ds=1000, recon_dt=0.5):
    """Fully optimized: Numba predict + downsample + fast recon."""
    n = len(d['t'])

    t0 = time.perf_counter()
    ode = setup_and_integrate(d['q'], d['chi1'], d['chi2'], 20.0, 0.0, 3.5,
                              d['e0'], 0.0, rtol=1e-8)
    t_ode = time.perf_counter() - t0

    t1 = time.perf_counter()
    t_ode_al = ode['t'] + d['t_ecc_start'] - d['t_peak_ecc']
    valid = (d['t'] >= t_ode_al[0]) & (d['t'] <= t_ode_al[-1])
    e_o = np.zeros(n); x_o = np.zeros(n); z_o = d['zeta'].copy()
    if np.sum(valid) > 10:
        e_o[valid] = np.interp(d['t'][valid], t_ode_al, ode['e'])
        x_o[valid] = np.interp(d['t'][valid], t_ode_al, ode['x'])
        z_o[valid] = np.interp(d['t'][valid], t_ode_al, ode['zeta'])
    e_o = np.clip(e_o, 1e-6, 0.95); x_o = np.clip(x_o, 1e-6, 0.5)
    t_interp = time.perf_counter() - t1

    t2 = time.perf_counter()
    xi_amp, xi_omega = full_predict_ds(e_o, x_o, z_o, d['nu'], d['chi_S'],
                                        d['chi_A'], specs, coefs_a, coefs_w, n_ds)
    t_predict = time.perf_counter() - t2

    t3 = time.perf_counter()
    reconstruct_fast(d, xi_amp, xi_omega, dt=recon_dt)
    t_recon = time.perf_counter() - t3

    return {'t_ode': t_ode*1e3, 't_interp': t_interp*1e3,
            't_predict': t_predict*1e3, 't_recon': t_recon*1e3,
            't_total': (t_ode+t_interp+t_predict+t_recon)*1e3}


def time_pyseobnr(q, chi1, chi2, e0, omega0=0.0085):
    from pyseobnr.generate_waveform import generate_modes_opt
    t0 = time.perf_counter()
    generate_modes_opt(q, chi1, chi2, omega0, eccentricity=e0, rel_anomaly=0.0,
                       approximant="SEOBNRv5EHM", debug=True,
                       settings={'use_wave_convention': True})
    return (time.perf_counter() - t0) * 1000


# ====================================================================
# Plotting
# ====================================================================
def make_all_plots(baseline, optimized, seob, opt_progress, val, idx_rep):
    # 1. Timing comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    n_c = len(idx_rep)
    x = np.arange(n_c)
    st = np.array([seob[i]['t_ms'] for i in range(n_c)])
    ot = np.array([optimized[i]['t_total'] for i in range(n_c)])
    order = np.argsort(st)
    ax.bar(x - 0.2, st[order], 0.35, color='#4c72b0', edgecolor='0.3', lw=0.4, label='pySEOBNR')
    ax.bar(x + 0.2, ot[order], 0.35, color='#c44e52', edgecolor='0.3', lw=0.4, label='Our model (optimized)')
    ax.axhline(np.median(st), color='#4c72b0', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(np.median(ot), color='#c44e52', ls=':', lw=0.8, alpha=0.5)
    ax.set_ylabel('Time [ms]'); ax.set_yscale('log')
    ax.set_title('End-to-end: our model vs pySEOBNR', fontsize=10, fontweight='bold')
    labels = [f'q={seob[i]["q"]:.0f} e={seob[i]["e0"]:.2f}' for i in order]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5, rotation=45, ha='right')
    ax.legend(fontsize=8); plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'timing_comparison.pdf')); plt.close(fig)

    # 2. Timing vs parameters (full val set)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    all_t = np.array([t['t_total'] for t in optimized])
    all_q = np.array([val[idx_rep[i]]['q'] for i in range(len(optimized))])
    all_e0 = np.array([val[idx_rep[i]]['e0'] for i in range(len(optimized))])
    all_chi = np.array([val[idx_rep[i]]['chi_eff'] for i in range(len(optimized))])
    all_wl = np.array([val[idx_rep[i]]['wf_length_M']/1e3 for i in range(len(optimized))])
    for ax_, xv, xl in zip(axes, [all_q, all_e0, all_chi, all_wl],
                            [r'$q$', r'$e_0$', r'$\chi_{\rm eff}$', r'Length [$10^3\,M$]']):
        sc = ax_.scatter(xv, all_t, c=all_wl, cmap='viridis', s=12, alpha=0.7,
                        edgecolors='0.3', linewidths=0.2)
        ax_.set_xlabel(xl); ax_.set_ylabel('Time [ms]')
        ax_.axhline(20, color='#2ca02c', ls='--', lw=0.8, alpha=0.5)
    axes[0].set_title('Optimized timing vs parameters', fontsize=9, fontweight='bold')
    plt.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'timing_vs_params.pdf')); plt.close(fig)

    # 3. Speedup histogram
    fig, ax = plt.subplots(figsize=(5, 3.5))
    speedups = np.array([seob[i]['t_ms'] / optimized[i]['t_total'] for i in range(n_c)])
    ax.hist(speedups, bins=15, color='#55a868', edgecolor='0.3', lw=0.4, alpha=0.85)
    ax.axvline(1.0, color='#c44e52', ls='--', lw=1.5, label='Break-even')
    ax.axvline(np.median(speedups), color='k', ls='-', lw=1.2,
               label=f'Median: {np.median(speedups):.1f}x')
    ax.set_xlabel('Speedup over pySEOBNR'); ax.set_ylabel('Count')
    ax.set_title('Speedup distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'speedup_histogram.pdf')); plt.close(fig)

    # 4. Optimization progress staircase
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [p['name'] for p in opt_progress]
    times = [p['median_ms'] for p in opt_progress]
    colors = ['#c44e52' if t > 20 else '#55a868' for t in times]
    bars = ax.bar(range(len(names)), times, color=colors, edgecolor='0.3', lw=0.4, width=0.6)
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width()/2, t + 1, f'{t:.1f}', ha='center', fontsize=7)
    ax.axhline(np.median([s['t_ms'] for s in seob]), color='#4c72b0', ls='--', lw=1.5,
               label=f'pySEOBNR median')
    ax.axhline(20, color='#2ca02c', ls=':', lw=1.2, label='20 ms target')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=20, ha='right')
    ax.set_ylabel('Median time [ms]')
    ax.set_title('Optimization progress', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'optimization_progress.pdf')); plt.close(fig)

    # 5. Breakdown pie charts
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax_, data, title in [(axes[0], baseline, 'Before'), (axes[1], optimized, 'After')]:
        vals = [np.median([t[k] for t in data]) for k in ['t_ode','t_interp','t_predict','t_recon']]
        labels = [f'ODE\n{vals[0]:.1f}ms', f'Interp\n{vals[1]:.1f}ms',
                  f'Predict\n{vals[2]:.1f}ms', f'Recon\n{vals[3]:.1f}ms']
        cols = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
        ax_.pie(vals, labels=labels, colors=cols, autopct='%1.0f%%',
                textprops={'fontsize': 7}, startangle=90)
        ax_.set_title(f'{title} (total: {sum(vals):.0f} ms)', fontsize=9, fontweight='bold')
    plt.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'breakdown_pie.pdf')); plt.close(fig)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    log("=" * 70)
    log("TIMING OPTIMIZATION — Target: 10-20 ms")
    log("=" * 70)

    log("\nLoading model and data...")
    with open(os.path.join(RESULTS_MOD, 'models', 'ridge_nh7_me5_mchi1_a1e-06', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(RESULTS_MOD, 'validation_data.pkl'), 'rb') as f:
        val = pickle.load(f)
    log(f"  {len(val)} val waveforms, model type: {model['type']}")

    bc = model['bc']
    coefs_a = model['m_a'].coef_.ravel().astype(np.float64)
    coefs_w = model['m_w'].coef_.ravel().astype(np.float64)
    specs = precompute_column_specs(**bc)
    log(f"  {len(coefs_a)} features, basis config: {bc}")

    # JIT warmup
    log("\nJIT warmup...")
    _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                            t_end=100, rtol=1e-4, max_steps=50)
    _ = setup_and_integrate(2.0, 0.1, -0.1, 18.0, 0.0, 3.8, 0.1, 0.0, rtol=1e-8)
    _e = np.array([0.1, 0.2]); _x = np.array([0.05, 0.1]); _z = np.array([0.5, 1.0])
    _nu = np.array([0.2, 0.2]); _cs = np.array([0.1, 0.1]); _ca = np.array([0.0, 0.0])
    _ = build_basis_numba(_e, _z, _x, _nu, _cs, _ca, specs)
    _ = predict_numba(np.zeros((2, len(coefs_a))), coefs_a, coefs_w)
    _ = ansatz_numba(_x, _e, _z, _nu)
    _ = full_predict_ds(_e, _x, _z, 0.2, 0.1, 0.0, specs, coefs_a, coefs_w, 2)
    log("  Done.")

    idx_rep = np.linspace(0, len(val) - 1, 30, dtype=int)
    opt_progress = []

    # ================================================================
    # PHASE 1: Baseline timing
    # ================================================================
    log("\n--- PHASE 1: Baseline (original code) ---")
    baseline = []
    for ii, i in enumerate(idx_rep):
        ts = [inference_baseline(val[i], model) for _ in range(3)]
        med = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}
        baseline.append(med)
        if (ii+1) % 10 == 0: log(f"  {ii+1}/30")

    bl_t = [t['t_total'] for t in baseline]
    log(f"  Baseline: median={np.median(bl_t):.1f}ms, mean={np.mean(bl_t):.1f}ms, "
        f"max={np.max(bl_t):.1f}ms")
    log(f"    ODE={np.median([t['t_ode'] for t in baseline]):.1f}ms  "
        f"interp={np.median([t['t_interp'] for t in baseline]):.1f}ms  "
        f"predict={np.median([t['t_predict'] for t in baseline]):.1f}ms  "
        f"recon={np.median([t['t_recon'] for t in baseline]):.1f}ms")
    opt_progress.append({'name': 'Baseline', 'median_ms': float(np.median(bl_t)),
                         'mean_ms': float(np.mean(bl_t)), 'max_ms': float(np.max(bl_t))})

    # ================================================================
    # PHASE 2: pySEOBNR reference
    # ================================================================
    log("\n--- PHASE 2: pySEOBNR timing ---")
    seob = []
    for ii, i in enumerate(idx_rep):
        d = val[i]
        ts = [time_pyseobnr(d['q'], d['chi1'], d['chi2'], d['e0']) for _ in range(3)]
        seob.append({'q': d['q'], 'chi1': d['chi1'], 'chi2': d['chi2'],
                     'e0': d['e0'], 'chi_eff': d['chi_eff'], 't_ms': float(np.median(ts))})
        if (ii+1) % 10 == 0: log(f"  {ii+1}/30")
    seob_t = [s['t_ms'] for s in seob]
    log(f"  pySEOBNR: median={np.median(seob_t):.1f}ms, mean={np.mean(seob_t):.1f}ms, "
        f"max={np.max(seob_t):.1f}ms")

    # ================================================================
    # PHASE 3: Optimization levels
    # ================================================================
    for n_ds, recon_dt, level_name in [
        (1000, 0.5, 'Numba+ds1000+dt0.5'),
        (500,  1.0, 'Numba+ds500+dt1.0'),
        (300,  1.0, 'Numba+ds300+dt1.0'),
    ]:
        log(f"\n--- Optimization: {level_name} ---")
        opt = []
        for ii, i in enumerate(idx_rep):
            ts = [inference_optimized(val[i], coefs_a, coefs_w, specs, n_ds, recon_dt)
                  for _ in range(3)]
            med = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}
            opt.append(med)
        ot = [t['t_total'] for t in opt]
        log(f"  {level_name}: median={np.median(ot):.1f}ms, mean={np.mean(ot):.1f}ms, "
            f"max={np.max(ot):.1f}ms")
        log(f"    ODE={np.median([t['t_ode'] for t in opt]):.1f}ms  "
            f"interp={np.median([t['t_interp'] for t in opt]):.1f}ms  "
            f"predict={np.median([t['t_predict'] for t in opt]):.1f}ms  "
            f"recon={np.median([t['t_recon'] for t in opt]):.1f}ms")
        speedups = [seob[i]['t_ms'] / opt[i]['t_total'] for i in range(len(seob))]
        log(f"  Speedup vs pySEOBNR: median={np.median(speedups):.1f}x, "
            f"min={np.min(speedups):.1f}x")
        opt_progress.append({'name': level_name, 'median_ms': float(np.median(ot)),
                             'mean_ms': float(np.mean(ot)), 'max_ms': float(np.max(ot))})

    # Pick best level that meets target or is closest
    best_level = min(opt_progress[1:], key=lambda p: abs(p['median_ms'] - 15))
    log(f"\n  Best level: {best_level['name']} ({best_level['median_ms']:.1f}ms)")

    # ================================================================
    # PHASE 4: Accuracy verification
    # ================================================================
    log("\n--- PHASE 4: Accuracy verification ---")
    # Numba vs original (exact match, no downsampling)
    max_diff = 0.0
    for i in range(min(20, len(val))):
        d = val[i]; n = len(d['t'])
        e = np.clip(d['e'][:n], 1e-6, 0.95); x = np.clip(d['x'][:n], 1e-6, 0.5)
        z = d['zeta'][:n]; nu_a = np.full(n, d['nu'])
        cs_a = np.full(n, d['chi_S']); ca_a = np.full(n, d['chi_A'])
        B_orig = build_basis_original(e, z, x, nu_a, cs_a, ca_a, **bc)
        da_orig = model['m_a'].predict(B_orig)
        B_nb = build_basis_numba(e, z, x, nu_a, cs_a, ca_a, specs)
        da_nb, _ = predict_numba(B_nb, coefs_a, coefs_w)
        max_diff = max(max_diff, np.max(np.abs(da_orig - da_nb)))
    log(f"  Numba vs original (no DS): max|diff| = {max_diff:.2e} -> {'PASS' if max_diff < 1e-8 else 'FAIL'}")

    # Downsampled accuracy vs full waveform error
    log("  Downsampling accuracy (waveform-level):")
    for n_ds in [1000, 500, 300]:
        errs = []
        for i in range(min(20, len(val))):
            d = val[i]; n = len(d['t'])
            e = np.clip(d['e'][:n], 1e-6, 0.95); x = np.clip(d['x'][:n], 1e-6, 0.5)
            z = d['zeta'][:n]
            # Full predict
            xi_a_f, xi_w_f = full_predict_ds(e, x, z, d['nu'], d['chi_S'], d['chi_A'],
                                              specs, coefs_a, coefs_w, n)
            h_f, h_ref_f, _ = reconstruct_fast(d, xi_a_f, xi_w_f, dt=0.5)
            E_full = mathcalE_error(h_ref_f, h_f)
            # DS predict
            xi_a_d, xi_w_d = full_predict_ds(e, x, z, d['nu'], d['chi_S'], d['chi_A'],
                                              specs, coefs_a, coefs_w, n_ds)
            h_d, h_ref_d, _ = reconstruct_fast(d, xi_a_d, xi_w_d, dt=0.5)
            E_ds = mathcalE_error(h_ref_d, h_d)
            errs.append(abs(E_ds - E_full))
        log(f"    n_ds={n_ds}: median|dE|={np.median(errs):.2e}, max|dE|={np.max(errs):.2e}")

    # ================================================================
    # PHASE 5: Full val timing with best config
    # ================================================================
    # Use n_ds=500, dt=1.0 as a good balance
    BEST_NDS = 500; BEST_DT = 1.0
    log(f"\n--- PHASE 5: Full validation timing (n_ds={BEST_NDS}, dt={BEST_DT}) ---")
    full_opt = []
    for i, d in enumerate(val):
        t = inference_optimized(d, coefs_a, coefs_w, specs, BEST_NDS, BEST_DT)
        full_opt.append({k: float(v) for k, v in t.items() if k.startswith('t_')})
        if (i+1) % 30 == 0: log(f"  {i+1}/{len(val)}")
    ft = [t['t_total'] for t in full_opt]
    log(f"  Full val ({len(val)} wf): median={np.median(ft):.1f}ms, "
        f"mean={np.mean(ft):.1f}ms, max={np.max(ft):.1f}ms")
    log(f"    ODE={np.median([t['t_ode'] for t in full_opt]):.1f}ms  "
        f"interp={np.median([t['t_interp'] for t in full_opt]):.1f}ms  "
        f"predict={np.median([t['t_predict'] for t in full_opt]):.1f}ms  "
        f"recon={np.median([t['t_recon'] for t in full_opt]):.1f}ms")

    # ================================================================
    # PHASE 6: Final optimized timing on 30 representative (for plots)
    # ================================================================
    log("\n--- PHASE 6: Final optimized timing (30 rep, for plots) ---")
    final_opt = []
    for ii, i in enumerate(idx_rep):
        ts = [inference_optimized(val[i], coefs_a, coefs_w, specs, BEST_NDS, BEST_DT)
              for _ in range(3)]
        med = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}
        final_opt.append(med)

    # ================================================================
    # PHASE 7: Plots and summary
    # ================================================================
    log("\n--- PHASE 7: Generating plots ---")
    make_all_plots(baseline, final_opt, seob, opt_progress, val, idx_rep)
    log("  Saved all plots.")

    # Save optimized model
    with open(os.path.join(OPTMODEL, 'model.pkl'), 'wb') as f:
        pickle.dump({'coefs_a': coefs_a, 'coefs_w': coefs_w, 'specs': specs,
                     'bc': bc, 'n_ds': BEST_NDS, 'recon_dt': BEST_DT,
                     'type': 'ridge_numba_optimized',
                     'base_model': 'ridge_nh7_me5_mchi1_a1e-06'}, f)

    # Summary
    fo_t = [t['t_total'] for t in final_opt]
    speedups = [seob[i]['t_ms'] / final_opt[i]['t_total'] for i in range(len(seob))]
    summary = {
        'target_ms': '10-20',
        'baseline': {
            'median_ms': float(np.median(bl_t)), 'max_ms': float(np.max(bl_t)),
            'breakdown': {k: float(np.median([t[k] for t in baseline]))
                         for k in ['t_ode','t_interp','t_predict','t_recon']},
        },
        'optimized': {
            'median_ms': float(np.median(fo_t)), 'max_ms': float(np.max(fo_t)),
            'breakdown': {k: float(np.median([t[k] for t in final_opt]))
                         for k in ['t_ode','t_interp','t_predict','t_recon']},
            'config': {'n_ds': BEST_NDS, 'recon_dt': BEST_DT},
        },
        'full_val': {
            'median_ms': float(np.median(ft)), 'mean_ms': float(np.mean(ft)),
            'max_ms': float(np.max(ft)), 'n_waveforms': len(val),
        },
        'pyseobnr': {
            'median_ms': float(np.median(seob_t)), 'max_ms': float(np.max(seob_t)),
        },
        'speedup_over_pyseobnr': {
            'median': float(np.median(speedups)), 'min': float(np.min(speedups)),
            'max': float(np.max(speedups)),
            'fraction_faster': float(np.mean(np.array(speedups) > 1)),
        },
        'optimization_speedup': float(np.median(bl_t) / np.median(fo_t)),
        'optimizations_applied': ['numba_jit_basis', 'numba_jit_ansatz',
                                  'numba_jit_ridge_matvec', 'downsample_predict',
                                  'np_interp_ode', 'np_interp_recon', 'coarser_recon_dt'],
        'accuracy_preserved_numba': bool(max_diff < 1e-8),
        'n_features': len(coefs_a),
        'optimization_progress': opt_progress,
    }
    with open(os.path.join(OUTDIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)
    with open(os.path.join(OUTDIR, 'baseline_timing.json'), 'w') as f:
        json.dump(baseline, f, indent=2, cls=NpEncoder)
    with open(os.path.join(OUTDIR, 'optimized_timing.json'), 'w') as f:
        json.dump(final_opt, f, indent=2, cls=NpEncoder)
    with open(os.path.join(OUTDIR, 'pyseobnr_timing.json'), 'w') as f:
        json.dump(seob, f, indent=2, cls=NpEncoder)
    with open(os.path.join(OUTDIR, 'optimization_log.json'), 'w') as f:
        json.dump({'log': LOG}, f, indent=2)

    log(f"\n{'=' * 70}")
    log("FINAL SUMMARY")
    log(f"{'=' * 70}")
    log(f"  Baseline:      {np.median(bl_t):7.1f} ms (median)")
    log(f"  Optimized:     {np.median(fo_t):7.1f} ms (median)")
    log(f"  pySEOBNR:      {np.median(seob_t):7.1f} ms (median)")
    log(f"  Full val:      {np.median(ft):7.1f} ms (median), {np.max(ft):.1f} ms (max)")
    log(f"  Opt speedup:   {np.median(bl_t)/np.median(fo_t):.0f}x over baseline")
    log(f"  vs pySEOBNR:   {np.median(speedups):.1f}x median, {np.min(speedups):.1f}x min")
    target_met = np.median(fo_t) <= 20
    log(f"  Target (10-20ms): {'MET' if target_met else 'NOT MET'} ({np.median(fo_t):.1f} ms)")
    log(f"\n  Results: {OUTDIR}")
