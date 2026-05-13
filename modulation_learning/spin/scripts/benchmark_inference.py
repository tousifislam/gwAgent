"""
Benchmark end-to-end inference time for the best model.

Measures: ODE integration + basis evaluation + reconstruction for various (q, chi1, chi2, e0).

Usage:
    conda activate kitp-py310
    cd modulation_learning/spin_05_04_26
    python scripts/benchmark_inference.py
"""
import sys, os, time, pickle, json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

DYN_SRC = '/Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/dyn_rewrite/src'
sys.path.insert(0, DYN_SRC)
from dynamics import setup_and_integrate

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')


# ====================================================================
# Load best model
# ====================================================================
def load_best_model():
    # Find the best ridge model
    model_dir = os.path.join(RESULTS, 'models', 'ridge_nh7_me5_mchi1_a1e-06')
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model


# ====================================================================
# Ansatz (copied from fit.py)
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


def build_basis(e, z, x, nu, chi_S, chi_A, max_e=4, max_x=3, max_nu=2,
                max_chi=1, n_harm=5):
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


def smooth_taper(t, ts=-50.0, te=0.0):
    w = np.ones_like(t)
    m = (t >= ts) & (t <= te)
    w[m] = 0.5 * (1 + np.cos(np.pi * (t[m] - ts) / (te - ts)))
    w[t > te] = 0
    return w


# ====================================================================
# End-to-end inference (without pySEOBNR — uses stored circular wf)
# ====================================================================
def inference_with_stored_circular(d, model):
    """
    Full inference using stored circular waveform (simulates production use
    where circular waveforms are pre-cached or fast to generate).

    Returns timing breakdown dict.
    """
    bc = model['bc']
    m_a = model['m_a']
    m_w = model['m_w']

    q = d['q']; chi1 = d['chi1']; chi2 = d['chi2']; e0 = d['e0']
    m_1 = q / (1 + q); m_2 = 1 / (1 + q); nu = m_1 * m_2
    chi_S = (chi1 + chi2) / 2; chi_A = (chi1 - chi2) / 2

    # --- Step 1: ODE integration ---
    # Need initial conditions — in production these come from a fast IC solver
    # Here we use the stored ones from pySEOBNR (this is what you'd replace)
    r0 = d.get('r0', 20.0)
    pr0 = d.get('pr0', 0.0)
    pphi0 = d.get('pphi0', 3.5)

    t0 = time.perf_counter()
    ode = setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0, rtol=1e-8)
    t_ode = time.perf_counter() - t0

    # --- Step 2: Interpolate ODE onto waveform grid ---
    t1 = time.perf_counter()
    # Use stored time grid and alignment
    t_ode_aligned = ode['t'] + d['t_ecc_start'] - d['t_peak_ecc']
    n = len(d['t'])
    t_ode_min, t_ode_max = t_ode_aligned[0], t_ode_aligned[-1]
    valid = (d['t'] >= t_ode_min) & (d['t'] <= t_ode_max)

    e_ode = np.zeros(n); x_ode = np.zeros(n); zeta_ode = np.zeros(n)
    if np.sum(valid) > 10:
        e_ode[valid] = CubicSpline(t_ode_aligned, ode['e'])(d['t'][valid])
        x_ode[valid] = CubicSpline(t_ode_aligned, ode['x'])(d['t'][valid])
        zeta_ode[valid] = CubicSpline(t_ode_aligned, ode['zeta'])(d['t'][valid])
    e_ode = np.clip(e_ode, 1e-6, 0.95)
    x_ode = np.clip(x_ode, 1e-6, 0.5)
    t_interp = time.perf_counter() - t1

    # --- Step 3: Ansatz + Ridge prediction ---
    t2 = time.perf_counter()
    xi_amp_ansatz = np.abs(h22_ecc_ansatz(x_ode, e_ode, zeta_ode, nu)) - 1.0
    xi_omega_ansatz = xi_amp_ansatz / 0.9

    B = build_basis(e_ode, zeta_ode, x_ode, np.full(n, nu),
                    np.full(n, chi_S), np.full(n, chi_A), **bc)
    delta_a = m_a.predict(B)
    delta_w = m_w.predict(B)
    xi_amp = xi_amp_ansatz + delta_a
    xi_omega = xi_omega_ansatz + delta_w
    t_predict = time.perf_counter() - t2

    # --- Step 4: Waveform reconstruction ---
    t3 = time.perf_counter()
    dt = 0.1
    t_d = np.arange(d['t'][0], d['t'][-1], dt)
    h_cir_d = CubicSpline(d['t'], np.real(d['h_cir']))(t_d) + \
              1j * CubicSpline(d['t'], np.imag(d['h_cir']))(t_d)
    xi_a_d = np.interp(t_d, d['t'], xi_amp)
    xi_w_d = np.interp(t_d, d['t'], xi_omega)
    taper = smooth_taper(t_d)
    xi_a_d *= taper; xi_w_d *= taper
    A_p = np.abs(h_cir_d) * (1 + xi_a_d)
    pc = np.unwrap(np.angle(h_cir_d)); oc = np.gradient(pc, dt)
    pp = cumulative_trapezoid(oc * (1 + xi_w_d), dx=dt, initial=0.0)
    h_pred = A_p * np.exp(1j * pp)
    t_recon = time.perf_counter() - t3

    return {
        't_ode_ms': t_ode * 1000,
        't_interp_ms': t_interp * 1000,
        't_predict_ms': t_predict * 1000,
        't_recon_ms': t_recon * 1000,
        't_total_ms': (t_ode + t_interp + t_predict + t_recon) * 1000,
        'n_ode_steps': len(ode['t']),
        'n_wf_pts': len(t_d),
        'wf_length_M': d['wf_length_M'],
    }


def inference_ode_only(q, chi1, chi2, e0, r0=20.0, pr0=0.0, pphi0=3.5):
    """Just time the ODE integration (the dominant cost)."""
    t0 = time.perf_counter()
    ode = setup_and_integrate(q, chi1, chi2, r0, pr0, pphi0, e0, 0.0, rtol=1e-8)
    elapsed = time.perf_counter() - t0
    return elapsed * 1000, len(ode['t'])


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("Loading model and data...", flush=True)
    model = load_best_model()

    with open(os.path.join(RESULTS, 'training_data.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(RESULTS, 'validation_data.pkl'), 'rb') as f:
        val = pickle.load(f)

    # JIT warmup
    print("Warming up JIT...", flush=True)
    _ = setup_and_integrate(1.0, 0.0, 0.0, 20.0, 0.0, 3.5, 0.05, 0.0,
                            t_end=100, rtol=1e-4, max_steps=50)
    # Second warmup to ensure full compilation
    _ = setup_and_integrate(2.0, 0.1, -0.1, 18.0, 0.0, 3.8, 0.1, 0.0, rtol=1e-8)
    print("Done.\n", flush=True)

    # ================================================================
    # Part 1: Full end-to-end timing on representative val waveforms
    # ================================================================
    print("=" * 80)
    print("PART 1: Full end-to-end inference (ODE + predict + reconstruct)")
    print("  Using stored circular waveforms and initial conditions")
    print("=" * 80)

    # Pick diverse cases from validation set
    cases = []
    # Low q, low e, low spin
    cases += [(d, f"q={d['q']:.1f} chi_eff={d['chi_eff']:.2f} e0={d['e0']:.3f}")
              for d in sorted(val, key=lambda d: d['q'] + d['e0'])[:3]]
    # High q, high e
    cases += [(d, f"q={d['q']:.1f} chi_eff={d['chi_eff']:.2f} e0={d['e0']:.3f}")
              for d in sorted(val, key=lambda d: -(d['q'] + d['e0']))[:3]]
    # High spin
    cases += [(d, f"q={d['q']:.1f} chi_eff={d['chi_eff']:.2f} e0={d['e0']:.3f}")
              for d in sorted(val, key=lambda d: -abs(d['chi_eff']))[:3]]
    # Mid-range
    cases += [(d, f"q={d['q']:.1f} chi_eff={d['chi_eff']:.2f} e0={d['e0']:.3f}")
              for d in sorted(val, key=lambda d: abs(d['q'] - 5) + abs(d['e0'] - 0.2))[:3]]

    # Remove duplicates
    seen = set()
    unique_cases = []
    for d, label in cases:
        if d['idx'] not in seen:
            seen.add(d['idx'])
            unique_cases.append((d, label))

    print(f"\n{'Case':<45s} {'ODE':>8s} {'Interp':>8s} {'Predict':>8s} {'Recon':>8s} {'TOTAL':>8s} {'Steps':>7s} {'WfPts':>8s} {'Length':>8s}")
    print("-" * 120)

    all_timings = []
    for d, label in unique_cases:
        # Run 3 times and take median
        timings = []
        for _ in range(3):
            t = inference_with_stored_circular(d, model)
            timings.append(t)
        med = {}
        for k in timings[0]:
            vals = [t[k] for t in timings]
            med[k] = np.median(vals)

        print(f"{label:<45s} {med['t_ode_ms']:7.1f}ms {med['t_interp_ms']:7.1f}ms "
              f"{med['t_predict_ms']:7.1f}ms {med['t_recon_ms']:7.1f}ms "
              f"{med['t_total_ms']:7.1f}ms {int(med['n_ode_steps']):>7d} "
              f"{int(med['n_wf_pts']):>8d} {med['wf_length_M']/1e3:7.1f}kM")

        med['q'] = d['q']; med['chi1'] = d['chi1']; med['chi2'] = d['chi2']
        med['e0'] = d['e0']; med['chi_eff'] = d['chi_eff']
        all_timings.append(med)

    # ================================================================
    # Part 2: ODE-only timing across full validation set
    # ================================================================
    print(f"\n{'=' * 80}")
    print("PART 2: ODE integration timing across full validation set")
    print(f"{'=' * 80}\n")

    ode_times = []
    for d in val:
        t_ms, n_steps = inference_ode_only(d['q'], d['chi1'], d['chi2'], d['e0'],
                                            r0=20.0, pr0=0.0, pphi0=3.5)
        ode_times.append({'q': d['q'], 'chi1': d['chi1'], 'chi2': d['chi2'],
                         'e0': d['e0'], 'chi_eff': d['chi_eff'],
                         't_ms': t_ms, 'n_steps': n_steps,
                         'wf_length_M': d['wf_length_M']})

    ts = np.array([t['t_ms'] for t in ode_times])
    print(f"ODE timing (150 val waveforms, rtol=1e-8):")
    print(f"  Median: {np.median(ts):.1f} ms")
    print(f"  Mean:   {np.mean(ts):.1f} ms")
    print(f"  Min:    {np.min(ts):.1f} ms")
    print(f"  Max:    {np.max(ts):.1f} ms")
    print(f"  Std:    {np.std(ts):.1f} ms")

    # By parameter bins
    print(f"\nODE timing by parameter:")
    print(f"  {'Bin':<30s} {'Median':>8s} {'Mean':>8s} {'Max':>8s} {'Count':>6s}")
    print(f"  {'-'*60}")

    for label, filt in [
        ("q < 3", lambda t: t['q'] < 3),
        ("q in [3, 7]", lambda t: 3 <= t['q'] < 7),
        ("q > 7", lambda t: t['q'] >= 7),
        ("e0 < 0.1", lambda t: t['e0'] < 0.1),
        ("e0 in [0.1, 0.3]", lambda t: 0.1 <= t['e0'] < 0.3),
        ("e0 > 0.3", lambda t: t['e0'] >= 0.3),
        ("|chi_eff| < 0.1", lambda t: abs(t['chi_eff']) < 0.1),
        ("|chi_eff| > 0.3", lambda t: abs(t['chi_eff']) > 0.3),
    ]:
        subset = [t['t_ms'] for t in ode_times if filt(t)]
        if subset:
            print(f"  {label:<30s} {np.median(subset):7.1f}ms {np.mean(subset):7.1f}ms "
                  f"{np.max(subset):7.1f}ms {len(subset):>6d}")

    # ================================================================
    # Part 3: Breakdown of prediction step (basis + Ridge)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("PART 3: Prediction step breakdown (basis construction + Ridge predict)")
    print(f"{'=' * 80}\n")

    bc = model['bc']
    m_a = model['m_a']; m_w = model['m_w']

    for n_pts in [1000, 5000, 10000, 30000, 50000]:
        e = np.random.uniform(0.01, 0.5, n_pts)
        x = np.random.uniform(0.01, 0.3, n_pts)
        z = np.random.uniform(0, 2*np.pi, n_pts)
        nu_arr = np.full(n_pts, 0.2)
        chiS_arr = np.full(n_pts, 0.1)
        chiA_arr = np.full(n_pts, 0.05)

        # Warmup
        B = build_basis(e, z, x, nu_arr, chiS_arr, chiA_arr, **bc)
        _ = m_a.predict(B)

        # Time basis construction
        t0 = time.perf_counter()
        for _ in range(5):
            B = build_basis(e, z, x, nu_arr, chiS_arr, chiA_arr, **bc)
        t_basis = (time.perf_counter() - t0) / 5 * 1000

        # Time Ridge prediction
        t0 = time.perf_counter()
        for _ in range(5):
            da = m_a.predict(B); dw = m_w.predict(B)
        t_ridge = (time.perf_counter() - t0) / 5 * 1000

        # Time ansatz
        t0 = time.perf_counter()
        for _ in range(5):
            xa = np.abs(h22_ecc_ansatz(x, e, z, nu_arr)) - 1.0
        t_ansatz = (time.perf_counter() - t0) / 5 * 1000

        print(f"  n_pts={n_pts:>6d}: basis={t_basis:7.1f}ms  ridge={t_ridge:7.1f}ms  "
              f"ansatz={t_ansatz:7.1f}ms  total_predict={t_basis+t_ridge+t_ansatz:7.1f}ms  "
              f"(B shape: {B.shape})")

    # ================================================================
    # Part 4: Comparison with pySEOBNR
    # ================================================================
    print(f"\n{'=' * 80}")
    print("PART 4: pySEOBNR reference timing (5 representative cases)")
    print(f"{'=' * 80}\n")

    try:
        from pyseobnr.generate_waveform import generate_modes_opt

        test_cases = [
            (1.5, 0.0, 0.0, 0.05, "q=1.5, nonspinning, low e"),
            (3.0, 0.3, -0.2, 0.2, "q=3, mild spin, mid e"),
            (5.0, 0.0, 0.0, 0.3, "q=5, nonspinning, high e"),
            (8.0, -0.4, 0.3, 0.1, "q=8, negative spin, low e"),
            (10.0, 0.5, 0.5, 0.4, "q=10, max spin, high e"),
        ]

        print(f"  {'Case':<40s} {'pySEOBNR':>10s} {'Our ODE':>10s} {'Speedup':>8s}")
        print(f"  {'-'*70}")

        for q, c1, c2, e0, label in test_cases:
            # pySEOBNR
            times_seob = []
            for _ in range(3):
                t0 = time.perf_counter()
                t_ecc, modes_ecc, model_ecc = generate_modes_opt(
                    q, c1, c2, 0.0085, eccentricity=e0, rel_anomaly=0.0,
                    approximant="SEOBNRv5EHM", debug=True,
                    settings={'use_wave_convention': True})
                times_seob.append((time.perf_counter() - t0) * 1000)
            t_seob = np.median(times_seob)

            # Our ODE
            dyn = model_ecc.dynamics
            r0, pr0, pphi0 = dyn[0, 1], dyn[0, 3], dyn[0, 4]
            times_ode = []
            for _ in range(3):
                t0 = time.perf_counter()
                ode = setup_and_integrate(q, c1, c2, r0, pr0, pphi0, e0, 0.0, rtol=1e-8)
                times_ode.append((time.perf_counter() - t0) * 1000)
            t_ode = np.median(times_ode)

            speedup = t_seob / t_ode if t_ode > 0 else float('inf')
            print(f"  {label:<40s} {t_seob:9.1f}ms {t_ode:9.1f}ms {speedup:7.1f}x")

    except ImportError:
        print("  pySEOBNR not available, skipping comparison")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY: End-to-end inference time breakdown")
    print(f"{'=' * 80}\n")

    if all_timings:
        ode_med = np.median([t['t_ode_ms'] for t in all_timings])
        interp_med = np.median([t['t_interp_ms'] for t in all_timings])
        pred_med = np.median([t['t_predict_ms'] for t in all_timings])
        recon_med = np.median([t['t_recon_ms'] for t in all_timings])
        total_med = np.median([t['t_total_ms'] for t in all_timings])

        print(f"  Component breakdown (median over {len(all_timings)} cases):")
        print(f"    ODE integration:    {ode_med:7.1f} ms  ({ode_med/total_med*100:4.1f}%)")
        print(f"    ODE interpolation:  {interp_med:7.1f} ms  ({interp_med/total_med*100:4.1f}%)")
        print(f"    Basis + Ridge:      {pred_med:7.1f} ms  ({pred_med/total_med*100:4.1f}%)")
        print(f"    Reconstruction:     {recon_med:7.1f} ms  ({recon_med/total_med*100:4.1f}%)")
        print(f"    ---")
        print(f"    TOTAL:              {total_med:7.1f} ms")
        print(f"\n  Note: Circular waveform assumed pre-cached. If generating circular")
        print(f"  waveform on-the-fly with pySEOBNR, add ~300-500ms.")

    # Save results
    out = {
        'detailed_timings': all_timings,
        'ode_timings': ode_times,
        'summary': {
            'median_total_ms': float(total_med) if all_timings else None,
            'median_ode_ms': float(np.median(ts)),
            'n_val': len(val),
        }
    }
    outfile = os.path.join(RESULTS, 'inference_timing.json')
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(outfile, 'w') as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {outfile}")
