# Modulation Learning Workflow — Small-Spin Eccentric

## Goal

Build a fast eccentric (2,2) waveform model for **small-spin** binaries (|chi| <= 0.5). Given (q, chi1, chi2, e0), produce h22(t) by modulating a quasi-circular baseline with learned xi_amp(t) and xi_omega(t).

This extends the non-spinning workflow to include mild spins. The parameter space is 4D but restricted to |chi| <= 0.5, avoiding the strong spin-orbit regime. Lessons from the non-spinning case (monotonicity constraints, validation-based model selection) are directly applicable and serve as the starting point.

---

## Parameter Space

```
q     in [1, 10]         mass ratio
chi1  in [-0.5, 0.5]     primary spin (aligned)
chi2  in [-0.5, 0.5]     secondary spin (aligned)
e0    in [0.001, 0.5]    initial eccentricity
zeta0 = 0                initial anomaly (fixed)
omega0 = 0.0085          starting frequency
```

4 free parameters (q, chi1, chi2, e0). Derived spin combinations:
```
chi_S = (chi1 + chi2) / 2       symmetric spin
chi_A = (chi1 - chi2) / 2       antisymmetric spin
chi_eff = (q*chi1 + chi2)/(1+q) effective spin
```

---

## Context Files

See `context/`:
1. `gwnrxhme.md` — gwNRXHME framework
2. `EOB_modes.dat.m` — hFactEccCorr[2,2] ansatz
3. `chandra_ecc_spin_paper_supplement.m` — TaylorT2 merger time
4. `tmerger_hinder_et_al.txt` — Hinder et al. functional form

---

## Output Directory

All results go to `results/`. Structure:

```
spin_05_04_26/
├── workflow.md                            # This file
├── context/                               # Physics background & reference files
│   ├── gwnrxhme.md
│   ├── EOB_modes.dat.m
│   ├── chandra_ecc_spin_paper_supplement.m
│   └── tmerger_hinder_et_al.txt
├── scripts/
│   ├── generate_data.py                   # Step 1: Generate training data
│   └── fit.py                             # Steps 2-5: Model fitting & analysis
├── explore.ipynb                          # Interactive notebook for inspection
├── tracking/                              # All logs, changelogs, and agent outputs
│   ├── CHANGELOG.md                       # Chronological record of all actions taken
│   ├── progress_log.md                    # THE narrative log (moved here from results/common/)
│   ├── progress_log.json                  # Machine-readable progress (moved here from results/comparison/)
│   ├── checklist_<model_name>.txt         # PASS/FAIL checklist output per model
│   └── stdout_<model_name>.log            # Full stdout capture per model fit
├── results/
│   ├── common/
│   │   ├── parameter_space.pdf            # (q, e0) scatter colored by chi_eff
│   │   ├── parameter_space_spin.pdf       # (chi1, chi2) scatter colored by e0
│   │   ├── wf_length_histogram.pdf        # Histogram of waveform lengths
│   │   ├── wf_length_vs_q.pdf             # Waveform length vs q, colored/lined by e0
│   │   ├── wf_length_vs_chieff.pdf        # Waveform length vs chi_eff, colored by e0
│   │   ├── feature_importance.pdf         # RF feature importances
│   │   ├── modulation_examples.pdf        # Raw xi_amp, xi_omega for representative cases
│   │   ├── spin_effect_examples.pdf       # Same (q,e0) with different spins, showing spin effect
│   │   └── training_summary.json          # N_train, N_val, param ranges, timing
│   ├── models/
│   │   ├── <model_name>/                  # One subfolder per model
│   │   │   ├── model.pkl
│   │   │   ├── histogram.pdf              # mathcalE + MM + dephasing distributions
│   │   │   ├── histogram_ligo_mm.pdf       # LIGO mismatch distributions (one panel per Mtot)
│   │   │   ├── dephasing_vs_e0.pdf        # max|dphi| vs e0
│   │   │   ├── dephasing_vs_chieff.pdf    # max|dphi| vs chi_eff
│   │   │   ├── mathcalE_vs_e0.pdf
│   │   │   ├── mathcalE_vs_chieff.pdf
│   │   │   ├── ligo_mm_vs_e0.pdf          # LIGO mismatch vs e0 (val only, one curve per Mtot)
│   │   │   ├── ligo_mm_vs_chieff.pdf      # LIGO mismatch vs chi_eff (val only, one curve per Mtot)
│   │   │   ├── best_modulation.pdf        # Split layout: 60% inspiral, 40% merger
│   │   │   ├── median_modulation.pdf
│   │   │   ├── worst_modulation.pdf
│   │   │   └── summary.json
│   │   └── ...
│   ├── comparison/
│   │   ├── progress.pdf                   # Staircase of all approaches
│   │   ├── progress_ligo_mm.pdf           # LIGO mismatch staircase
│   │   ├── comparison_vs_e0.pdf           # All models overlaid
│   │   ├── comparison_vs_chieff.pdf       # All models overlaid vs chi_eff
│   │   ├── comparison_dephasing.pdf       # Dephasing comparison
│   │   ├── comparison_ligo_mm.pdf         # LIGO mismatch comparison
│   │   └── comparison_summary.json
│   ├── training_data.pkl
│   ├── validation_data.pkl
│   └── errors/                            # Per-model error arrays (npy)
│       └── <model_name>/
│           ├── params.npy                 # shape (N, 4): [q, chi1, chi2, e0] for each waveform
│           ├── mathcalE.npy               # shape (N,): L2 norm error per waveform
│           ├── td_mismatch.npy            # shape (N,): time-domain mismatch per waveform
│           ├── dephasing.npy              # shape (N,): max|dphi| per waveform
│           └── ligo_mismatch.npy          # shape (N, 5): LIGO mismatch per waveform, columns = [20, 65, 110, 155, 200] Msun
```

---

## Error Metrics

Report ALL of these for every model:

```python
def mathcalE_error(h_ref, h):
    n1 = np.sum(np.abs(h_ref)**2); n2 = np.sum(np.abs(h)**2)
    sdot = np.real(np.sum(h_ref * np.conj(h)))
    return ((n1 + n2) - 2 * sdot) / (2 * n1)

def compute_mismatch(h_pred, h_ref):
    inner = np.real(np.sum(h_pred * np.conj(h_ref)))
    norm_pred, norm_ref = np.linalg.norm(h_pred), np.linalg.norm(h_ref)
    return 1.0 - inner / (norm_pred * norm_ref) if norm_pred > 0 and norm_ref > 0 else 1.0

def max_dephasing(h_pred, h_ref):
    phi_ref = np.unwrap(np.angle(h_ref))
    phi_pred = np.unwrap(np.angle(h_pred))
    phi_pred -= (phi_pred[0] - phi_ref[0])
    return np.max(np.abs(phi_pred - phi_ref))
```

### 4. Frequency-domain LIGO mismatch

Compute the noise-weighted mismatch using the aLIGO design PSD for **5 total mass values**: Mtot = 20, 65, 110, 155, 200 Msun. This converts the geometric-unit waveforms to physical frequencies and weights them by detector sensitivity.

```python
from pycbc.types import TimeSeries
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower

def ligo_mismatch(h_pred, h_ref, dt_geometric, Mtot_msun, f_low=20.0):
    """Frequency-domain noise-weighted mismatch for a given total mass.
    
    Parameters:
        h_pred, h_ref: complex h22 arrays in geometric units (dt in M)
        dt_geometric: time step in geometric units (M)
        Mtot_msun: total mass in solar masses
        f_low: low frequency cutoff in Hz (default 20 Hz)
    Returns:
        mismatch: 1 - match (optimized over time and phase shifts)
    """
    # Convert geometric time to physical: t_sec = t_M * Mtot * G/c^3
    Mtot_sec = Mtot_msun * 4.925491025543576e-06  # Msun in seconds
    dt_sec = dt_geometric * Mtot_sec
    
    # Use only the real part (h_plus for face-on)
    hp_pred = TimeSeries(np.real(h_pred).astype(np.float64), delta_t=dt_sec)
    hp_ref  = TimeSeries(np.real(h_ref).astype(np.float64), delta_t=dt_sec)
    
    # Resize to same length
    tlen = max(len(hp_pred), len(hp_ref))
    hp_pred.resize(tlen)
    hp_ref.resize(tlen)
    
    # Generate PSD
    delta_f = 1.0 / hp_ref.duration
    flen = tlen // 2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
    
    m, _ = match(hp_ref, hp_pred, psd=psd, low_frequency_cutoff=f_low)
    return 1.0 - m
```

For each model evaluation, compute LIGO mismatches at all 5 Mtot values. Report median and max over the validation set for each Mtot.

### Targets

| Metric | Target |
|--------|--------|
| Waveform mathcalE | < 1% (0.01) |
| Waveform MM (time-domain) | < 1% (0.01) |
| max|dphi| | < 0.1 rad |
| xi_amp mathcalE | < 0.1% (0.001) |
| xi_omega mathcalE | < 1% (0.01) |
| LIGO mismatch (all Mtot) | < 1% (0.01) |

---

## Ansatz Decomposition

All models learn **residuals** on top of a physics-motivated ansatz, not the raw modulations directly. This decomposes the problem into known PN physics (ansatz) + learned correction (residual).

### Ansatz: h22_ecc

The leading eccentric correction to the (2,2) mode, truncated at O(epsilon^2):

```python
import numpy as np

def h22_ecc(x, e, zeta, nu, epsilon=1.0):
    """
    Eccentric correction to the (2,2) waveform mode h22^ecc.

    Parameters
    ----------
    x     : frequency parameter ~ (M*Omega)^(2/3)
    e     : orbital eccentricity
    zeta  : relativistic anomaly
    nu    : symmetric mass ratio
    epsilon : PN book-keeping parameter (set to 1 for physical use)

    Returns
    -------
    complex : h22^ecc truncated at O(epsilon^2)
    """
    e2   = e * e
    e3   = e2 * e
    eiz  = np.exp( 1j * zeta)
    emiz = np.exp(-1j * zeta)

    # --- Leading term (epsilon^0) ---
    leading = (4.0 + 2.0*e2*eiz**2 + e*emiz + 5.0*e*eiz) / (4.0*(1.0 - e2))

    # --- epsilon^2 terms inside the curly brace ---
    # constant in zeta
    term_const  = e * (26.0*nu/7.0 - 559.0/84.0)

    # e^{-2i zeta}
    term_em2iz  = e * np.exp(-2j*zeta) * (15.0*nu/14.0 - 95.0/168.0)

    # e^{-3i zeta}
    term_em3iz  = e2 * np.exp(-3j*zeta) * (9.0*nu/56.0 + 1.0/112.0)

    # e^{+3i zeta}
    term_e3iz   = e2 * np.exp( 3j*zeta) * (nu/8.0 - 49.0/48.0)

    # e^{+2i zeta} bracket
    term_e2iz   = np.exp( 2j*zeta) * (e3*(6.0*nu/7.0 - 41.0/21.0)
                                     + e *(nu/14.0 - 153.0/56.0))

    # e^{-i zeta} bracket
    term_emiz   = emiz * (e2*(7.0*nu/8.0 - 59.0/48.0)
                         + 27.0*nu/14.0 - 23.0/14.0)

    # e^{+i zeta} bracket
    term_eiz    = eiz  * (e2*(143.0*nu/56.0 - 2071.0/336.0)
                         + nu/14.0 - 13.0/7.0)

    curly = (term_const + term_em3iz + term_e3iz + term_em2iz
             + term_e2iz + term_emiz + term_eiz)

    pa_term = (x * e * epsilon**2) / (1.0 - e2)**2 * curly

    return leading + pa_term
```

### Deriving the ansatz modulations

From `h22_ecc`, the ansatz modulations are:
```python
xi_amp_ansatz = np.abs(h22_ecc(x, e, zeta, nu)) - 1.0
xi_omega_ansatz = xi_amp_ansatz / 0.9    # from Relation III (gwNRXHME), recalibrate with data
```

For circular orbits (e=0), `h22_ecc = 1` so `xi_amp_ansatz = 0` — the ansatz automatically satisfies the e=0 vanishing constraint.

### Residual definition

The residuals are what the models learn:
```
delta_xi_amp(t)   = xi_amp_SEOB(t)   - xi_amp_ansatz(t)
delta_xi_omega(t) = xi_omega_SEOB(t) - xi_omega_ansatz(t)
```

At inference, the full modulation is reconstructed as:
```
xi_amp(t)   = xi_amp_ansatz(t)   + delta_xi_amp_model(t)
xi_omega(t) = xi_omega_ansatz(t) + delta_xi_omega_model(t)
```

### What the residuals capture

- Spin-orbit and spin-spin effects (not in the non-spinning ansatz)
- Higher-order PN terms beyond O(epsilon^2)
- Resummation differences between the truncated ansatz and full SEOB
- Systematic ODE dynamics offsets (absorbed implicitly since we train on our ODE output)

The residuals should be significantly smaller than the raw modulations, making them easier to fit accurately.

### Data generation must store both

During data generation (Step 1), for each waveform compute and store:
- `xi_amp`, `xi_omega` — the raw SEOB modulations (as before)
- `xi_amp_ansatz`, `xi_omega_ansatz` — evaluated from `h22_ecc` using our ODE's (e, x, zeta, nu)
- `delta_xi_amp`, `delta_xi_omega` — the residuals (difference of the above)

### Feature importance and all fitting is on the residuals

Step 2 (feature importance) and Step 3 (model fitting) operate on `delta_xi_amp` and `delta_xi_omega`, not the raw modulations. The ansatz-only model (delta = 0) serves as the baseline instead of the circular-only model.

---

## Reconstruction Requirements

1. **Dense interpolation**: dt = 0.1M for all waveforms and modulations before computing errors.
2. **Smooth tapering**: Half-cosine taper from t = -50M to t = 0M. No hard cutoff. Applied at reconstruction time, not during data generation.
3. **Phase alignment**: Initial phase = 0 for both reference and model before computing metrics.
4. **Modulation computation**: Compute xi_amp and xi_omega through the full waveform up to t = +50M (into ringdown). Raw data should extend past merger — tapering is applied later during reconstruction, not during extraction. Having data past t = -50M prevents edge effects and keeps fits stable near merger.
5. **Modulation errors**: Computed on inspiral only (t < -50M), excluding taper region.
6. **Envelope monotonicity**: Enforce monotonicity on the upper and lower envelopes of both xi_amp and xi_omega. Upper envelope must be non-increasing and lower envelope must be non-decreasing as the binary approaches merger (eccentricity circularizes -> oscillation amplitude shrinks). Apply this as a post-processing constraint on extracted modulations before fitting.

---

## Implementation Steps

### Step 1: Training Data Generation

**Script**: `scripts/generate_data.py`

**Sampling**: Latin Hypercube over 4D parameter space:
```python
from scipy.stats.qmc import LatinHypercube
sampler = LatinHypercube(d=4, seed=42)
samples = sampler.random(n=300)

q    = 1.0 + 9.0 * samples[:, 0]              # [1, 10]
chi1 = -0.5 + 1.0 * samples[:, 1]             # [-0.5, 0.5]
chi2 = -0.5 + 1.0 * samples[:, 2]             # [-0.5, 0.5]
e0   = 0.001 + 0.499 * samples[:, 3]          # [0.001, 0.5]
```

- **300 training points** + **150 validation points** (separate LHC draws)
- zeta0 = 0, omega0 = 0.0085
- 300 points in 4D gives ~4x the per-dimension density as 100 in 4D, comparable coverage to 200 in 2D

For each point:
1. Run pySEOBNR eccentric -> h22_ecc(t), extract ICs (r0, pr0, pphi0)
2. Run pySEOBNR circular -> h22_cir(t) (cache per unique (q, chi1, chi2))
3. Run **our ODE** -> (e, x, zeta)(t)
4. Compute raw modulations xi_amp, xi_omega on common grid with dt=1M (storage)
5. Compute ansatz modulations xi_amp_ansatz, xi_omega_ansatz from `h22_ecc(x, e, zeta, nu)` using our ODE dynamics
6. Compute residuals: delta_xi_amp = xi_amp - xi_amp_ansatz, delta_xi_omega = xi_omega - xi_omega_ansatz
7. Store everything including chi1, chi2, chi_S, chi_A, raw modulations, ansatz, and residuals

**Critical**: the (e, x, zeta) used as features come from **our ODE**, not pySEOBNR dynamics. This ensures the model learns to work with the dynamics it will receive at inference.

**Plots to generate immediately** (in `results/common/`):
- `parameter_space.pdf` — scatter of (q, e0) colored by chi_eff, for train + val
- `parameter_space_spin.pdf` — scatter of (chi1, chi2) colored by e0, for train + val
- `wf_length_histogram.pdf` — histogram of waveform lengths in M
- `wf_length_vs_q.pdf` — waveform length vs q with separate curves/colors for e0 bins (e0 < 0.05, 0.05-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.5)
- `wf_length_vs_chieff.pdf` — waveform length vs chi_eff, colored by e0 bins
- `modulation_examples.pdf` — raw xi_amp, xi_omega, ansatz, and residuals for 4 representative cases (low q/low e, low q/high e, high q/low e, high q/high e), all at low spin. Show SEOB (black), ansatz (blue dashed), and residual (red) to visualize the decomposition.
- `spin_effect_examples.pdf` — for 2-3 fixed (q, e0) pairs, show xi_amp and xi_omega for the most negative-spin and most positive-spin cases in the training set at similar (q, e0). This isolates the spin effect on the modulations.
- `feature_importance.pdf` — RF importances

---

### Step 2: Feature Importance

Train a Random Forest on raw features (e, x, nu, chi_S, chi_A, cos(zeta), sin(zeta), cos(2*zeta), sin(2*zeta)) for **delta_xi_amp** and **delta_xi_omega** (the residuals, not the raw modulations). Plot importances.

**Key question**: how much do chi_S and chi_A contribute to the residuals? The ansatz is non-spinning, so spin effects should appear primarily in the residuals. With restricted spin range |chi| <= 0.5, spin contributions may still be small — but they should be more visible in the residuals than in the raw modulations (where cos(zeta) and e dominate).

---

### Step 3: Model Fitting + Mandatory Post-Training Checklist

**Script**: `scripts/fit.py`

Use the non-spinning workflow (`workflow_nospin_03_04_26.md`) and its results as the starting point. Adapt the approaches that worked there (see `results_nospin_03_04_26/`) to include spin parameters. Try multiple approaches, starting from the simplest.

**All models fit the residuals** (delta_xi_amp, delta_xi_omega), not the raw modulations. The ansatz is always added back at reconstruction time.

**Guidelines**:
- Start with an ansatz-only baseline (delta = 0, no learned correction) to establish the floor.
- Try multiple fitting approaches of increasing complexity for the residuals. Use your judgment on which methods to try based on what works.
- **Model selection must use validation data**, not training metrics. Overfitting is the primary risk — monitor for it.
- **Monotonicity as hard constraint**: Any model whose predictions produce non-monotonic envelopes (oscillation amplitude growing toward merger) is penalized. Envelope monotonicity violation is measured per-waveform and reported for all models. Models with median violation > 0.05 on validation are flagged as overfitting.
- **Key physical constraint**: modulations must vanish at e=0. Any model should enforce or respect this.
- **Preference**: analytical/closed-form models over black-box models when accuracy is comparable, for speed, interpretability, and Numba-compilability.
- If a phase correction step improves dephasing, generate `phase_corrections.pdf` showing dphi(t) for 5 representative cases across parameter space.

---

### MANDATORY: Post-Training Completion Checklist

**After EVERY model is trained, ALL of the following must be completed before moving to the next model.** Do not skip any item. Do not batch models. The agent must execute this checklist in order for each model and verify every item.

#### A. Save model
- [ ] Save model to `results/models/<name>/model.pkl`

#### B. Evaluate on training AND validation sets
- [ ] Compute mathcalE (L2 norm error) for every waveform in train and val
- [ ] Compute time-domain mismatch for every waveform in train and val
- [ ] Compute max|dphi| (dephasing) for every waveform in train and val
- [ ] Compute LIGO mismatch at all 5 Mtot = [20, 65, 110, 155, 200] Msun for every waveform in train and val
- [ ] Compute envelope monotonicity violation for every waveform in train and val

#### C. Save error arrays (npy)
- [ ] Save `results/errors/<name>/train_params.npy` — shape (N_train, 4)
- [ ] Save `results/errors/<name>/train_mathcalE.npy` — shape (N_train,)
- [ ] Save `results/errors/<name>/train_td_mismatch.npy` — shape (N_train,)
- [ ] Save `results/errors/<name>/train_dephasing.npy` — shape (N_train,)
- [ ] Save `results/errors/<name>/train_ligo_mismatch.npy` — shape (N_train, 5)
- [ ] Save `results/errors/<name>/val_params.npy` — shape (N_val, 4)
- [ ] Save `results/errors/<name>/val_mathcalE.npy` — shape (N_val,)
- [ ] Save `results/errors/<name>/val_td_mismatch.npy` — shape (N_val,)
- [ ] Save `results/errors/<name>/val_dephasing.npy` — shape (N_val,)
- [ ] Save `results/errors/<name>/val_ligo_mismatch.npy` — shape (N_val, 5)

#### D. Generate ALL diagnostic plots
- [ ] `histogram.pdf` — mathcalE + MM + dephasing distributions (train blue, val red), with target lines at 1e-2 (mathcalE), 1e-3 (mathcalE), 0.1 rad (dephasing)
- [ ] `histogram_ligo_mm.pdf` — LIGO mismatch distributions for all 5 Mtot values (train blue, val red). Layout: 5 panels (one per Mtot = 20, 65, 110, 155, 200 Msun), with 1e-2 target line.
- [ ] `dephasing_vs_e0.pdf` — max|dphi| vs e0, colored by q, with 0.1 rad target
- [ ] `dephasing_vs_chieff.pdf` — max|dphi| vs chi_eff, colored by e0, with 0.1 rad target
- [ ] `mathcalE_vs_e0.pdf` — mathcalE vs e0, colored by q
- [ ] `mathcalE_vs_chieff.pdf` — mathcalE vs chi_eff, colored by e0
- [ ] `ligo_mm_vs_e0.pdf` — LIGO mismatch vs e0, validation only, one curve/color per Mtot, colored by q within each. Log-scale y-axis, 1e-2 target line.
- [ ] `ligo_mm_vs_chieff.pdf` — LIGO mismatch vs chi_eff, validation only, one curve/color per Mtot. Log-scale y-axis, 1e-2 target line.
- [ ] `best_modulation.pdf` — split layout (3 rows x 2 cols, 60/40 inspiral/merger-ringdown):
  - Row 1: xi_amp — SEOB (black) vs model (red dashed), both panels
  - Row 2: xi_omega — same, both panels (data now extends to t = +50M)
  - Row 3: Re(h22) — same
  - Title: params (q, chi1, chi2, e0) + all error metrics
  - Vertical dashed line at t = -50M marking taper onset
- [ ] `median_modulation.pdf` — same for median case
- [ ] `worst_modulation.pdf` — same for worst case

#### E. Save summary
- [ ] `summary.json` — all metrics: mathcalE (median/max, train/val), MM (median/max, train/val), dephasing (median/max, train/val), LIGO mismatch (median/max per Mtot, train/val), monotonicity violation (median/max, train/val), fraction of val cases meeting each target

#### F. Log to tracking/
- [ ] Append timestamped section to `tracking/progress_log.md` with:
  - Model name, configuration summary
  - All metrics (mathcalE, MM, dephasing, LIGO mismatch at each Mtot — median, max for train and val)
  - Fraction meeting targets: dphi < 0.1 rad, mathcalE < 0.01, LIGO MM < 0.01 at each Mtot
  - Worst-case parameters and error values
  - Monotonicity violation stats
  - Insights: what works, what fails, why
- [ ] Append entry to `tracking/CHANGELOG.md` (MODEL TRAINED + CHECKLIST PASSED/FAILED)
- [ ] Append entry to `tracking/progress_log.json`

#### G. Verify completion
- [ ] Verify ALL files exist in `results/models/<name>/` (model.pkl, all PDFs, summary.json)
- [ ] Verify ALL npy files exist in `results/errors/<name>/`
- [ ] Print checklist status to stdout AND save to `tracking/checklist_<model_name>.txt`:
```
=== CHECKLIST: <model_name> ===
[PASS/FAIL] model.pkl saved
[PASS/FAIL] error arrays saved (10 npy files)
[PASS/FAIL] histogram.pdf
[PASS/FAIL] histogram_ligo_mm.pdf
[PASS/FAIL] dephasing_vs_e0.pdf
[PASS/FAIL] dephasing_vs_chieff.pdf
[PASS/FAIL] mathcalE_vs_e0.pdf
[PASS/FAIL] mathcalE_vs_chieff.pdf
[PASS/FAIL] ligo_mm_vs_e0.pdf
[PASS/FAIL] ligo_mm_vs_chieff.pdf
[PASS/FAIL] best_modulation.pdf
[PASS/FAIL] median_modulation.pdf
[PASS/FAIL] worst_modulation.pdf
[PASS/FAIL] summary.json
[PASS/FAIL] progress_log.md updated
[PASS/FAIL] CHANGELOG.md updated
=== ALL CHECKS PASSED / X CHECKS FAILED ===
```

**If any check fails, the agent must fix it before proceeding to the next model.**

---

### Step 4: Model Comparison (after all models complete checklist)

Generate `results/comparison/`:
- `progress.pdf` — staircase bar chart (val median mathcalE, val median dephasing)
- `progress_ligo_mm.pdf` — staircase bar chart of val median LIGO mismatch, one group of bars per model, one bar per Mtot
- `comparison_vs_e0.pdf` — all models overlaid
- `comparison_vs_chieff.pdf` — all models overlaid vs chi_eff
- `comparison_dephasing.pdf` — dephasing comparison
- `comparison_ligo_mm.pdf` — LIGO mismatch comparison across models for each Mtot
- `comparison_summary.json`

Also generate a **comparison with the non-spinning results**:
- `comparison_nospin.pdf` — for the best model here vs the best model from `results_nospin_03_04_26/`, compare validation dephasing distributions. This quantifies the accuracy cost of adding spins.

---

## Logging & Tracking

All logs, changelogs, checklist outputs, and stdout captures go in the `tracking/` folder. This folder is the primary record for writing the paper — it must contain enough detail to reconstruct the full story of which methods were tried, why each succeeded or failed, what the key insights were, and how the final model was selected.

### CHANGELOG.md

**`tracking/CHANGELOG.md`** — chronological record of every action taken.

Format: reverse-chronological (newest first). Each entry:
```markdown
## [YYYY-MM-DD HH:MM] <action type>

**What**: <one-line summary>
**Details**: <specifics — model name, config, metrics, files created>
**Status**: DONE / FAILED / IN PROGRESS
```

Action types:
- `DATA GENERATION` — training/validation data created
- `FEATURE IMPORTANCE` — feature importance analysis completed
- `MODEL TRAINED` — a model was fitted
- `CHECKLIST PASSED` — post-training checklist verified
- `CHECKLIST FAILED` — checklist had failures (list which)
- `COMPARISON` — comparison plots generated
- `NOTEBOOK` — explore.ipynb created/updated
- `FIX` — a failed checklist item was fixed

Every script and every Ralph Loop iteration must append to this file.

### progress_log.md

**`tracking/progress_log.md`** — THE narrative log for the paper.

**EVERYTHING** goes here:
- Data generation stats, timing, parameter coverage
- Feature importance analysis (especially: how much do spin features matter?)
- Every model attempt with full metrics (mathcalE, MM, dephasing, LIGO mismatches at all 5 Mtot — median, max, distributions)
- Worst-case analysis: which (q, chi1, chi2, e0) combinations fail and why
- Physics insights: how modulations depend on spin at fixed (q, e0)
- Comparison with non-spinning results: accuracy degradation from adding spins
- Comparison insights: why some methods work better than others
- Final model selection with justification

Write as if drafting the methods/results sections of a paper.

### progress_log.json

**`tracking/progress_log.json`** — machine-readable progress array.

### Per-model outputs

- **`tracking/checklist_<model_name>.txt`** — the PASS/FAIL checklist output from section G of the post-training checklist.
- **`tracking/stdout_<model_name>.log`** — full stdout capture during model fitting and evaluation.

---

## Key Differences from Non-Spinning Workflow

1. **4D parameter space** (q, chi1, chi2, e0) instead of 2D (q, e0)
2. **Spin features needed** — chi_S and chi_A are available as input features for fitting
3. **300 training + 150 validation** points (up from 200+100) to cover the extra 2 spin dimensions
4. **Circular waveforms cached per (q, chi1, chi2)** — no longer just per q, so cache reuse is lower
5. **Spin-dependent diagnostics** — plots vs chi_eff added for all models
6. **spin_effect_examples.pdf** — new diagnostic isolating the spin contribution to modulations
7. **Expected accuracy**: worse than non-spinning (more parameters, sparser coverage, spin-orbit effects), but should still reach the targets for most of the parameter space. High-e0 + high-|chi| corner is expected to be hardest.

---

## Why Small Spin (|chi| <= 0.5) Is a Good Intermediate Step

1. **Moderate spin-orbit coupling**: spin effects are present but not dominant, so the modulation structure found in the non-spinning case should still hold
2. **Spin enters perturbatively**: at |chi| <= 0.5, spin corrections to the modulations are expected to be small, keeping the problem tractable
3. **Avoids near-extremal physics**: high-spin binaries have qualitatively different orbital dynamics (e.g., zoom-whirl orbits), which are absent at |chi| <= 0.5
4. **Direct comparison with non-spinning**: the non-spinning case is a subset, so we can validate that the small-spin model reproduces the non-spinning results when chi1=chi2=0
5. **Stepping stone to full spin**: if this works well, extending to |chi| <= 0.8 or 0.99 requires only more training data, not a fundamentally different approach

---

## Error Arrays (npy files)

After evaluating each model, save all per-waveform errors as npy arrays in `results/errors/<model_name>/`. This enables quick reloading without re-running model evaluation.

For **both training and validation** sets (use prefix `train_` and `val_` for filenames):
- `params.npy` — shape (N, 4): columns are [q, chi1, chi2, e0]
- `mathcalE.npy` — shape (N,): L2 norm error
- `td_mismatch.npy` — shape (N,): time-domain mismatch
- `dephasing.npy` — shape (N,): max|dphi|
- `ligo_mismatch.npy` — shape (N, 5): LIGO mismatch, columns correspond to Mtot = [20, 65, 110, 155, 200] Msun

Example: `results/errors/ridge_best/val_ligo_mismatch.npy` contains the validation LIGO mismatches for the best Ridge model.

---

## Exploration Notebook

`explore.ipynb` — interactive notebook for post-hoc inspection. Must support:

1. **Load any saved model** from `results/models/<name>/model.pkl`
2. **Load training/validation data** from pickle files
3. **Load all error arrays** from `results/errors/<model_name>/` npy files
4. **Evaluate a model on any waveform** and inspect xi_amp, xi_omega, h22 in any user-specified time window
5. **Plot modulations and waveforms** with adjustable time range (zoom into inspiral, merger, or ringdown)
6. **Display error summaries**: histograms, scatter plots, parameter-space colormaps for any error metric (mathcalE, TD mismatch, dephasing, LIGO mismatch at any Mtot)
7. **Compare models side-by-side** on the same waveform

The notebook should have clearly labeled sections with helper functions at the top. It is a tool for the user, not an automated pipeline — keep it flexible and interactive.

---

## Connection to dyn_rewrite

The `dyn_rewrite` module provides:
- Fast EOB dynamics: (e, x, zeta)(t)
- All physics: Hamiltonian, flux, RR force, evolution equations
- Numba-compiled for speed

**Training uses both**:
- pySEOBNR for reference h22_ecc and h22_cir
- Our ODE for (e, x, zeta) input features

**Inference uses only**:
- Our ODE
- Fitted modulation model
- Circular waveform generator

---

## Environment

```
conda activate kitp-py310
```

pySEOBNR required for training. Inference uses only our ODE + model.

---

## Ralph Loop Configuration

This workflow is designed to be driven by a Ralph Loop. The loop iterates over models, and the mandatory checklist (Step 3) acts as the gate — the agent cannot proceed until every check passes.

### Running the full workflow

```
/ralph-loop:ralph-loop "Read modulation_learning/spin_05_04_26/workflow.md and execute it end-to-end.

STATE TRACKING: At each iteration, check what has been completed so far:
1. Does training_data.pkl exist? If not, run Step 1 (generate_data.py).
2. Does feature_importance.pdf exist? If not, run Step 2.
3. For each model in sequence (ansatz_only, then increasingly complex residual models):
   a. Has this model's checklist been completed? Check results/models/<name>/summary.json exists AND all npy files in results/errors/<name>/ exist AND all PDFs in results/models/<name>/ exist.
   b. If not complete, train the model (or resume from where it left off) and execute the MANDATORY Post-Training Completion Checklist from the workflow. Print the PASS/FAIL verification. Fix any failures before proceeding.
   c. If complete, move to the next model.
4. After all models pass their checklists, run Step 4 (comparison plots).
5. Generate explore.ipynb.

IMPORTANT: After each model, you MUST run the full checklist (sections A-G) and print the verification output. Do not skip any plots, npy files, or summary.json. Do not batch multiple models.

Output <promise>WORKFLOW COMPLETE</promise> when all models are trained, all checklists pass, comparison plots are generated, and explore.ipynb exists." --max-iterations 50 --completion-promise "WORKFLOW COMPLETE"
```

### Running a single model iteration

For debugging or adding a new model approach:

```
/ralph-loop:ralph-loop "Read modulation_learning/spin_05_04_26/workflow.md. Load training and validation data from results/training_data.pkl and results/validation_data.pkl.

Train the next untrained residual model (check which models already have summary.json in results/models/). Use the ansatz decomposition from the workflow — fit delta_xi_amp and delta_xi_omega.

After training, execute the MANDATORY Post-Training Completion Checklist (sections A-G). Print the full PASS/FAIL verification. Fix any failures.

Output <promise>MODEL COMPLETE</promise> when all checks pass." --max-iterations 10 --completion-promise "MODEL COMPLETE"
```

### Cancelling

```
/ralph-loop:cancel-ralph
```
