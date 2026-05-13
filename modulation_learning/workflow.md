# Modulation Learning Workflow

## Goal

Learn the eccentric modulation functions ξ_amp(t) and ξ_ω(t) as functions of the orbital dynamics (e, x, ζ) from `dyn_rewrite` and intrinsic parameters (q, χ₁, χ₂). These modulations describe how eccentricity perturbs the amplitude and frequency of the (2,2) gravitational wave mode relative to a quasi-circular baseline.

The output of this step feeds directly into waveform reconstruction:
```
A_pred(t)     = A_cir(t) * (1 + ξ_amp(t))
ω_pred(t)     = ω_cir(t) * (1 + ξ_ω(t))
φ_pred(t)     = ∫ ω_pred dt
h22_pred(t)   = A_pred * exp(i φ_pred)
```

---

## Background & Context Files

Four context files in `context/`:

1. **`gwnrxhme.md`** — the gwNRXHME framework (Islam & Venumadhav): definitions of ξ_A and ξ_ω, the three universal phenomenological relations, and the multi-modal reconstruction recipe. Key facts:
   - **ξ_amp** = fractional deviation of amplitude: `(A_ecc - A_cir) / A_cir`
   - **ξ_ω** = fractional deviation of frequency: `(ω_ecc - ω_cir) / ω_cir`
   - **Relation III** (amplitude–frequency): `ξ_amp ≈ B · ξ_ω` with B ≈ 0.9 (empirical)
   - **Universality**: modulations from different (ℓ,m) modes collapse onto a single curve

2. **`EOB_modes.dat.m`** — Gamboa, Khalil & Buonanno supplementary Mathematica file. Contains `hFactEccCorr[2,2]` (the PN eccentric correction factor for the (2,2) mode) which provides the physics-motivated ansatz:
   ```
   ξ_amp_ansatz = |hFactEccCorr[2,2](e, x, ζ, ν)| - 1
   ```
   Lines L13232–13811 contain the full expression. This is the same factor already translated in `dyn_rewrite/src/ecc_mode_corrections.py`.

3. **`chandra_ecc_spin_paper_supplement.m`** — TaylorT2 time-to-merger expressions `te8` with eccentricity corrections to O(e₀⁸) and spin terms to 3PN. Useful for merger time estimation (mapping ODE time to peak-aligned waveform time).

4. **`tmerger_hinder_et_al.txt`** — Hinder et al. 2017 functional form for time to merger as a function of (q, e, l): `δt = δt₀ + a₁e + a₂e² + b₁q + b₂q² + c₁e·cos(l+c₂) + c₃eq`. Key insight: Fourier coupling in mean anomaly, `e·cos(l+phase)`, with no l-dependence at e=0.

---

## Inputs from dyn_rewrite

Our Numba dynamics (`dyn_rewrite/src/dynamics.py`) produces:
- `e(t)` — eccentricity evolution
- `x(t)` — orbit-averaged frequency parameter
- `ζ(t)` — relativistic anomaly
- `ω(t)` — instantaneous orbital frequency

These are used **both for training and inference** — the model is trained on our ODE output, so it learns to work with our dynamics (including any systematic differences from pySEOBNR). pySEOBNR is only needed to generate the reference waveforms h22_ecc and h22_cir during training.

---

## Parameter Space

```
q     ∈ [1, 10]
χ₁    ∈ [-0.5, 0.5]
χ₂    ∈ [-0.5, 0.5]
e₀    ∈ [0.0, 0.5]
ζ₀    = 0    (fix for now, extend later)
ω₀    = 0.009 (standard starting frequency)
```

---

## Modulation Decomposition

```
ξ_amp   = ξ_amp_ansatz(e, x, ζ, ν)   + δξ_amp(e, x, ζ, ν, χ_S, χ_A)
ξ_ω     = ξ_ω_ansatz(e, x, ζ, ν)     + δξ_ω(e, x, ζ, ν, χ_S, χ_A)
```

### Ansatz (physics-motivated baseline)

The (2,2) eccentric correction factor `hFactEccCorr[2,2]` from Gamboa et al. provides:
```
ξ_amp_ansatz = |hFactEccCorr[2,2](e, x, ζ, ν)| - 1
ξ_ω_ansatz   = ξ_amp_ansatz / 0.9   (from Relation III, recalibrate with data)
```

This is the same eccentric correction we already translated in `dyn_rewrite/src/ecc_mode_corrections.py` — it captures the leading PN eccentricity modulations at machine precision.

### Residual (learned correction)

The residual `δξ` captures:
- Spin effects (not in the non-spinning ansatz)
- Higher-order PN terms missing from the ansatz
- Resummation differences between ansatz and full SEOB
- Any systematic ODE dynamics offsets (absorbed implicitly since we train on our ODE output)

Fit with Ridge regression on a polynomial × Fourier basis:
```
δξ = Σ c_{abcdfk} · e^a · x^b · ν^c · χ_S^d · χ_A^f · {1, cos(kζ), sin(kζ)}
```

Key constraint: `a ≥ 1` (modulations vanish at e=0).

---

## Project Structure

```
modulation_learning/
├── workflow.md                     # This file
├── context/
│   └── gwnrxhme.md                # gwNRXHME framework reference
├── scripts/
│   ├── generate_data.py            # LHC sampling + SEOB waveforms + our ODE dynamics
│   ├── fit_model.py                # Ridge fitting with basis construction
│   ├── evaluate.py                 # Waveform reconstruction + mathcalE
│   ├── make_plots.py               # All Nature-quality plots
│   └── run_all.py                  # End-to-end pipeline
├── results/
│   ├── *.pdf / *.png               # All plots (Nature-quality)
│   ├── *.json / *.pkl              # All data for re-plotting
│   ├── training_data.pkl           # 100+ training waveforms + ODE dynamics
│   ├── validation_data.pkl         # 100+ validation waveforms + ODE dynamics
│   ├── best_model.pkl              # Ridge coefficients + basis config
│   ├── progress_log.json           # Per-approach accuracy + timing
│   └── progress_log.md             # Detailed narrative log (insights, bugs, decisions)
```

---

## Logging

**Every script must write to `results/progress_log.md`** — a running narrative of insights, results, intermediate outputs, bugs encountered, and decisions made. This is the primary record of the development process.

The log should include:
- Intermediate numerical results (print them AND log them)
- Intuitions about why something works or doesn't
- Parameter choices and their justification
- Error distributions and what they reveal about the physics
- Timing breakdowns
- Any unexpected behavior or surprises

Format: append timestamped entries like:
```markdown
## [Step N: description] — YYYY-MM-DD HH:MM

### What was done
...
### Results
...
### Insights
...
```

Also maintain `results/progress_log.json` for machine-readable progress tracking (per-approach mathcalE, timing, model size).

---

## Error Metrics

Two metrics must be computed and reported for every model evaluation:

### 1. mathcalE — time-domain L2 norm error
(Eq 21 of arxiv:1701.00550)

```python
def mathcalE_error(h_ref, h):
    n1 = np.sum(np.abs(h_ref)**2)
    n2 = np.sum(np.abs(h)**2)
    sdot = np.real(np.sum(h_ref * np.conj(h)))
    return ((n1 + n2) - 2 * sdot) / (2 * n1)
```

### 2. MM — time-domain mismatch

```python
def compute_mismatch(h_pred, h_ref):
    inner = np.real(np.sum(h_pred * np.conj(h_ref)))
    norm_pred = np.linalg.norm(h_pred)
    norm_ref = np.linalg.norm(h_ref)
    if norm_pred > 0 and norm_ref > 0:
        return 1.0 - inner / (norm_pred * norm_ref)
    else:
        return 1.0
```

Both metrics must be reported (median + max) on training AND validation for every approach. Phase-align both waveforms (set initial phase to zero) before computing either metric.

### Reconstruction Requirements

1. **Dense interpolation**: All modulations (ξ_amp, ξ_ω) and waveforms (h22_ecc, h22_cir, h22_pred) must be interpolated onto a dense grid with **dt=0.1M** before computing any error metric. This ensures the phase integration is accurate.

2. **Smooth tapering to zero**: Modulations must NOT be hard-clipped at t=0. Instead, apply a **smooth half-cosine taper** from t=-50M to t=0:
```python
def smooth_taper(t, t_start=-50.0, t_end=0.0):
    w = np.ones_like(t)
    mask = (t >= t_start) & (t <= t_end)
    w[mask] = 0.5 * (1.0 + np.cos(np.pi * (t[mask] - t_start) / (t_end - t_start)))
    w[t > t_end] = 0.0
    return w
```
This ensures continuity at the merger transition and prevents Gibbs-like artifacts.

3. **Modulation error metrics**: Compute ξ_amp and ξ_ω mathcalE on the dense grid, **excluding the taper region** (t < -50M only). This ensures the modulation fit quality reflects the inspiral region where the model is active.

---

## Implementation Steps

### Step 1: Training Data Generation (`scripts/generate_data.py`)

**Sampling**: Latin Hypercube over 4D parameter space:
```python
from scipy.stats.qmc import LatinHypercube

sampler = LatinHypercube(d=4, seed=42)
samples = sampler.random(n=100)

q     = 1 + 9 * samples[:, 0]          # [1, 10]
chi1  = -0.5 + 1.0 * samples[:, 1]     # [-0.5, 0.5]
chi2  = -0.5 + 1.0 * samples[:, 2]     # [-0.5, 0.5]
e0    = 0.5 * samples[:, 3]            # [0, 0.6]
```

- **100 training points** + **100 validation points** (separate LHC draws)
- ζ₀ = 0, ω₀ = 0.009

**For each point**:
1. Run pySEOBNR eccentric → h22_ecc(t), and extract ICs (r0, pr0, pphi0)
2. Run pySEOBNR circular → h22_cir(t)  (cache per unique (q, χ₁, χ₂))
3. Run **our ODE** (from `dyn_rewrite/src/dynamics.py`) with pySEOBNR ICs → (e_ode, x_ode, ζ_ode)(t)
4. Time-align ODE onto pySEOBNR's waveform time grid using `t_ecc[0]`
5. Compute modulations on common grid:
   - `ξ_amp = (|h22_ecc| - |h22_cir|) / |h22_cir|`
   - `ξ_ω = (ω_ecc - ω_cir) / ω_cir`  (from finite differences of phase)
6. Downsample to ~500–4000 points per waveform
7. Store: params, **our ODE dynamics** (e, x, ζ, ν, χ_S, χ_A) arrays, modulations, h22 waveforms, time shift

**Critical**: the (e, x, ζ) used as features come from **our ODE**, not pySEOBNR dynamics. This ensures the model learns to work with the dynamics it will receive at inference.

**Output**: `results/training_data.pkl`, `results/validation_data.pkl`

**Plots** (generated immediately):
- `results/parameter_space.pdf` — scatter of (q, e₀) colored by χ_eff

**Log**: record number of successful/failed waveforms, parameter ranges, timing per waveform.

---

### Step 2: Feature Importance Analysis

Before fitting, train a Random Forest on the raw features to identify which terms matter most.

**Features**: e, x, ν, χ_S, χ_A, cos(ζ), sin(ζ), cos(2ζ), sin(2ζ), ...

**Output**: `results/feature_importance.pdf` — bar charts for δξ_amp and δξ_ω

**Log**: which features dominate, whether spin matters, whether harmonics beyond k=2 are needed.

---

### Step 3: Ansatz Evaluation

Compute ξ_ansatz from the eccentric mode correction factor. The (2,2) eccentric correction `hFactEccCorr[2,2]` from `context/EOB_modes.dat.m` (lines L13232–13811) provides a physics-motivated baseline:
```
ξ_amp_ansatz = |hFactEccCorr[2,2](e, x, ζ, ν)| - 1
```

This is already translated in `dyn_rewrite/src/ecc_mode_corrections.py`. Evaluate mathcalE using ansatz-only reconstruction (no learned correction). This is the baseline.

**Output**: baseline mathcalE → logged as Step 0 in progress plot.

**Log**: ansatz-only mathcalE distribution (median, max, worst case params), how well Relation III holds.

---

### Step 4: Model Fitting (`scripts/fit_model.py`)

**Read the context files** (`context/gwnrxhme.md`, `context/EOB_modes.dat.m`) to understand the physics of the modulations before designing the basis or choosing a method. The modulations have known structure:
- Oscillate with the relativistic anomaly ζ at orbital frequency (and harmonics)
- Scale with eccentricity e (vanish at e=0)
- Depend weakly on spin (feature importance shows <0.2%)
- The PN structure from `hFactEccCorr[2,2]` captures the leading behavior

**Use your judgment** to design a suitable basis and try multiple fitting methods. The goal is to find the approach with the best mathcalE on validation. Here are methods to try — but feel free to explore others or modify these:

| # | Method | Notes |
|---|--------|-------|
| 0 | Ansatz only | No correction — baseline |
| 1 | Ridge regression | Polynomial × Fourier basis in (e, x, ν, χ_S, χ_A, ζ). Constraint: basis functions must have a ≥ 1 power of e (modulations vanish at e=0). Vary basis size, regularization alpha, number of harmonics. |
| 2 | Polynomial regression | sklearn PolynomialFeatures + Ridge on raw features (e, x, ν, χ_S, χ_A, cos(kζ), sin(kζ)). Try degree 3–6. |
| 3 | Random Forest | sklearn RandomForestRegressor, n_estimators=100–500. Use as accuracy reference — if RF beats Ridge significantly, the basis is missing important features. |
| 4 | Gaussian Process Regression | sklearn GaussianProcessRegressor. May capture nonlinear interactions better but expensive for large datasets. Subsample if needed. |
| 5 | Discrete Empirical Interpolation (DEIM) | Greedy algorithm: select interpolation points in parameter space, build interpolant. Good for smooth functions with known structure. |
| 6 | SVD / POD decomposition | Treat modulations as a matrix (waveforms × time points), compute SVD, project onto top-k modes. The coefficients depend on (q, χ₁, χ₂, e₀) — fit those with Ridge or GPR. |
| 7 | Neural network / MLP | Small network (2–3 hidden layers) as a nonlinear function approximator. Use with caution — needs careful regularization with limited data. |
| 8 | Hybrid: ansatz + best correction model | Combine the PN ansatz with the best-performing correction method. This decomposes the problem into physics (ansatz) + learned residual. |

**For each approach**: report mathcalE (median, max) on training AND validation, number of parameters, evaluation time, and any insights about what the model captures or misses.

**Save every model**: Save the best model from each approach to `results/models/` for future loading:
```
results/models/
├── model_ridge_small.pkl       # Ridge (small basis)
├── model_ridge_large.pkl       # Ridge (large basis)
├── model_polynomial.pkl        # Polynomial regression
├── model_rf.pkl                # Random Forest
├── model_gpr.pkl               # Gaussian Process
├── model_svd.pkl               # SVD/POD decomposition
├── model_deim.pkl              # DEIM
├── model_mlp.pkl               # Neural network
├── model_hybrid.pkl            # Ansatz + best correction
└── best_model.pkl              # Symlink or copy of the overall best
```

Each saved model must include everything needed to evaluate it at inference: coefficients, basis configuration, scaler parameters, etc. Include a `predict(e, x, zeta, nu, chiS, chiA)` interface or equivalent.

**Model preference**: Analytical/closed-form models (Ridge, polynomial) are **strongly preferred** over black-box models (RF, GPR, MLP) when accuracy is comparable, because:
- Faster evaluation (dot product vs tree traversal or kernel computation)
- Easier to Numba-compile for the final pipeline
- More interpretable (can inspect which basis terms matter)
- Smaller storage footprint

Only use RF/GPR/MLP if they provide a significant accuracy advantage (>2x better mathcalE) over analytical alternatives.

**Waveform-level CV**: When tuning hyperparameters, cross-validate at the waveform level (not point level) to prevent overfitting to correlated time-series data.

**Key constraint**: modulations must vanish at e=0. Any model should enforce or respect this.

**Log everything**: which methods were tried, why some worked better than others, what the error distributions look like, what the worst cases have in common.

---

### Step 4b: Per-Model Diagnostics

**Every model** (Ridge, polynomial, RF, GPR, SVD, DEIM, MLP, hybrid, etc.) must produce a full set of diagnostic plots saved to its own subdirectory:

```
results/common/
├── parameter_space.pdf                 # Scatter of training+validation in (q, e₀), colored by χ_eff
├── wf_length_histogram.pdf             # Histogram of waveform lengths in M
├── feature_importance.pdf              # RF feature importances for ξ_amp and ξ_ω
├── modulation_examples.pdf             # Example ξ_amp, ξ_ω for 3-4 representative waveforms (raw data, no model)
├── training_summary.json               # N_train, N_val, param ranges, timing stats
```

These plots are **model-independent** — they describe the training data itself. Generate them once before fitting any models.

```
results/models/<model_name>/
├── model.pkl                           # Saved model for future loading
├── histogram_mathcalE.pdf              # mathcalE + MM distributions (train blue, val red)
├── histogram_modulation_error.pdf      # Residual distribution: ξ_amp_pred - ξ_amp_true, ξ_ω_pred - ξ_ω_true
├── mathcalE_vs_e0.pdf                  # mathcalE vs e0 colored by q
├── mathcalE_vs_q.pdf                   # mathcalE vs q colored by e0
├── feature_importance.pdf              # Feature importances (if applicable — RF, Ridge coeff magnitudes)
├── basis_vectors_vs_time.pdf           # Top basis functions vs time for a representative case (if applicable)
├── modulation_fit_quality.pdf          # ξ_amp and ξ_ω: predicted vs true scatter plot (all points)
├── best_modulation.pdf                 # 3-panel (ξ_amp, ξ_ω, Re(h22)) for lowest-error case
├── median_modulation.pdf               # Same for median-error case
├── worst_modulation.pdf                # Same for worst-error case
├── waveform_best.pdf                   # Full h22 waveform overlay for best case (up to t=100M)
├── waveform_worst.pdf                  # Full h22 waveform overlay for worst case (up to t=100M)
└── summary.json                        # All metrics: mathcalE/MM median/max/std, n_params, eval_time_ms
```

For the **modulation panels** (best/median/worst), use a **split layout** for each row:
- **Left 60%**: inspiral (t < -250M) — shows the long oscillatory modulation structure
- **Right 40%**: late inspiral + merger + ringdown (t ≥ -250M to t = 100M) — shows the critical merger region at higher resolution

Each plot has 3 rows × 2 columns (6 sub-panels):
```
Row 1: ξ_amp(t)   | [inspiral, t < -250M]  | [merger, -250M < t < 100M]
Row 2: ξ_ω(t)     | [inspiral, t < -250M]  | [merger, -250M < t < 100M]
Row 3: Re(h22)(t)  | [inspiral, t < -250M]  | [merger, -250M < t < 100M]
```

- Left panels: x-axis in t/1000M, showing the full inspiral structure
- Right panels: x-axis in t/M, showing the merger transition at high resolution
- In each sub-panel: SEOB data (black), model (red dashed)
- Title includes parameters (q, χ₁, χ₂, e₀) and all error metrics (mathcalE, MM, max Δφ, ξ_amp mathcalE, ξ_ω mathcalE)

Use `fig, axes = plt.subplots(3, 2, figsize=(8, 9), gridspec_kw={'width_ratios': [3, 2]})`

For **basis_vectors_vs_time.pdf**: pick a representative waveform, evaluate the top 5–10 basis functions vs time, show which terms dominate the prediction. This helps understand what the model learns. Skip for models without explicit basis (RF, GPR, MLP).

---

### Step 4c: Model Comparison (`results/comparison/`)

After all models are fitted, generate a unified comparison:

```
results/comparison/
├── comparison_table.pdf                # Table: model name, n_params, eval_time, mathcalE/MM train/val median/max
├── comparison_histogram.pdf            # Overlaid mathcalE histograms for all models
├── comparison_vs_e0.pdf                # mathcalE vs e0, one curve per model
├── comparison_vs_q.pdf                 # mathcalE vs q, one curve per model
├── comparison_pareto.pdf               # Pareto: mathcalE vs eval_time (accuracy vs speed tradeoff)
├── comparison_best_worst.pdf           # For the 3 worst cases: overlay predictions from all models
├── progress.pdf                        # Staircase: val median mathcalE per approach, running best, target line
└── comparison_summary.json             # Machine-readable: all metrics for all models
```

The **progress plot** (`progress.pdf`) shows:
- One bar per approach (ansatz-only, Ridge small, Ridge large, polynomial, RF, GPR, SVD, DEIM, MLP, hybrid)
- Bar height = validation median mathcalE (log scale)
- Running-best staircase line in red
- Horizontal target lines at 1e-2 and 1e-3
- Bars colored: blue for analytical models, gray for black-box models
- Bars greyed out if the model is rejected (fails validation or too slow)

The **Pareto plot** shows accuracy (val median mathcalE) vs speed (eval time per waveform), identifying the efficient frontier.

---

### Logging Requirements

**IMPORTANT**: The progress log is the primary record for writing the paper. It should contain enough detail that someone can reconstruct the full story of which methods were tried, why each succeeded or failed, what the key insights were, and how the final model was selected. Write it as if you're drafting the methods/results sections of a paper — include quantitative comparisons, state conclusions clearly, and explain the physics behind why certain approaches work better.

**EVERYTHING goes in the log** — not just model results, but also:
- Data generation stats (N_train, N_val, failures, timing, param coverage)
- Feature importance analysis (which features matter, which don't)
- Common diagnostic observations (waveform length distribution, modulation amplitude ranges)
- Every model attempt with full metrics
- Comparison insights (why model A beats model B)
- Final model selection reasoning

**Progress plot** (`results/comparison/progress.pdf`) must be regenerated after every new model is fitted. It shows the running best across all approaches, making it easy to see the optimization trajectory at a glance. The x-axis lists every approach chronologically, the y-axis shows validation metrics on a log scale.

**`results/progress_log.md`**: Running narrative. Every model gets a timestamped section:
```markdown
## [Model: Ridge (small basis)] — YYYY-MM-DD HH:MM

### Configuration
  Basis: max_e=4, max_x=3, n_harm=5, 990 features
  Alpha: 1e-6 (from CV scan)

### Results
  Train mathcalE: median=X, max=Y
  Val   mathcalE: median=X, max=Y
  Train MM: median=X
  Val   MM: median=X
  Eval time: Z ms/waveform
  N parameters: W

### Worst cases
  q=5.7, e0=0.49, chi=(0.3,-0.2): mathcalE=0.84
  ...

### Insights
  - The model captures the oscillatory structure well but misses the secular drift at high q
  - Increasing n_harm from 5 to 8 improved median by 15% but didn't help the worst cases
  - ...

### Decision
  KEPT / REJECTED — reason
```

**`results/progress_log.json`**: Machine-readable array of all approaches:
```json
[
  {"model": "ansatz_only", "val_median_E": 1.1, "val_max_E": 2.3, "val_median_MM": 1.0, "n_params": 0, "eval_ms": 0.1, "kept": false},
  {"model": "ridge_small", "val_median_E": 0.085, ...},
  ...
]
```

**Log**: per-approach mathcalE (median, max), optimal alpha, number of basis functions, CV scores, timing breakdown (basis construction + predict).

---

### Step 5: Waveform Reconstruction & Evaluation (`scripts/evaluate.py`)

```python
# From learned model (using our ODE dynamics as features)
ξ_amp_pred  = ξ_amp_ansatz + δξ_amp_model(basis)
ξ_ω_pred    = ξ_ω_ansatz + δξ_ω_model(basis)

# Reconstruct waveform
A_pred   = A_cir * (1 + ξ_amp_pred)
ω_pred   = ω_cir * (1 + ξ_ω_pred)
φ_pred   = cumulative_trapezoid(ω_pred, t) + φ₀
h22_pred = A_pred * exp(i φ_pred)

# Phase-align and compute error
mathcalE = mathcalE_error(h22_ref, h22_pred)
```

Evaluate on both training and validation sets.

**Log**: per-waveform mathcalE, identify worst cases and why they fail, error vs (q, e0, chi).

---

### Step 6: Plots (`scripts/make_plots.py`)

All Nature-quality, saved as PDF+PNG with data in JSON for re-plotting.

1. **`parameter_space.pdf`** — scatter of training+validation in (q, e₀), colored by χ_eff
2. **`feature_importance.pdf`** — RF feature importances for δξ_amp and δξ_ω
3. **`histogram_mathcalE.pdf`** — histogram of log₁₀(mathcalE), training (blue) + validation (red)
4. **`mathcalE_vs_q_e0.pdf`** — scatter in (q, e₀) plane, colored by log₁₀(mathcalE)
5. **`mathcalE_vs_e0.pdf`** — log₁₀(mathcalE) vs e₀, colored by q
6. **`best_modulation.pdf`** — 3-panel for lowest-error case:
   - Panel 1: ξ_amp(t) — SEOB (black), ansatz (blue dashed), model (red)
   - Panel 2: ξ_ω(t) — same
   - Panel 3: Re(h22) last 3000M — data vs reconstructed
7. **`median_modulation.pdf`** — same for median-error case
8. **`worst_modulation.pdf`** — same for worst-error case
9. **`progress.pdf`** — staircase: median mathcalE per approach, target line, running best
10. **`wf_length_histogram.pdf`** — histogram of waveform lengths

---

## Accuracy & Speed Targets

| Metric | Target |
|--------|--------|
| **Δφ (phase error)** | **< π/4 radians** across inspiral |
| **Waveform mathcalE** | **< 1% (0.01)** |
| **ξ_ω mathcalE (frequency modulation)** | **< 1% (0.01)** |
| **ξ_amp mathcalE (amplitude modulation)** | **< 0.1% (0.001)** |
| Modulation eval time (500 pts) | < 15 ms |

The amplitude modulation target (0.1%) is tighter than frequency because amplitude errors don't accumulate, while frequency errors integrate into phase. The phase target (π/4) is the physically meaningful constraint — it determines whether the reconstructed waveform is in phase with the reference.

**Report all four metrics** for every model. The phase error should be computed as:
```python
phi_ecc = np.unwrap(np.angle(h_ecc))
phi_pred = np.unwrap(np.angle(h_pred))
delta_phi = phi_pred - phi_ecc  # after initial alignment
max_delta_phi = np.max(np.abs(delta_phi))
```
| Modulation eval time (ansatz + Ridge, 500 pts) | < 15 ms |

---

## Key Design Decisions

1. **Train on our ODE dynamics, not pySEOBNR dynamics**: Both training and inference use (e, x, ζ) from our Numba ODE. This ensures consistency — the model learns to work with our dynamics, including any systematic differences from pySEOBNR (e.g., the ~1.4% flux truncation). If we trained on pySEOBNR dynamics but inferred on ours, the model would see out-of-distribution inputs.

2. **Ansatz + residual decomposition**: The PN eccentric correction captures leading-order physics; the Ridge regression only needs to learn the small spin-dependent and resummation residual.

3. **a ≥ 1 constraint**: All basis functions have at least one power of e, ensuring modulations vanish identically at e=0. This is physically required and dramatically improves extrapolation.

4. **Fit ξ_amp and ξ_ω separately**: Though related by B≈0.9, fitting independently avoids propagating errors between amplitude and frequency.

5. **Waveform-level CV**: Cross-validation at the waveform level (not point level) prevents overfitting to correlated time-series data within a single waveform.

6. **Downsample to 500–4000 points**: Keeps fitting tractable (100 waveforms × 500 pts = 50,000 training points) while preserving the oscillatory structure.

---

## Connection to dyn_rewrite

The dynamics from `dyn_rewrite` are used as input features everywhere:
- **Training**: Our ODE provides (e, x, ζ)(t) that are used as features for the basis. pySEOBNR ICs are used to start the ODE, but the integration itself is ours.
- **Inference**: The full standalone pipeline is: our ODE dynamics → ansatz + Ridge model → waveform reconstruction. No pySEOBNR needed.

The ~1.4% flux truncation error in `dyn_rewrite` means our (e, x, ζ) deviate slightly from pySEOBNR's. By training on our ODE output, the Ridge model implicitly absorbs this difference.

---

## Environment

```
conda activate kitp-py310
```

pySEOBNR required for training data generation (reference waveforms). At inference, only our ODE + model needed.
