# Workflow: Timing Optimization

## Goal

Take the best model from the modulation learning workflow, benchmark its end-to-end inference time against pySEOBNR, identify bottlenecks, optimize until the model is faster than pySEOBNR for all parameter combinations, and document everything.

---

## Context

The best model is `ridge_nh7_me5_mchi1_a1e-06+phase_corr` — a 2955-feature Ridge regression on residuals with a polynomial phase correction. Initial benchmarking revealed:

- **ODE integration**: 1.2 ms median (fast, Numba-JIT)
- **Basis construction**: ~640 ms median (slow, pure Python loops building (N, 2955) matrix)
- **Ridge prediction**: ~10 ms (fast, matrix multiply)
- **Reconstruction**: ~23 ms (CubicSpline + phase integration)
- **Total**: ~668 ms median — **comparable to or slower than pySEOBNR** (~200-400 ms)

The bottleneck is `build_basis()` — a pure-Python nested loop that constructs polynomial x Fourier features at every time point. Reconstruction (CubicSpline) is also slow at ~23ms. Both must be optimized.

**Target: 10-20 ms end-to-end** (comparable to a single circular waveform lookup + fast predict). This requires ~30-60x speedup over baseline.

---

## Output Directory

All results go to `timing_optimization/`. Structure:

```
timing_optimization/
├── workflow_timing.md              # This file
├── optimize.py                     # Main script (does everything)
├── results/
│   ├── baseline_timing.json        # Pre-optimization timing (all val waveforms)
│   ├── optimized_timing.json       # Post-optimization timing (all val waveforms)
│   ├── pyseobnr_timing.json        # pySEOBNR reference timing
│   ├── optimization_log.json       # Log of each optimization step
│   ├── accuracy_check.json         # Verify optimized model matches original
│   ├── timing_comparison.pdf       # Bar chart: our model vs pySEOBNR per case
│   ├── timing_vs_params.pdf        # Scatter: our time vs q, e0, chi_eff, wf_length
│   ├── speedup_histogram.pdf       # Histogram of speedup over pySEOBNR
│   ├── optimization_progress.pdf   # Staircase of timing improvements
│   ├── breakdown_pie.pdf           # Pie chart: time breakdown (before/after)
│   └── summary.json                # Final summary with all numbers
└── optimized_model/
    ├── model.pkl                   # Optimized model object
    └── predict.py                  # Standalone prediction function (Numba-compiled)
```

---

## Optimization Strategy

Apply optimizations in sequence, benchmarking after each. Stop when the model is consistently faster than pySEOBNR.

### Level 0: Baseline
Measure the current end-to-end time on all 150 validation waveforms. Measure pySEOBNR on the same set of parameters (or a representative subset).

### Level 1: Downsample + Interpolate
The basis is evaluated at every time point (10k-70k points at dt=1M). Instead:
1. Evaluate basis at N_ds = 1000 evenly-spaced points
2. Predict delta_xi at those points
3. Interpolate delta_xi back to the full grid with `np.interp`

This reduces the basis matrix from (N, 2955) to (1000, 2955). Expected speedup: ~20-50x on the predict step.

### Level 2: Precompute Basis Column Specs
The nested loop in `build_basis()` recomputes the same power/harmonic structure every call. Instead:
1. At model load time, precompute a list of (a, b, c, d_s, d_a, k) tuples specifying each column
2. At predict time, vectorize over this list

### Level 3: Numba-JIT the Basis Construction
Compile the basis construction with `@numba.njit`. This eliminates Python loop overhead and enables SIMD vectorization.

### Level 4: Reduce Feature Count
If still slow, retrain a smaller Ridge model. From the scan results, `ridge_nh3_me3_mchi1` has only 714 features with dphi=0.317 (vs 0.315 for the 2955-feature model). Test whether a smaller basis with similar accuracy is faster end-to-end.

### Level 5: Numba-JIT the Full Predict Pipeline
Compile the entire predict function (basis + Ridge + ansatz) as a single Numba function. This avoids intermediate array allocations.

---

## Implementation: `optimize.py`

### Single Script Structure

```python
# Phase 0: Load model + data, define helper functions
# Phase 1: Baseline timing (all val waveforms + pySEOBNR subset)
# Phase 2: Apply optimizations sequentially, benchmark after each
# Phase 3: Accuracy verification (optimized must match original to <1e-10)
# Phase 4: Full timing on all val waveforms with optimized model
# Phase 5: Generate all plots and summary
```

### Timing Protocol
- Each waveform timed 3x, take median
- Report: ODE, interpolation, basis, predict, reconstruct, total
- pySEOBNR timed on 30 representative cases (diverse q, chi, e0) 3x each

### Accuracy Verification
After each optimization, verify:
```python
# For 20 random val waveforms:
xi_a_orig, xi_w_orig = predict_original(e, x, z, nu, chiS, chiA)
xi_a_opt, xi_w_opt = predict_optimized(e, x, z, nu, chiS, chiA)
assert np.max(np.abs(xi_a_orig - xi_a_opt)) < 1e-10
assert np.max(np.abs(xi_w_orig - xi_w_opt)) < 1e-10
```

### Plots

1. **timing_comparison.pdf**: Grouped bar chart. X-axis = parameter cases (sorted by pySEOBNR time). Two bars per case: pySEOBNR (blue) and our model (red). Horizontal line at y=pySEOBNR median.

2. **timing_vs_params.pdf**: 4-panel scatter. Our end-to-end time vs (a) q, (b) e0, (c) chi_eff, (d) waveform length. Color by speedup over pySEOBNR.

3. **speedup_histogram.pdf**: Histogram of (pySEOBNR time / our time) across all cases. Vertical line at speedup=1.

4. **optimization_progress.pdf**: Staircase bar chart. X-axis = optimization level (baseline, downsample, precompute, numba, ...). Y-axis = median end-to-end time. Include pySEOBNR median as horizontal line.

5. **breakdown_pie.pdf**: Two pie charts side by side. Left = before optimization (ODE, basis, predict, reconstruct). Right = after optimization. Shows where time went and how the bottleneck shifted.

### Summary JSON

```json
{
  "baseline": {
    "median_ms": ..., "mean_ms": ..., "max_ms": ...,
    "breakdown": {"ode": ..., "basis": ..., "predict": ..., "recon": ...}
  },
  "optimized": {
    "median_ms": ..., "mean_ms": ..., "max_ms": ...,
    "breakdown": {"ode": ..., "basis": ..., "predict": ..., "recon": ...}
  },
  "pyseobnr": {
    "median_ms": ..., "mean_ms": ..., "max_ms": ...
  },
  "speedup_over_pyseobnr": {
    "median": ..., "min": ..., "max": ...
  },
  "optimizations_applied": [...],
  "accuracy_preserved": true,
  "best_optimization_level": "...",
  "n_features_original": 2955,
  "n_features_optimized": ...,
  "n_val_waveforms": 150
}
```

---

## Accuracy Targets (must not degrade)

The optimized model must reproduce the original model's predictions to machine precision. Waveform-level metrics (mathcalE, dephasing, LIGO mismatch) must be identical. This is a code optimization, not a retraining — the Ridge coefficients are frozen.

If Level 4 (retrain smaller model) is needed, accuracy degradation is acceptable only if:
- Val dphi median < 0.20 rad (vs 0.144 for original)
- Val mathcalE < 1e-3 (vs 4.2e-4 for original)
- LIGO mismatch targets still met (>75% below 1%)

---

## Running

```bash
conda activate kitp-py310
cd modulation_learning/spin_05_04_26
python timing_optimization/optimize.py
```

Expected runtime: ~30-60 minutes (dominated by pySEOBNR timing of 30 cases x 3 repeats + full val set timing x 2).
