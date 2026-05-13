# Failure Modes and Human Oversight in Agentic Waveform Modeling

Comprehensive analysis of failure modes observed across all experiments in
`modulation_learning/`. Based on examining `spin_05_04_26` (Opus, successful),
`spin_05_04_26_haiku`, `spin_05_04_26_sonnet`, `spin_05_04_26_symbolic`,
`spin_10_05_26_without_ansatz`, and `spin_05_04_26/timing_optimization`.

---

## 1. Incorrect Waveform Reconstruction Formula (Haiku & Sonnet)

The most severe failure. Both Haiku and Sonnet implemented the phase
reconstruction incorrectly, producing models that appeared to complete
the workflow but were physically nonsensical.

**Haiku** (`spin_05_04_26_haiku/scripts/fit.py`, line 161):
```python
h_rec = h_cir_dense * (1.0 + xi_amp_dense) * np.exp(1j * np.cumsum(xi_omega_dense) * 0.1)
```
Applies `xi_omega` as a direct phase increment instead of a frequency
modulation. Produces dephasing of 1,400--26,844 rad (target: 0.1 rad).

**Sonnet** (`spin_05_04_26_sonnet/scripts/fit.py`, line 176):
```python
dphi = np.cumsum(omega_cir * xi_o_i) * dt_fine
phi_model = phi_cir + dphi
```
Applies the correction additively to circular phase instead of computing
`omega_pred = omega_cir * (1 + xi_omega)` and integrating from scratch.
Produces dephasing of 158--194 rad.

**Correct implementation** (Opus, `spin_05_04_26/scripts/fit.py`, line 206):
```python
pp = cumulative_trapezoid(oc * (1 + xi_w_d), dx=dt, initial=0.0)
```
This correctly integrates the full modulated frequency to obtain the phase.

**Impact**: Both weaker models declared "WORKFLOW COMPLETE" with all
checklist items passing (files existed, plots generated), but the
underlying physics was wrong. Checklists verified procedural correctness,
not physical correctness. The successful run only succeeded because the
agent derived the reconstruction formula correctly from the physics
specification.

**Quantitative comparison**:
| Agent  | Val dephasing (median) | Ratio to target |
|--------|----------------------|-----------------|
| Opus   | 0.144 rad            | 1.4x            |
| Sonnet | 158.7 rad            | 1,587x          |
| Haiku  | 26,844 rad           | 268,440x        |

---

## 2. Premature Termination with False Success Declaration

The Haiku log (`spin_05_04_26_haiku/tracking/CHANGELOG.md`) contains
**6 separate "WORKFLOW COMPLETE" declarations** across multiple runs, all
with catastrophic accuracy. The agent declared success because:
- All required files were created (checklist passes)
- The workflow script ran without Python errors
- It selected the "best" model from a uniformly bad set

The Sonnet run similarly declared "WORKFLOW COMPLETE, Best model:
ridge_poly_fourier" with 158.7 rad dephasing -- 1,587x worse than the
0.1 rad target.

**Lesson**: Procedural checklists (file existence, plot generation)
are necessary but not sufficient. Domain-aware validation that checks
whether metric values are physically reasonable is essential.

---

## 3. Reporting Anomalous Metrics Without Investigation

The Haiku log (`spin_05_04_26_haiku/tracking/progress_log.md`) at
timestamp `2026-04-09T12:03:54` shows one run reporting **dphi = 0.07 rad**
(suspiciously good), while all other runs from the same code produce
dphi = 411--1,435 rad. The agent did not flag or investigate this 6,000x
discrepancy between its own results.

Likely cause: that early run used a simplified metric computation that
did not properly integrate the phase, or evaluated on a different data
subset. The agent proceeded without investigating the contradiction.

**Lesson**: Agents should be required to flag and explain order-of-magnitude
discrepancies between runs on the same data.

---

## 4. Downsampling Accuracy Degradation in Timing Optimization

The timing optimization (`spin_05_04_26/timing_optimization/`) achieved
12x speedup over pySEOBNR by downsampling from full resolution to 500
points. The accuracy verification phase (optimization_log.json, line 36--40)
shows:

| Downsampling | median|dE|  | max|dE|      | Timing    |
|-------------|-------------|--------------|-----------|
| n_ds=1000   | 3.20e-05    | 4.29e-04     | 37.5 ms   |
| n_ds=500    | 2.46e-04    | 4.09e-03     | 19.1 ms   |
| n_ds=300    | 2.09e-03    | **3.57e-02** | 13.4 ms   |

The agent initially reported n_ds=300 as "best level" (fastest at 13.4ms)
despite its worst-case error (3.57e-2) being 85x worse than the model's
native accuracy (4.2e-4). The human-specified accuracy verification phase
constrained the final choice to n_ds=500.

**Lesson**: Speed optimization must be bounded by accuracy constraints
that only domain experts can define. The agent optimized for speed
without internalizing the accuracy floor.

---

## 5. Numba cache=True Creates Validation-Set-Specific JIT Artifacts

The timing optimization uses `@nb.njit(cache=True)` (lines 109, 117, 144,
158, 196 of `optimize.py`). Numba caches compiled functions to `__pycache__/`
based on function signature and input types. If the validation set has
uniform array lengths, the cached JIT code path is optimized for that
specific shape. When validation configurations change (different waveform
lengths, different number of time points), the cached code path may not
reflect general performance.

After human inspection revealed this issue, the workflow was updated to
require cache clearing and regeneration for new parameter configurations.

---

## 6. OOM Kills Without Graceful Degradation

The `spin_10_05_26_without_ansatz` experiment repeatedly hit macOS jetsam
(exit code 137) when:
- Building a 730-feature basis matrix for 300 waveforms (~1.4 GB)
- Running PyCBC FFTs on dense reconstructions concurrently
- The agent's initial approach (run everything in one process) exceeded
  available memory and was killed silently

The agent initially responded by retrying the same script. Human
intervention was required to:
1. Split the workflow into separate training and evaluation processes
2. Add `gc.collect()` after each waveform evaluation
3. Subsample training data (800 points per waveform instead of full ~3000)
4. Delete large intermediate arrays immediately after use

**Lesson**: Agents don't anticipate memory limits. They retry failing
commands without diagnosing the root cause. Memory management for
scientific computing workloads currently requires human intervention.

---

## 7. Model Complexity Selection Without Parsimony

The Ridge basis scan in `spin_05_04_26` (`tracking/progress_log.md`,
lines 119--131) found:

| Features | Val dphi | Complexity ratio |
|----------|----------|-----------------|
| 2955     | 0.315    | 4.1x            |
| 1379     | 0.317    | 1.9x            |
| **714**  | **0.318**| **1.0x**         |

The agent selected the 2955-feature model -- 4.1x more complex for a
0.3% accuracy improvement. This caused the timing bottleneck (575 ms
for basis construction alone, slower than pySEOBNR at 228 ms) that
required a separate optimization workflow to fix.

A domain expert exercising parsimony would have chosen the 714-feature
model, avoiding the entire timing optimization effort.

**Lesson**: Agents maximize the metric they are given without considering
downstream costs (inference speed, model complexity, interpretability).
Explicit parsimony criteria must be part of the objective.

---

## 8. Random Forest Overfitting Undetected

In `spin_05_04_26`, the Random Forest model shows:

| Metric    | Train      | Val        | Ratio |
|-----------|-----------|------------|-------|
| mathcalE  | 3.03e-3   | 4.01e-2   | 13.2x |
| Dephasing | 0.264 rad | 0.662 rad | 2.5x  |

A 13x train/val gap in the primary metric is severe overfitting. The agent
reported this model alongside others in the comparison table without
flagging the generalization failure, attempting regularization (fewer trees,
deeper minimum leaf size), or increasing training data.

In the without-ansatz experiment, the RF model similarly showed strong
training R^2 (0.9998) but validation-level metrics comparable to simpler
analytical models.

**Lesson**: Agents report metrics as instructed but don't independently
diagnose pathological patterns in those metrics.

---

## 9. Symbolic Regression Produces Physically Nonsensical Expressions

The `spin_05_04_26_symbolic` experiment ran PySR, gplearn, AI-Feynman,
and PyOperon on the modulation residuals. Results:

| Engine      | Val E   | Best expression                                      |
|-------------|---------|-----------------------------------------------------|
| gplearn     | 0.85    | `delta_xi_amp ~ -0.70 * e^2`                        |
| PySR        | 0.12    | Complex 39-term expression (R^2=0.93)               |
| AI-Feynman  | 0.33    | `e^2 * sqrt(exp(cos_zeta))`                         |
| Ansatz only | 1.05    | (baseline)                                           |
| **Ridge**   |**4.2e-4**| Human-specified 2955-feature basis                  |

PySR (best SR engine) is 300x worse than the Ridge model. The gplearn
expression captures only the leading-order term. AI-Feynman produces a
functional form (`sqrt(exp(cos_zeta))`) with no physical interpretation
in post-Newtonian theory.

None of the automated SR approaches discovered the physically meaningful
basis structure (Fourier harmonics in relativistic anomaly, PN parameter
coupling) that the human-specified ansatz provides.

**Lesson**: Symbolic regression is a useful exploratory tool but
currently cannot replace domain-informed basis specification for this
class of problems. The PN ansatz encodes physical knowledge that
generic function discovery cannot reproduce from data alone.

---

## 10. Phase Correction Overfitting

In `spin_10_05_26_without_ansatz`, a phase-correction RF trained on
50 waveforms achieved R^2 = 0.99 on those training waveforms but
produced **catastrophic degradation on validation** (dephasing went
from 0.38 to 1.48 rad -- 4x worse). The phase error structure from
a pointwise-fit model is highly waveform-specific; a generic correction
model memorizes the training set rather than learning transferable
corrections.

The Sonnet run's `ridge_phase_corrected` model similarly degraded:
val E went from 0.028 (ridge alone) to 0.063 (with phase correction),
and dephasing remained at 159 rad.

**Lesson**: Phase corrections that look excellent on training data can
be counterproductive on validation data. The success of phase correction
in the Opus run (dphi: 0.375 -> 0.144 rad) relied on a specific
polynomial form chosen by domain knowledge, not a generic ML correction.

---

## Summary: Human Oversight Requirements

| Category | What the human must verify | Why the agent can't |
|----------|---------------------------|---------------------|
| Physics correctness | Reconstruction formula implements omega_pred = omega_cir * (1 + xi_omega) correctly | Weaker models get the formula wrong; checklist can't catch physics errors |
| Metric sanity | Dephasing > 1 rad means the model is broken; > 100 rad means reconstruction is wrong | Agent treats all metric values equally, doesn't know physical bounds |
| Train/val integrity | Verify which samples contribute to which metrics | Agent sometimes confuses train/val or doesn't investigate anomalous discrepancies |
| Parsimony | Choose simplest model that meets accuracy targets | Agent maximizes accuracy without considering complexity/speed tradeoffs |
| Generalization | Flag > 3x train/val gap as overfitting | Agent reports metrics as instructed but doesn't diagnose pathologies |
| Memory management | Restructure compute for memory constraints | Agent retries failing commands without diagnosing root cause |
| Accuracy floors | Define acceptable accuracy degradation during optimization | Agent optimizes speed without internalizing accuracy requirements |
| Physical interpretability | Reject SR expressions with no physical meaning | Agent evaluates expressions by fit quality, not physical content |

---

## Experiment Directory Reference

| Directory | Agent | Best val dphi | Status |
|-----------|-------|--------------|--------|
| `spin_05_04_26` | Opus | 0.144 rad | Success |
| `spin_05_04_26_sonnet` | Sonnet | 158.7 rad | Failed (wrong reconstruction) |
| `spin_05_04_26_haiku` | Haiku | 26,844 rad | Failed (wrong reconstruction) |
| `spin_05_04_26_symbolic` | Opus | N/A (SR exploration) | Partial (SR < Ridge by 300x) |
| `spin_10_05_26_without_ansatz` | Opus | 0.31 rad (ensemble) | Completed but 16x worse than ansatz |
| `spin_05_04_26/timing_optimization` | Opus | N/A (speed) | 12x faster than pySEOBNR |
