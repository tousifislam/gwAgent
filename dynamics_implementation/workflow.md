# Fast Eccentric EOB Dynamics: Full Numba Rewrite Workflow

## Goal

Produce `(e(t), x(t), zeta(t))` matching pySEOBNR's full EOB dynamics **visually** on dense output grids, while achieving maximum speed. Find the optimal integration scheme, tolerances, and output strategy through systematic experimentation — all findings logged and shown in a cumulative progress plot.

---

## Guiding Principles

1. **Plots as you go**: Every step produces Nature-quality comparison plots immediately — do not defer plotting to the end.
2. **Optimize through experimentation**: Try multiple integration schemes (RK45, DOP853, RK8, fixed-step RK4, etc.), sweep rtol/atol, and find the sweet spot where dynamics still visually agree with pySEOBNR.
3. **Dense output**: The final output grid must be dense (matching pySEOBNR's output density or denser). Test both sparse and dense output modes.
4. **Progress plot is cumulative**: Every optimization experiment (scheme change, tolerance change, output mode change) adds a bar to the progress staircase. The progress plot is regenerated after every step.
5. **Save everything**: All results data (timing, errors, dynamics arrays) saved as JSON/PKL for future re-plotting.

---

## Project Structure

```
dyn_rewrite/
├── findings.md                          # Background research & profiling
├── workflow.md                          # This file
├── src/
│   ├── __init__.py
│   ├── hamiltonian.py                   # Numba evaluate_H + FD gradient
│   ├── fits.py                          # a6_NS, dSO calibration fits, GSF fits
│   ├── evolution_equations.py           # Numba edot/zdot/xavg (Keplerian eqs)
│   ├── waveform_modes.py               # rho_lm coefficients, Newtonian prefixes, tail
│   ├── ecc_corrections.py              # Eccentric mode corrections + RR force corrections
│   ├── flux.py                          # Numba flux computation (3 dominant modes)
│   ├── rr_force.py                      # Numba RR force Fr, Fphi
│   ├── initial_conditions.py            # IC solver (r0, pr0, pphi0)
│   ├── integrator.py                    # Numba integrator(s) — multiple schemes
│   ├── dynamics.py                      # Top-level: lean_setup + full integrate
│   └── utils.py                         # Shared constants, compute_x helper
├── comparison_scripts/
│   ├── compare_hamiltonian.py           # H, xi, gradients vs pySEOBNR
│   ├── compare_evolution_eqs.py         # edot, zdot, xavg vs pySEOBNR
│   ├── compare_flux.py                  # 3-mode flux vs pySEOBNR 35-mode flux
│   ├── compare_rr_force.py             # Fr, Fphi vs pySEOBNR
│   ├── compare_rhs.py                   # Full 6-vector RHS vs pySEOBNR get_rhs_ecc
│   ├── compare_ics.py                   # Initial conditions vs pySEOBNR
│   ├── compare_dynamics.py              # Full e(t), x(t), zeta(t) vs pySEOBNR (dense)
│   ├── benchmark_timing.py              # Timing histogram across parameter space
│   ├── optimize_tolerances.py           # rtol/atol sweep: find best speed with visual agreement
│   ├── compare_integrators.py           # Compare integration schemes (RK45, DOP853, RK8, ...)
│   └── make_progress_plot.py            # Cumulative staircase optimization progress
├── results/
│   ├── *.pdf / *.png                    # All plots (Nature-quality, generated incrementally)
│   ├── *.json / *.pkl                   # All data saved for re-plotting
│   ├── progress_log.json                # Cumulative log of every optimization experiment
│   └── summary.md                       # Key findings and decisions
```

---

## pySEOBNR Source Files (for translation)

Base path: `Research/projects/nr_projects/wf_agents/just_pyseobnr_rewrite/pyseobnr/pyseobnr/eob/`

| Component | Source file | Key functions/lines |
|-----------|-----------|-------------------|
| Hamiltonian | `hamiltonian/Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx` | `evaluate_H` (L23-168), `dynamics()` |
| Eccentric RHS | `dynamics/rhs_aligned_ecc.pyx` | `get_rhs_ecc()`, `compute_x()` |
| Integration loop | `dynamics/integrate_ode_ecc.py` | `compute_dynamics_ecc_opt()`, `ColsEccDyn` |
| IC solver | `dynamics/initial_conditions_aligned_ecc_opt.py` | `compute_IC_ecc_opt()` |
| Keplerian evol eqs | `dynamics/Keplerian_evolution_equations_flags/_implementation.pyx` | `_initialize()`, `_compute()` |
| Secular evol eqs | `dynamics/secular_evolution_equations_flags/_implementation.pyx` | `_initialize()`, `_compute()` |
| Flux (eccentric) | `waveform/waveform_ecc.pyx` | `compute_flux_ecc()` (L383-500), `RR_force_ecc()` (L318-375) |
| Waveform infrastructure | `waveform/waveform.pyx` | `compute_rho_coeffs()`, `compute_rholm()`, `compute_tail()`, `EOBFluxCalculateNewtonianMultipoleAbs()` |
| Ecc mode corrections | `waveform/modes_ecc_corr_NS_v5EHM_v1_flags/_implementation.pyx` | `_initialize()`, `_compute()`, `get(l,m)` |
| RR force corrections | `waveform/RRforce_NS_v5EHM_v1_flags/_implementation.pyx` | `_initialize()`, `_compute()`, `get("radial"/"azimuthal")` |
| Calibration fits | `fits/fits_Hamiltonian.py` | `a6_NS(nu)`, `dSO(nu, ap, am)` |
| GSF fits | `fits/GSF_fits.py` | `GSF_amplitude_fits(nu)` |

Old prototype code (read-only reference, not anchor code): `Research/projects/nr_projects/wf_agents/just_pyseobnr_rewrite/test/`
- `fast_ecc_dynamics.py` — lean_setup + scipy integration (current ~10-45ms baseline)
- `numba_eob/hamiltonian.py` — partial Numba Hamiltonian (evaluate_H + FD gradient, already working)

---

## Implementation Steps

Each step produces:
- Working Numba code in `src/`
- A comparison script in `comparison_scripts/` that generates Nature-quality plots
- Results data saved to `results/` (JSON/PKL for re-plotting, PDF/PNG plots)

---

### Step 1: Hamiltonian (`src/hamiltonian.py`, `src/fits.py`)

**What**: Translate `evaluate_H` (~80 lines of dense algebra) and finite-difference gradient.

**Source**: `Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx` lines 23-168. Prototype in `test/numba_eob/hamiltonian.py`.

**Translation**:
- `evaluate_H(r, prst, L, chi1, chi2, m1, m2, nu, a6, dSO)` → `(H, xi)`. All `@njit(cache=True, fastmath=True)`.
- Gradient via central finite differences: 4 extra `evaluate_H` calls → `(dHdr, dHdphi, dHdpr, omega, H_val, xi)`.
- `fits.py`: translate `a6_NS(nu)`, `dSO(nu, ap, am)`, `GSF_amplitude_fits(nu)`.

**Plots** (`compare_hamiltonian.py` → `results/`):
- Relative error histograms for H, xi, dHdr, dHdpr, omega (200+ random points).
- Save comparison data as JSON.

---

### Step 2: Evolution Equations (`src/evolution_equations.py`)

**What**: Translate Keplerian `edot`, `zdot`, `xavg_omegainst`.

**Source**: `Keplerian_evolution_equations_flags/_implementation.pyx`

**Translation**:
- `initialize_keplerian_coeffs(nu, delta, chiA, chiS)` → flat `float64[N]` array
- `compute_edot_zdot_xavg(e, z, omega, coeffs)` → `(edot, zdot, xavg_omegainst)`

**Plots** (`compare_evolution_eqs.py` → `results/`):
- Relative error of edot, zdot, xavg vs pySEOBNR at a grid of (e, z, omega) for several (q, chi1, chi2).
- Save comparison data.

---

### Step 3: Flux & Waveform Modes (`src/waveform_modes.py`, `src/ecc_corrections.py`, `src/flux.py`)

**What**: Translate flux computation, simplified to 3 dominant modes: (2,2) + (3,3) + (2,1).

**Source**: `waveform.pyx`, `waveform_ecc.pyx`, `modes_ecc_corr_NS_v5EHM_v1_flags/_implementation.pyx`

**Translation**:
- Precompute rho coefficients once: `compute_rho_coeffs_numba(...)` → arrays
- Per-mode: `rho_lm`, `tail_lm`, `newtonian_multipole`, `ecc_corr_lm`
- Full flux: `compute_flux_ecc_numba(...)` → scalar flux

**Plots** (`compare_flux.py` → `results/`):
- 3-mode flux vs pySEOBNR full 35-mode flux at grid of configurations.
- Quantify mode-truncation error (expect <1%).
- Save data.

---

### Step 4: RR Force (`src/rr_force.py`)

**What**: Translate radiation reaction force `(Fr, Fphi)`.

**Source**: `waveform_ecc.pyx` L318-375, `RRforce_NS_v5EHM_v1_flags/_implementation.pyx`

**Plots** (`compare_rr_force.py` → `results/`):
- Fr, Fphi relative error at grid of state points.
- Save data.

---

### Step 5: Full RHS Assembly (`src/dynamics.py`)

**What**: Assemble the complete 6-variable RHS.

```
rhs_ecc: (r, phi, pr, pphi, e, z) → (drdt, dphidt, dprdt, dpphidt, edot, zdot)
```

**Plots** (`compare_rhs.py` → `results/`):
- All 6 RHS components vs pySEOBNR's `get_rhs_ecc()` at identical state points.
- Save data.

---

### Step 6: Initial Conditions (`src/initial_conditions.py`)

**What**: Translate IC solver `(omega_start, e0, zeta0)` → `(r0, pr0, pphi0)`.

**Source**: `initial_conditions_aligned_ecc_opt.py`

**Plots** (`compare_ics.py` → `results/`):
- Table and bar chart of r0, pr0, pphi0 errors for all 8 validation cases.

---

### Step 7: Integrator & Optimization (`src/integrator.py`)

**What**: Implement multiple integration schemes in Numba, find the fastest that gives visual agreement with pySEOBNR on dense output grids.

**Schemes to implement and test**:
1. Adaptive RK45 (DOPRI5) — standard baseline
2. Adaptive DOP853 (8th order) — fewer steps but heavier per step
3. Fixed-step RK4 — no error control overhead, needs step-size tuning
4. Any other scheme that looks promising (e.g., RK8(5,3), Cash-Karp, BS3)

**Optimization experiments** (each becomes a bar in the progress plot):
- For each scheme: sweep rtol from 1e-10 down to 1e-4
- For each (scheme, rtol): measure timing AND visual agreement with pySEOBNR
- Test both sparse output (adaptive steps only) and dense output (interpolated onto pySEOBNR-like grid)
- Dense output cost: measure overhead of interpolation/evaluation at N_dense points
- **Selection criterion**: visually indistinguishable from pySEOBNR on e(t), x(t) plots at the fastest speed

**Plots** (`compare_integrators.py`, `optimize_tolerances.py` → `results/`):
- Scheme comparison: time vs max|Δe| Pareto frontier (one curve per scheme)
- Tolerance sweep: rtol vs time AND rtol vs accuracy for best scheme
- Dense vs sparse timing comparison
- Visual agreement panels: e(t), x(t) overlaid with pySEOBNR for the winning configuration

---

### Step 8: Full Dynamics Comparison & Benchmarking

**What**: Run the optimized pipeline end-to-end across all validation cases with dense output.

**Plots** (`compare_dynamics.py` → `results/`):
- **Multi-panel validation**: 8-case grid, each showing e(t) and x(t) overlaid with pySEOBNR (Nature-quality, similar to spin_apr1's `validation_spinning.pdf`)
- **Residual panels**: Δe(t) and Δx(t) for each case
- **e(x) trajectory**: parametric plot comparing orbital evolution

**Plots** (`benchmark_timing.py` → `results/`):
- Side-by-side horizontal bar chart: our timing vs pySEOBNR for all 8 cases
- Speedup factors annotated

**Plots** (`make_progress_plot.py` → `results/`):
- **Cumulative staircase progress plot** — every optimization experiment is a bar:
  - pySEOBNR full waveform (baseline, ~90-150 ms)
  - Lean setup + scipy DOP853 (from findings.md, ~10-45 ms)
  - Numba + scheme A, rtol=X (dense output)
  - Numba + scheme A, rtol=Y (dense output)
  - Numba + scheme B, rtol=X (dense output)
  - ... (every experiment that was tried)
  - Best configuration highlighted
- Horizontal dashed lines for pySEOBNR baseline and 5 ms target
- Running-best staircase line in red
- Speedup annotation
- Bars greyed out for configurations that fail visual agreement
- Data: `results/progress_log.json` (append after every experiment)

---

## Validation Cases

| q | chi1 | chi2 | e0 | Purpose |
|---|------|------|-----|---------|
| 1 | 0 | 0 | 0.3 | Non-spinning baseline |
| 3 | 0.5 | 0.3 | 0.2 | Moderate spin |
| 6 | 0.9 | 0 | 0.1 | High spin, asymmetric |
| 10 | 0.7 | 0.7 | 0.4 | High q, high e |
| 2 | 0 | 0.8 | 0.5 | Single-spin |
| 1 | 0.5 | 0.5 | 0.01 | Nearly circular |
| 5 | 0.3 | 0.1 | 0.6 | High eccentricity |
| 8 | 0.9 | 0.9 | 0.3 | Near-extremal spins |

---

## Plot Conventions (Nature-quality, used throughout)

```python
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'legend.frameon': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'lines.linewidth': 1.0,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})
COL_REF   = '#1a1a1a'   # near-black for pySEOBNR reference
COL_OURS  = '#d62728'   # red for our implementation
COL_FILL  = '#d6272820' # transparent red for error bands
COL_TARG  = '#2ca02c'   # green for targets
COL_BAR   = '#4c72b0'   # steel blue for bars
```

Every comparison script saves both PDF and PNG. Every script saves its raw data (JSON or PKL) for future re-plotting without re-running pySEOBNR.

---

## Progress Log Format

`results/progress_log.json` is a list of experiments, appended after each optimization step:

```json
[
  {"step": "pySEOBNR baseline", "time_ms": 120.0, "scheme": "GSL rk8pd", "rtol": "1e-11", "dense": true, "visual_ok": true},
  {"step": "Lean + scipy DOP853", "time_ms": 25.0, "scheme": "DOP853", "rtol": "1e-8", "dense": true, "visual_ok": true},
  {"step": "Numba RK45 rtol=1e-10", "time_ms": 6.0, "scheme": "RK45", "rtol": "1e-10", "dense": true, "visual_ok": true},
  {"step": "Numba RK45 rtol=1e-6", "time_ms": 2.0, "scheme": "RK45", "rtol": "1e-6", "dense": true, "visual_ok": true},
  {"step": "Numba RK45 rtol=1e-4", "time_ms": 1.0, "scheme": "RK45", "rtol": "1e-4", "dense": false, "visual_ok": false},
  ...
]
```

---

## Key Design Decisions

1. **Finite-difference gradient** instead of translating the 250-line analytical Jacobian — 4 extra `evaluate_H` calls at ~0.2 us each.
2. **3-mode flux** (2,2)+(3,3)+(2,1) instead of all 35 modes — captures >99% of flux.
3. **All `@njit(cache=True, fastmath=True)`** — first call compiles, subsequent calls run at native speed.
4. **Flat coefficient arrays** instead of class objects — Numba-friendly, zero Python dispatch.
5. **Multiple integrators tested** — pick the fastest with visual agreement, not a preset choice.
6. **Dense output** — final product must have dense time grid, cost of interpolation included in timing.

---

## Environment

```
conda activate kitp-py310
```

pySEOBNR must be importable for comparison scripts (validation against reference).

---

## Completed Results

### Final configuration: 16-mode flux
Modes: (2,2), (2,1), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (4,4), (5,2), (5,3), (5,4), (5,5), (6,6), (7,7), (8,8)

### All comparison scripts regenerated
- `comparison_scripts/compare_hamiltonian.py` → `results/compare_hamiltonian.pdf`
- `comparison_scripts/compare_evolution_eqs.py` → `results/compare_evolution_eqs.pdf`
- `comparison_scripts/compare_flux.py` → `results/compare_flux.pdf`
- `comparison_scripts/compare_dynamics.py` → `results/compare_dynamics.pdf`
- `comparison_scripts/compare_dynamics_extended.py` → `results/dynamics_residuals.pdf`, `results/error_summary.pdf`
- `comparison_scripts/plot_dt_vs_e0.py` → `results/dt_vs_e0.pdf`
- `comparison_scripts/make_progress_plot.py` → `results/progress.pdf`, `results/timing_histogram.pdf`
- `comparison_scripts/tolerance_sweep.py` → `results/tolerance_sweep.pdf`

### Code generation scripts
- `scripts/gen_mode_corrections.py` — auto-generates `src/ecc_mode_corrections.py` from pySEOBNR Cython
- `scripts/gen_ecc_corrections.py` — auto-generates `src/ecc_corrections.py` (RR force corrections)
