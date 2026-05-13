# Global Workflow: Eccentric NR Waveform Surrogate

## Goal

Build a fast, self-contained surrogate for eccentric (2,2) waveforms validated
against NR simulations. Given `(q, e0, zeta0)`, predict the full h22 waveform.

## Pipeline Overview

```
Step 1: contexts/          Step 2: dynamics/          Step 3: modulation_learning/     Step 4: waveforms/
─────────────────          ─────────────────          ──────────────────────────────    ──────────────────
Read Mathematica     -->   Secular ODE integration    -->   Learn residuals in           -->  Assemble h22
files (EOB_*.m)            for SEOB resummed                xi_amp, xi_omega                 from modulations,
Parse & summarize          dynamics. Numba-            using NR sims + SEOB              test vs NR & SEOB,
into LLM context.          optimized (~5-8ms).         initial conditions.               compute mismatches.
                           Validate vs pyseobnr.       Explore: analytical,
                                                       neural ODE, PySR,
                                                       GPR, RF.
```

## Step Details

### Step 1 — Context Extraction (`contexts/`)

**Input**: `EOB_fluxes.dat.m`, `EOB_Keplerian.dat.m`, `EOB_modes.dat.m`,
plus `chandra_ecc_spin_paper_supplement.m` from ecc_wf_agents_v2/src/.

**Output**: `contexts/summary.md` — a structured summary of all PN expressions,
coordinate transformations, fluxes, waveform modes, and resummed eccentricity
corrections that downstream steps need.

**Why**: These Mathematica files contain the authoritative expressions. The
summary serves as LLM context so that Steps 2-4 can reference the correct
formulas without re-reading raw Mathematica.

### Step 2 — Dynamics (`dynamics/`)

**Input**: Expressions from Step 1 summary.

**Output**:
- `dynamics/secular_ode.py` — secular ODE integration (e, zeta, x evolution)
  using SEOB resummed expressions from the Mathematica files.
- `dynamics/secular_ode_numba.py` — Numba-JIT version targeting 5-8ms per
  integration with relaxed tolerance.
- Validation plots in `dynamics/results/` comparing against pyseobnr
  SEOBNRv5EHM dynamics.

**Key note**: The ODE will not exactly match pyseobnr because pyseobnr includes
additional features (post-adiabatic corrections, attachment, etc.). The goal is
to capture the secular evolution accurately enough for the modulation ansatz.

**Time alignment**: pyseobnr peak-aligns waveform time (`t_seob`), but dynamics
time (`t_dyn`) starts at 0. Correct alignment: `t_dyn_aligned = t_dyn + t_seob[0]`
(since `t_seob[0] = -t_peak_raw`). The approximate `t_dyn - t_dyn[-1]` is off
by ~1-2M.

### Step 3 — Modulation Learning (`modulation_learning/`)

**Input**: NR simulations (SXS catalog), SEOB initial conditions from
`ancillary_ecc_NR_sims.json`, dynamics from Step 2, H22ecc ansatz.

**Output**: Trained model(s) for residuals `delta_xi_amp`, `delta_xi_omega`
where:
```
xi_amp_NR   = xi_amp_ansatz + delta_xi_amp
xi_omega_NR = xi_omega_ansatz + delta_xi_omega
```

**Methods to explore** (preference: analytical > symbolic > ML):
1. Analytical expressions — use Mathematica context to identify basis structures
2. Symbolic regression (PySR) — discover compact expressions
3. Neural ODE — if dynamics-coupled residuals needed
4. GPR / RF — as baselines

**Definitions**:
- `xi_amp = (A_NR - A_cir) / A_cir` (NRHybSur3dq8 as circular baseline)
- `xi_omega = (omega_NR - omega_cir) / omega_cir`

### Step 4 — Waveform Assembly (`waveforms/`)

**Input**: Dynamics from Step 2, modulations from Step 3, circular baseline.

**Output**:
- `waveforms/assemble.py` — reconstruct h22:
  `h22_pred = A_cir * (1 + xi_amp) * exp(i * integral(omega_pred) dt)`
- Time-domain mismatch comparisons vs NR and SEOB
- Accuracy plots in `waveforms/results/`

**Metrics**:
- E_amp: L2 norm error on amplitude
- E_freq: L2 norm error on frequency
- E_phase: max |phase error| (radians)
- MM: mismatch = 1 - Re(<h_pred|h_ref>) / (||h_pred|| ||h_ref||)

## Parameter Space

| Parameter | Range | Notes |
|-----------|-------|-------|
| q         | [1, 10] | Extended from v5's [1,6] to cover NR catalog |
| e0        | [0.001, 0.4] | At reference frequency |
| zeta0     | [0, 2pi) | Relativistic anomaly |
| omega0    | 0.0085 (fixed) | Geometric units, M=1 |
| chi1, chi2 | 0 (fixed) | Non-spinning for now |

## Key Data Files

- `ancillary_ecc_NR_sims.json` — NR simulation metadata + optimized SEOB ICs
- `nr_simulations.txt` — list of SXS simulation IDs
- NR waveforms downloaded via `sxs.load()`
- Circular baseline from `NRHybSur3dq8` surrogate

## Progress Tracking

Every milestone logs two key metrics via `tracking/progress_log.py`:
- **Waveform generation time** (ms) — target: < 50 ms
- **Time-domain mismatch** (%) — target: < 0.1%

```python
from tracking.progress_log import log_entry
log_entry("step_name", "description", time_ms=..., mismatch_pct=..., kept=True)
```

Progress plots (Clax/AutoResearch style): `python tracking/plot_progress.py`
Overview figure (Nature style): `python figures/plot_overview.py`

Outputs go to `figures/agent_progress.{pdf,png}` and `figures/overview.{pdf,png}`.

## Dependencies

- pyseobnr, lal, lalsimulation, gwtools, gwsurrogate, sxs
- numba, numpy, scipy, matplotlib, scikit-learn
- pysr (for symbolic regression, Step 3)
- Conda env: `kitp-py310`
