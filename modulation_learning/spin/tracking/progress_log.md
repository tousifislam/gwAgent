# Small-Spin Eccentric Modulation Learning — Progress Log

## [Step 1: Data Generation] — 2026-04-05 19:18

### Setup
  q in [1, 10], chi1/chi2 in [-0.5, 0.5], e0 in [0.001, 0.5], omega0 = 0.0085
  Ansatz decomposition: h22_ecc(x, e, zeta, nu) baseline + learned residuals
  300 training + 150 validation (4D LHC)

Warming up JIT...
  Done.

### Generating training data (300 points, seed=42)
  30/300: 30 ok, 0 fail
  60/300: 60 ok, 0 fail
  90/300: 90 ok, 0 fail
  120/300: 120 ok, 0 fail
  150/300: 150 ok, 0 fail
  180/300: 180 ok, 0 fail
  210/300: 210 ok, 0 fail
  240/300: 240 ok, 0 fail
  270/300: 270 ok, 0 fail
  300/300: 300 ok, 0 fail
  training: 300/300 successful, 0 failed, 206s
  Saved /Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/modulation_learning/spin_05_04_26/results/training_data.pkl
  Wf lengths: median=34092M, range=[10487, 72766]M
  SEOB timing: median=419ms | ODE: median=58ms
  Residual std: delta_xi_amp median=1.7768e-02, delta_xi_omega median=2.3737e-02

### Generating validation data (150 points, seed=123)
  30/150: 30 ok, 0 fail
  60/150: 60 ok, 0 fail
  90/150: 90 ok, 0 fail
  120/150: 120 ok, 0 fail
  150/150: 150 ok, 0 fail
  validation: 150/150 successful, 0 failed, 101s
  Saved /Users/tousifislam/Research/projects/nr_projects/wf_agents/agent_dyn_mod_wf/modulation_learning/spin_05_04_26/results/validation_data.pkl
  Wf lengths: median=34084M, range=[11240, 72038]M
  SEOB timing: median=426ms | ODE: median=60ms
  Residual std: delta_xi_amp median=1.6800e-02, delta_xi_omega median=2.2762e-02

### Generating common plots
  Saved parameter_space.pdf, parameter_space_spin.pdf, wf_length_histogram.pdf,
  wf_length_vs_q.pdf, wf_length_vs_chieff.pdf, modulation_examples.pdf,
  spin_effect_examples.pdf

### Feature importance (on residuals)
  Feature importances for delta_xi_amp:
                  e: 0.8555
          cos(zeta): 0.0683
                  x: 0.0312
          sin(zeta): 0.0192
                 nu: 0.0059
              chi_A: 0.0058
              chi_S: 0.0056
        cos(2*zeta): 0.0049
        sin(2*zeta): 0.0036
  Feature importances for delta_xi_omega:
                  e: 0.8015
          cos(zeta): 0.1159
                  x: 0.0272
          sin(zeta): 0.0192
        cos(2*zeta): 0.0091
                 nu: 0.0080
              chi_S: 0.0071
              chi_A: 0.0071
        sin(2*zeta): 0.0049
  Saved feature_importance.pdf

### Summary
  300 train + 150 val waveforms generated

Step 1 complete.


======================================================================
## [Small-Spin Model Fitting] — 2026-04-05 19:24
======================================================================

Loaded 300 train + 150 val

### Model 0: Ansatz only (delta=0)
  Evaluating on training set...
  Evaluating on validation set...
  Train: E=1.0318e+00 MM=1.0006e+00 dphi=27.842
  Val:   E=1.0457e+00 MM=1.0096e+00 dphi=24.185 <0.1:2% <0.5:5% mono_viol=0.000
  Val LIGO MM (Mtot=20): med=5.6126e-01 max=8.2312e-01 <0.01:7%
  Val LIGO MM (Mtot=65): med=5.1069e-02 max=6.5633e-01 <0.01:29%
  Val LIGO MM (Mtot=110): med=1.2171e-02 max=3.4742e-01 <0.01:43%
  Val LIGO MM (Mtot=155): med=9.1985e-03 max=7.3598e-02 <0.01:53%
  Val LIGO MM (Mtot=200): med=7.4330e-03 max=4.9593e-02 <0.01:62%
  Generating diagnostic plots...

  [2026-04-05 19:27] Model ansatz_only trained and evaluated
=== CHECKLIST: ansatz_only ===
[PASS] model.pkl saved
[PASS] error arrays saved (10 npy files)
[PASS] histogram.pdf
[PASS] histogram_ligo_mm.pdf
[PASS] dephasing_vs_e0.pdf
[PASS] dephasing_vs_chieff.pdf
[PASS] mathcalE_vs_e0.pdf
[PASS] mathcalE_vs_chieff.pdf
[PASS] ligo_mm_vs_e0.pdf
[PASS] ligo_mm_vs_chieff.pdf
[PASS] best_modulation.pdf
[PASS] median_modulation.pdf
[PASS] worst_modulation.pdf
[PASS] summary.json
[PASS] progress_log.md updated
[PASS] CHANGELOG.md updated
=== ALL 16 CHECKS PASSED ===

### Ridge basis scan (with spin features)
  n_harm=3: 72 configs tested
  n_harm=5: 144 configs tested
  n_harm=7: 216 configs tested

  Top 10 configs (of 216 passing monotonicity):
  n_harm max_e max_x  mchi    alpha    nf     dphi   mono
       7     5     3     1    1e-06  2955    0.315  0.000
       3     5     3     1    1e-06  1379    0.317  0.000
       7     3     3     1    1e-04  1530    0.317  0.000
       7     5     2     1    1e-06  2415    0.317  0.000
       3     3     3     1    1e-06   714    0.318  0.000
       3     5     2     1    1e-06  1127    0.318  0.000
       5     5     3     1    1e-06  2167    0.318  0.000
       7     3     2     1    1e-04  1335    0.319  0.000
       5     5     2     1    1e-06  1771    0.320  0.000
       5     3     3     1    1e-04  1122    0.321  0.000

### Best Ridge: ridge_nh7_me5_mchi1_a1e-06 (2955 features)
  Evaluating on training set...
  Evaluating on validation set...
  Train: E=5.3650e-03 MM=5.3639e-03 dphi=0.332
  Val:   E=7.2627e-03 MM=7.2624e-03 dphi=0.375 <0.1:15% <0.5:62% mono_viol=0.000
  Val LIGO MM (Mtot=20): med=1.8737e-03 max=8.1002e-02 <0.01:82%
  Val LIGO MM (Mtot=65): med=2.2700e-03 max=2.3238e-01 <0.01:76%
  Val LIGO MM (Mtot=110): med=2.0414e-03 max=2.3476e-01 <0.01:76%
  Val LIGO MM (Mtot=155): med=1.0775e-03 max=9.0449e-02 <0.01:83%
  Val LIGO MM (Mtot=200): med=5.3951e-04 max=5.7924e-02 <0.01:87%
  Generating diagnostic plots...

  [2026-04-05 20:51] Model ridge_nh7_me5_mchi1_a1e-06 trained and evaluated
=== CHECKLIST: ridge_nh7_me5_mchi1_a1e-06 ===
[PASS] model.pkl saved
[PASS] error arrays saved (10 npy files)
[PASS] histogram.pdf
[PASS] histogram_ligo_mm.pdf
[PASS] dephasing_vs_e0.pdf
[PASS] dephasing_vs_chieff.pdf
[PASS] mathcalE_vs_e0.pdf
[PASS] mathcalE_vs_chieff.pdf
[PASS] ligo_mm_vs_e0.pdf
[PASS] ligo_mm_vs_chieff.pdf
[PASS] best_modulation.pdf
[PASS] median_modulation.pdf
[PASS] worst_modulation.pdf
[PASS] summary.json
[PASS] progress_log.md updated
[PASS] CHANGELOG.md updated
=== ALL 16 CHECKS PASSED ===

### Polynomial scan (with spin features)
  degree=3 scanned
  degree=4 scanned
  degree=5 scanned

### Best Polynomial: deg=3, alpha=1e-01, 363 features
  Evaluating on training set...
  Evaluating on validation set...
  Train: E=2.5248e-02 MM=2.5262e-02 dphi=0.549
  Val:   E=3.6373e-02 MM=3.6349e-02 dphi=0.643 <0.1:2% <0.5:43% mono_viol=0.000
  Val LIGO MM (Mtot=20): med=5.2080e-03 max=2.1074e-01 <0.01:67%
  Val LIGO MM (Mtot=65): med=7.5516e-03 max=2.5329e-01 <0.01:57%
  Val LIGO MM (Mtot=110): med=8.0834e-03 max=3.1267e-01 <0.01:56%
  Val LIGO MM (Mtot=155): med=5.3053e-03 max=8.9041e-02 <0.01:64%
  Val LIGO MM (Mtot=200): med=2.8971e-03 max=5.0182e-02 <0.01:80%
  Generating diagnostic plots...

  [2026-04-05 21:00] Model polynomial_deg3 trained and evaluated
=== CHECKLIST: polynomial_deg3 ===
[PASS] model.pkl saved
[PASS] error arrays saved (10 npy files)
[PASS] histogram.pdf
[PASS] histogram_ligo_mm.pdf
[PASS] dephasing_vs_e0.pdf
[PASS] dephasing_vs_chieff.pdf
[PASS] mathcalE_vs_e0.pdf
[PASS] mathcalE_vs_chieff.pdf
[PASS] ligo_mm_vs_e0.pdf
[PASS] ligo_mm_vs_chieff.pdf
[PASS] best_modulation.pdf
[PASS] median_modulation.pdf
[PASS] worst_modulation.pdf
[PASS] summary.json
[PASS] progress_log.md updated
[PASS] CHANGELOG.md updated
=== ALL 16 CHECKS PASSED ===

### Random Forest
  Fitting RF...
  Evaluating on training set...
  Evaluating on validation set...
  Train: E=3.0341e-03 MM=3.0326e-03 dphi=0.264
  Val:   E=4.0114e-02 MM=4.0098e-02 dphi=0.662 <0.1:2% <0.5:33% mono_viol=0.000
  Val LIGO MM (Mtot=20): med=1.1966e-02 max=4.1146e-01 <0.01:43%
  Val LIGO MM (Mtot=65): med=4.8641e-03 max=2.6344e-01 <0.01:67%
  Val LIGO MM (Mtot=110): med=3.5304e-03 max=2.5145e-01 <0.01:71%
  Val LIGO MM (Mtot=155): med=2.0148e-03 max=8.3494e-02 <0.01:80%
  Val LIGO MM (Mtot=200): med=1.3479e-03 max=5.6635e-02 <0.01:85%
  Generating diagnostic plots...

  [2026-04-05 21:04] Model random_forest trained and evaluated
=== CHECKLIST: random_forest ===
[PASS] model.pkl saved
[PASS] error arrays saved (10 npy files)
[PASS] histogram.pdf
[PASS] histogram_ligo_mm.pdf
[PASS] dephasing_vs_e0.pdf
[PASS] dephasing_vs_chieff.pdf
[PASS] mathcalE_vs_e0.pdf
[PASS] mathcalE_vs_chieff.pdf
[PASS] ligo_mm_vs_e0.pdf
[PASS] ligo_mm_vs_chieff.pdf
[PASS] best_modulation.pdf
[PASS] median_modulation.pdf
[PASS] worst_modulation.pdf
[PASS] summary.json
[PASS] progress_log.md updated
[PASS] CHANGELOG.md updated
=== ALL 16 CHECKS PASSED ===

### Hybrid (ansatz + Ridge residual)
  Evaluating on training set...
  Evaluating on validation set...
  Train: E=5.3650e-03 MM=5.3639e-03 dphi=0.332
  Val:   E=7.2627e-03 MM=7.2624e-03 dphi=0.375 <0.1:15% <0.5:62% mono_viol=0.000
  Val LIGO MM (Mtot=20): med=1.8737e-03 max=8.1002e-02 <0.01:82%
  Val LIGO MM (Mtot=65): med=2.2700e-03 max=2.3238e-01 <0.01:76%
  Val LIGO MM (Mtot=110): med=2.0414e-03 max=2.3476e-01 <0.01:76%
  Val LIGO MM (Mtot=155): med=1.0775e-03 max=9.0449e-02 <0.01:83%
  Val LIGO MM (Mtot=200): med=5.3951e-04 max=5.7924e-02 <0.01:87%
  Generating diagnostic plots...

  [2026-04-05 21:13] Model hybrid_ansatz trained and evaluated
=== CHECKLIST: hybrid_ansatz ===
[PASS] model.pkl saved
[PASS] error arrays saved (10 npy files)
[PASS] histogram.pdf
[PASS] histogram_ligo_mm.pdf
[PASS] dephasing_vs_e0.pdf
[PASS] dephasing_vs_chieff.pdf
[PASS] mathcalE_vs_e0.pdf
[PASS] mathcalE_vs_chieff.pdf
[PASS] ligo_mm_vs_e0.pdf
[PASS] ligo_mm_vs_chieff.pdf
[PASS] best_modulation.pdf
[PASS] median_modulation.pdf
[PASS] worst_modulation.pdf
[PASS] summary.json
[PASS] progress_log.md updated
[PASS] CHANGELOG.md updated
=== ALL 16 CHECKS PASSED ===

### Best model (ridge_nh7_me5_mchi1_a1e-06) + phase correction
  Evaluating on training set...
  Evaluating on validation set...
  Train: E=3.5728e-04 MM=3.5691e-04 dphi=0.142
  Val:   E=4.1970e-04 MM=4.1964e-04 dphi=0.144 <0.1:37% <0.5:88% mono_viol=0.000
  Val LIGO MM (Mtot=20): med=5.0781e-04 max=4.7918e-02 <0.01:94%
  Val LIGO MM (Mtot=65): med=1.1426e-03 max=8.5110e-02 <0.01:83%
  Val LIGO MM (Mtot=110): med=1.2879e-03 max=1.2613e-01 <0.01:83%
  Val LIGO MM (Mtot=155): med=5.7695e-04 max=7.1165e-02 <0.01:91%
  Val LIGO MM (Mtot=200): med=2.1671e-04 max=4.4960e-02 <0.01:98%
  Generating diagnostic plots...

  [2026-04-05 21:22] Model ridge_nh7_me5_mchi1_a1e-06+phase_corr trained and evaluated
=== CHECKLIST: ridge_nh7_me5_mchi1_a1e-06+phase_corr ===
[PASS] model.pkl saved
[PASS] error arrays saved (10 npy files)
[PASS] histogram.pdf
[PASS] histogram_ligo_mm.pdf
[PASS] dephasing_vs_e0.pdf
[PASS] dephasing_vs_chieff.pdf
[PASS] mathcalE_vs_e0.pdf
[PASS] mathcalE_vs_chieff.pdf
[PASS] ligo_mm_vs_e0.pdf
[PASS] ligo_mm_vs_chieff.pdf
[PASS] best_modulation.pdf
[PASS] median_modulation.pdf
[PASS] worst_modulation.pdf
[PASS] summary.json
[PASS] progress_log.md updated
[PASS] CHANGELOG.md updated
=== ALL 16 CHECKS PASSED ===

### Comparison
Name                                          Val E     Val MM   Val dphi   <0.1   <0.5
--------------------------------------------------------------------------------
ansatz_only                              1.0457e+00 1.0096e+00     24.185    2%    5%
ridge_nh7_me5_mchi1_a1e-06               7.2627e-03 7.2624e-03      0.375   15%   62%
polynomial_deg3                          3.6373e-02 3.6349e-02      0.643    2%   43%
random_forest                            4.0114e-02 4.0098e-02      0.662    2%   33%
hybrid_ansatz                            7.2627e-03 7.2624e-03      0.375   15%   62%
ridge_nh7_me5_mchi1_a1e-06+phase_corr    4.1970e-04 4.1964e-04      0.144   37%   88%

Comparison plots saved to results/comparison/

Best: ridge_nh7_me5_mchi1_a1e-06+phase_corr (dphi=0.144 rad, E=4.1970e-04)

Done.

Total time: 2h 23m 54s