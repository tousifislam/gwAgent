# Eccentric Modulation Functions & the `gwNRXHME` Framework

> Based on: *"Universal phenomenological relations between spherical harmonic modes in non-precessing eccentric binary black hole merger waveforms"* — Islam & Venumadhav

---

## 1. Background: Spherical Harmonic Modes of BBH Waveforms

The gravitational waveform from a binary black hole (BBH) merger is decomposed into spin-weighted spherical harmonic modes:

$$h(t, \theta, \phi; \boldsymbol{\lambda}) = \sum_{\ell=2}^{\infty} \sum_{m=-\ell}^{\ell} h_{\ell m}(t; \boldsymbol{\lambda}) \; {}_{-2}Y_{\ell m}(\theta, \phi)$$

Each mode $h_{\ell m}$ is decomposed into an amplitude and phase:

$$h_{\ell m}(t; \boldsymbol{\lambda}) = A_{\ell m}(t) \; e^{i\phi_{\ell m}(t)}$$

The instantaneous frequency of each mode is:

$$\omega_{\ell m}(t; \boldsymbol{\lambda}) = \frac{d\phi_{\ell m}(t)}{dt}$$

For **non-precessing binaries**, the parameter space is:

$$\boldsymbol{\lambda} := \{q, \chi_1, \chi_2, e_{\rm ref}, l_{\rm ref}\}$$

where $q = m_1/m_2$ is the mass ratio, $\chi_1, \chi_2$ are the dimensionless spin magnitudes, and $e_{\rm ref}$, $l_{\rm ref}$ are the eccentricity and mean anomaly at a reference time/frequency.

The negative-$m$ modes follow by symmetry: $h_{\ell,-m} = (-1)^\ell h^*_{\ell m}$.

---

## 2. What Are Eccentric Modulation Functions?

The central idea is to **quantify the effect of eccentricity** on a waveform mode by comparing the eccentric waveform $h_{\ell m}(t; \boldsymbol{\lambda})$ to its quasi-circular counterpart $h_{\ell m}(t; \boldsymbol{\lambda}^0)$, where:

$$\boldsymbol{\lambda}^0 := \{q, \chi_1, \chi_2, e_{\rm ref}=0, l_{\rm ref}=0\}$$

This comparison yields two distinct modulation functions.

---

### 2.1 Eccentric Frequency Modulation

$$\boxed{\xi_{\ell m}^{\omega}(t; \boldsymbol{\lambda}) = b_{\ell m}^{\omega} \frac{\omega_{\ell m}(t; \boldsymbol{\lambda}) - \omega_{\ell m}(t; \boldsymbol{\lambda}^0)}{\omega_{\ell m}(t; \boldsymbol{\lambda}^0)}}$$

- Measures the **fractional deviation in instantaneous frequency** due to eccentricity.
- The prefactor $b_{\ell m}^{\omega} = 1$ — **no mode-dependent scaling**.
- Eccentricity induces oscillatory modulations around the quasi-circular frequency evolution.

---

### 2.2 Eccentric Amplitude Modulation

$$\boxed{\xi_{\ell m}^{A}(t; \boldsymbol{\lambda}) = b_{\ell m}^{A} \; \frac{2}{\ell} \; \frac{A_{\ell m}(t; \boldsymbol{\lambda}) - A_{\ell m}(t; \boldsymbol{\lambda}^0)}{A_{\ell m}(t; \boldsymbol{\lambda}^0)}}$$

- Measures the **fractional deviation in amplitude** due to eccentricity.
- The prefactor $b_{\ell m}^{A} = 1$, but the factor $\frac{2}{\ell}$ **does depend on the mode** — it normalises out the expected mode-dependent scaling.
- This normalisation is what makes the modulations comparable across different $(\ell, m)$ modes.

---

## 3. The Three Phenomenological Relations

The central empirical observations — validated across 83 NR simulations from the SXS, RIT, and MAYA catalogs — are:

### Relation I: Universality of Amplitude Modulations Across Modes

$$\xi_{22}^{A}(t) \approx \xi_{33}^{A}(t) \approx \xi_{44}^{A}(t) \approx \cdots$$

The amplitude modulations extracted from **different spherical harmonic modes are mutually consistent**. Despite $\ell$ varying, the $2/\ell$ normalisation in the definition collapses all modes onto a single universal curve.

### Relation II: Universality of Frequency Modulations Across Modes

$$\xi_{22}^{\omega}(t) \approx \xi_{33}^{\omega}(t) \approx \xi_{44}^{\omega}(t) \approx \cdots$$

Similarly, the frequency modulations obtained from different modes are all consistent with each other, with no additional normalisation needed (since $b_{\ell m}^\omega = 1$ already).

### Relation III: Amplitude–Frequency Modulation Relation

$$\boxed{\xi_{\ell m}^{A}(t; \boldsymbol{\lambda}) = B \; \xi_{\ell m}^{\omega}(t; \boldsymbol{\lambda})}$$

where the **universal scaling factor** $B = 0.9$.

This is the most powerful relation: the amplitude modulation and the frequency modulation are **not independent** — they are proportional. A single time series $\xi(t)$ encodes both.

> **Why is this remarkable?** Eccentricity introduces a rich oscillatory structure into both the amplitude and frequency of every mode. These three relations together say that *all* of this structure — across all modes and both observables — is governed by a **single universal modulation function**.

---

## 4. Degree of Departure: Quantifying How Well Relations Hold

To measure how strictly a given NR simulation adheres to these relations, the authors use the relative $L_2$-norm between two time series $s_1(t)$ and $s_2(t)$:

$$\mathcal{E}(s_1, s_2) = \frac{1}{2} \frac{\int_{t_{\rm min}}^{t_{\rm max}} |s_1(t) - s_2(t)|^2 \, dt}{\int_{t_{\rm min}}^{t_{\rm max}} |s_1(t)|^2 \, dt}$$

Five individual error measures are defined:

| Error | Definition | Tests |
|---|---|---|
| $\mathcal{E}_1$ | $\mathcal{E}(\xi_{22}^A, \xi_{33}^A)$ | Amplitude universality: (2,2) vs (3,3) |
| $\mathcal{E}_2$ | $\mathcal{E}(\xi_{22}^A, \xi_{44}^A)$ | Amplitude universality: (2,2) vs (4,4) |
| $\mathcal{E}_3$ | $\mathcal{E}(\xi_{22}^\omega, \xi_{33}^\omega)$ | Frequency universality: (2,2) vs (3,3) |
| $\mathcal{E}_4$ | $\mathcal{E}(\xi_{22}^\omega, \xi_{44}^\omega)$ | Frequency universality: (2,2) vs (4,4) |
| $\mathcal{E}_5$ | $\mathcal{E}(\xi_{22}^A, B\xi_{22}^\omega)$ | Amplitude–frequency relation |

The **overall degree of departure** is:

$$\mathcal{E}_{\xi} = \frac{\mathcal{E}_1 + \mathcal{E}_2 + \mathcal{E}_3 + \mathcal{E}_4 + \mathcal{E}_5}{5}$$

> **Note:** For equal-mass binaries, odd-$m$ modes vanish by symmetry, so $\mathcal{E}_3$ and $\mathcal{E}_5$ are excluded. The overall error is then $\mathcal{E}_\xi = (\mathcal{E}_1 + \mathcal{E}_2 + \mathcal{E}_4)/3$.

### Results by Catalog

| Catalog | $\mathcal{E}_\xi$ range | Notes |
|---|---|---|
| **SXS** | $10^{-3}$ – $10^{-2}$ | Best adherence; highest NR accuracy |
| **RIT** | $0.004$ – $0.2$ | Larger scatter; spinning cases worse |
| **MAYA** | $10^{-2}$ – $10^{-1}$ | Moderate; higher numerical noise |

The larger departures in RIT and MAYA are conjectured to be primarily due to **numerical errors** in those simulations rather than a true physical violation of the relations.

---

## 5. The `gwNRXHME` Framework

### 5.1 Motivation

Relations I–III together imply that one can reconstruct the full multi-modal eccentric waveform using only:
1. The **dominant $(2,2)$ mode** of an eccentric (NR) waveform — to extract the universal modulation.
2. The **full multi-modal quasi-circular waveform** — to provide the baseline mode structure.

This is the foundation of `gwNRXHME` (**Gravitational-wave Numerical Relativity eXtended Higher-Mode Eccentric**), an extension of the earlier `gwNRHME` framework (which covered non-spinning binaries) to all non-precessing binaries.

### 5.2 The Universal Modulation Parameter

Define a single common modulation time series:

$$\xi(t) := \xi_{22}^{A}(t; \boldsymbol{\lambda})$$

This single scalar time series, extracted from the $(2,2)$ mode of the eccentric simulation, drives all higher-mode reconstruction.

### 5.3 Reconstructing Mode Amplitudes

$$\boxed{A_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda}) = A_{\ell m}(t; \boldsymbol{\lambda}^0) \left[\frac{\ell}{2}\,\xi(t) + 1\right]}$$

- The factor $\ell/2$ is the inverse of the $2/\ell$ normalisation in the amplitude modulation definition.
- For each mode, the circular amplitude is **modulated multiplicatively** by the eccentric correction.
- When $\xi(t) = 0$ (circular limit), $A_{\ell m}^{\tt gwNRXHME} \to A_{\ell m}(t; \boldsymbol{\lambda}^0)$ exactly.

### 5.4 Reconstructing Mode Frequencies

$$\boxed{\omega_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda}) = \omega_{\ell m}(t; \boldsymbol{\lambda}^0) \left[\frac{\xi(t)}{B} + 1\right]}$$

- Uses the amplitude–frequency relation (Relation III) with $B = 0.9$.
- The circular frequency of each mode is modulated by the same $\xi(t)$, rescaled by $1/B$.

### 5.5 Reconstructing Mode Phases

Integrating the frequency:

$$\boxed{\phi_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda}) = \phi_0 + \int \omega_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda})\, dt}$$

where $\phi_0 = \phi_{\ell m}(t; \boldsymbol{\lambda}^0)$ is the integration constant taken from the quasi-circular phase.

### 5.6 Full Complex Mode Reconstruction

$$\boxed{h_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda}) = A_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda})\; e^{i\,\phi_{\ell m}^{\tt gwNRXHME}(t; \boldsymbol{\lambda})}}$$

---

## 6. Summary of the `gwNRXHME` Recipe

Given:
- An eccentric NR (or model) waveform → extract the $(2,2)$ mode → compute $\xi(t) = \xi_{22}^A(t)$
- A quasi-circular multi-modal waveform → provides $A_{\ell m}(t; \boldsymbol{\lambda}^0)$, $\omega_{\ell m}(t; \boldsymbol{\lambda}^0)$, $\phi_{\ell m}(t; \boldsymbol{\lambda}^0)$

Reconstruct every higher mode $(\ell, m)$ via:

```
Step 1:  A_lm_ecc  = A_lm_circ  × [ (l/2) × ξ(t)  + 1 ]
Step 2:  ω_lm_ecc  = ω_lm_circ  × [ ξ(t)/B         + 1 ]
Step 3:  φ_lm_ecc  = φ_0 + ∫ ω_lm_ecc dt
Step 4:  h_lm_ecc  = A_lm_ecc × exp(i × φ_lm_ecc)
```

with $B = 0.9$.

---

## 7. Scope, Validity, and Implications

### Scope
- Validated for **non-precessing** (aligned-spin and non-spinning) eccentric BBH mergers.
- Mass ratios $q = 1$–$4$, spins $\chi_{1,2} \in [-0.8, 0.8]$.
- Tested on 83 NR simulations across SXS, RIT, MAYA.

### Key Implications

**1. Simplification of Eccentric Waveform Modeling**
Eccentric multi-modal waveforms — previously requiring independent modeling of each $(\ell, m)$ mode — can be constructed from just two ingredients: a quadrupolar eccentric waveform and a quasi-circular multi-modal model.

**2. New NR Catalog Benchmark**
The degree of departure $\mathcal{E}_\xi$ provides a quantitative metric to compare eccentric NR catalogs against each other, independently of matching to observations.

**3. Noise Filtering of Higher Modes**
Higher-order modes in eccentric NR data are often contaminated by numerical noise (especially $(4,4)$ and above). `gwNRXHME` provides a **clean, noise-filtered prediction** for these modes by exploiting the universal relation — essentially denoising NR data.

**4. Pathway to Full Eccentric Models**
`gwNRXHME` can directly combine existing multi-modal quasi-circular models (`NRHybSur3dq8`, `IMRPhenomTHM`, `SEOBNRv5HM`) with any quadrupolar eccentric waveform model to produce accurate **multi-modal non-precessing eccentric waveform models**, dramatically reducing modeling complexity.

### Current Limitations
- Not yet validated for **precessing eccentric** binaries — very few such NR simulations exist.
- The universal factor $B = 0.9$ is phenomenological; its PN or NR origin is not yet analytically derived.
- Larger $\mathcal{E}_\xi$ in RIT and MAYA may partly reflect genuine physical deviations, not just numerical noise — further investigation is needed.

---

## 8. Quick Reference: Key Equations

| Quantity | Expression |
|---|---|
| Frequency modulation | $\xi_{\ell m}^\omega = \dfrac{\omega_{\ell m}^{\rm ecc} - \omega_{\ell m}^{\rm circ}}{\omega_{\ell m}^{\rm circ}}$ |
| Amplitude modulation | $\xi_{\ell m}^A = \dfrac{2}{\ell}\dfrac{A_{\ell m}^{\rm ecc} - A_{\ell m}^{\rm circ}}{A_{\ell m}^{\rm circ}}$ |
| Amp–freq relation | $\xi_{\ell m}^A = B \, \xi_{\ell m}^\omega$, $\quad B = 0.9$ |
| Universal modulation | $\xi(t) = \xi_{22}^A(t)$ |
| Reconstructed amplitude | $A_{\ell m}^{\rm ecc} = A_{\ell m}^{\rm circ}\left[\frac{\ell}{2}\xi + 1\right]$ |
| Reconstructed frequency | $\omega_{\ell m}^{\rm ecc} = \omega_{\ell m}^{\rm circ}\left[\frac{\xi}{B} + 1\right]$ |

---

*References: Islam & Venumadhav (2024); Islam et al. (2024), arXiv:2409.17636; Memmesheimer, Gopakumar & Schaefer, Phys. Rev. D 70, 104011 (2004)*