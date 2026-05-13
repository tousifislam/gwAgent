# Interpreting the Fitted Analytic Modulation Formula

*Structural analysis, PN comparison, and compactification strategies for the Ridge residual model*

---

## 1. Total Modulation Structure

The full eccentric modulation is decomposed as:

$$\xi_{\rm amp}(t) = \underbrace{\bigl|\,h_{22}^{\rm ecc}(x, e, \zeta, \nu)\bigr| - 1}_{\text{PN ansatz (known)}} \;+\; \underbrace{\delta\xi_{\rm amp}(e, x, \zeta, \nu, \chi_S, \chi_A)}_{\text{Ridge residual (fitted)}}$$

and similarly for $\xi_\omega$, with $\xi_{\omega,\text{ansatz}} = \xi_{\rm amp,\text{ansatz}} / 0.9$ (Relation III).

The ansatz comes from the 1PN eccentric correction factor `hFactEccCorr[2,2]` in `EOB_modes.dat.m` (Gamboa, Khalil & Buonanno). The Ridge model learns the **residual** $\delta\xi$ that captures higher PN orders, spin effects, resummation differences, and ODE systematics.

---

## 2. Structure of the Fitted Residual

### 2.1 Basis specification

The best model (`ridge_nh7_me5_mchi1_a1e-06`) uses:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `max_e` | 5 | Eccentricity powers $e^1$ through $e^5$ |
| `max_x` | 3 | Frequency powers $x^0$ through $x^3$ (0PN to 3PN) |
| `max_nu` | 2 | Mass ratio powers $\nu^0$ through $\nu^2$ |
| `max_chi` | 1 | Spin powers $\chi_S^{0,1}$, $\chi_A^{0,1}$ |
| `n_harm` | 7 | Fourier harmonics $k = 1, \ldots, 7$ in $\zeta$ |
| Constraint | $a + b + c + d_S + d_A \leq 8$ | Total degree bound |

This produces **197 power-law groups**, each dressed with 15 features (1 base + 7 cos + 7 sin), giving **2955 total coefficients** for each of $\delta\xi_{\rm amp}$ and $\delta\xi_\omega$.

### 2.2 Mathematical form

$$\delta\xi = \sum_{g=1}^{197} e^{a_g}\, x^{b_g}\, \nu^{c_g}\, \chi_S^{d_g}\, \chi_A^{f_g} \left[\, c_0^{(g)} + \sum_{k=1}^{7} \bigl(\alpha_k^{(g)} \cos k\zeta + \beta_k^{(g)} \sin k\zeta\bigr) \right]$$

with constraint $a_g \geq 1$ ensuring $\delta\xi \to 0$ as $e \to 0$.

---

## 3. Dominant Structure of the Residual

### 3.1 Harmonic hierarchy

The k=1 harmonic (fundamental orbital frequency) overwhelmingly dominates:

| Harmonic $k$ | Total $\|\mathbf{c}\|$ (amp) | Fraction |
|:---:|---:|---:|
| 0 (base) | 350 | 8.6% |
| **1** | **2579** | **63.3%** |
| 2 | 342 | 8.4% |
| 3 | 124 | 3.0% |
| 4 | 200 | 4.9% |
| 5 | 120 | 2.9% |
| 6 | 153 | 3.8% |
| 7 | 190 | 4.7% |

**Interpretation**: The PN ansatz at 0PN and 1PN already captures the $\cos\zeta$ structure accurately. The residual's dominant k=1 content is predominantly $\sin\zeta$ --- a **phase shift** of the fundamental harmonic that the real-valued ansatz $|h_{22}^{\rm ecc}| - 1$ cannot produce. This $\sin\zeta$ component arises from:

- The complex phase of $h_{22}^{\rm ecc}$ (the ansatz only uses the magnitude)
- Higher-PN corrections that mix $\cos\zeta$ and $\sin\zeta$ via tail effects ($\sim \pi x^{3/2}$)
- Spin-orbit coupling (1.5PN), which introduces odd-parity $\sin\zeta$ terms

### 3.2 Eccentricity power hierarchy

| Power $e^a$ | Total $\|\mathbf{c}\|$ | Fraction |
|:---:|---:|---:|
| $e^1$ | 2001 | 39.1% |
| $e^2$ | 1467 | 28.7% |
| $e^3$ | 788 | 15.4% |
| $e^4$ | 507 | 9.9% |
| $e^5$ | 346 | 6.8% |

The $e^1$ and $e^2$ terms dominate, consistent with PN theory where leading eccentricity corrections enter at $O(e)$ and quadratic corrections appear with smaller coefficients. The slow decay suggests the series has not fully converged at $e \sim 0.5$, motivating resummation (see Section 6).

### 3.3 PN order hierarchy

| Power $x^b$ | PN order | Total $\|\mathbf{c}\|$ | Fraction |
|:---:|:---:|---:|---:|
| $x^0$ | 0PN | 1107 | 21.6% |
| $x^1$ | 1PN | 2349 | 45.9% |
| $x^2$ | 2PN | 1149 | 22.5% |
| $x^3$ | 3PN | 503 | 9.8% |

The 1PN residual dominates, which is expected: the ansatz includes the 0PN leading term and 1PN correction explicitly, but with a truncated eccentricity expansion. The residual at 1PN therefore captures (i) higher eccentricity orders ($e^3$--$e^5$) at 1PN, and (ii) the $\sin\zeta$ phase content missing from $|h_{22}^{\rm ecc}| - 1$.

### 3.4 Spin sector analysis

| Sector | Weight fraction |
|--------|---:|
| Non-spin ($\chi_S^0 \chi_A^0$) | 36.9% |
| $\chi_S$-linear | 27.2% |
| $\chi_A$-linear | 27.2% |
| $\chi_S \chi_A$ (bilinear) | 8.8% |

**Key finding**: Spin corrections contribute **63% of the residual weight**. This is the most important physics the residual captures, since the PN ansatz (`hFactEccCorr[2,2]`) is non-spinning.

The near-equal weight of $\chi_S$ and $\chi_A$ sectors is physically meaningful:
- $\chi_S = (\chi_1 + \chi_2)/2$ enters at 1.5PN (spin-orbit) and 2PN (spin-spin)
- $\chi_A = (\chi_1 - \chi_2)/2$ enters at 1.5PN multiplied by $\Delta = (m_1 - m_2)/M$

In PN theory, spin-orbit terms have the structure $x^{3/2} \cdot (\chi_S + \chi_A \Delta/\nu)$. The Ridge model parameterizes this as separate $\chi_S$ and $\chi_A$ terms (absorbing the $\Delta/\nu$ factor into the coefficients), which is equivalent but more flexible.

---

## 4. Comparison with the PN Ansatz (`hFactEccCorr[2,2]`)

### 4.1 Structure of the ansatz

The ansatz from `EOB_modes.dat.m` (implemented in `fit.py:59-76`) has the form:

$$h_{22}^{\rm ecc} = \underbrace{\frac{4 + 2e^2 e^{2i\zeta} + e\,e^{-i\zeta} + 5e\,e^{i\zeta}}{4(1 - e^2)}}_{\text{0PN (Newtonian)}} \;+\; \underbrace{\frac{x\,e}{(1 - e^2)^2}\, \mathcal{C}(\zeta, e, \nu)}_{\text{1PN correction}}$$

where $\mathcal{C}$ contains Fourier modes $e^{ik\zeta}$ for $k \in \{-3, -2, -1, 0, 1, 2, 3\}$ with $\nu$-dependent coefficients.

#### 0PN Fourier content

At $e = 0.3$, $\nu = 0.25$, $x = 0$:

| Harmonic | $\xi_{\rm amp,ansatz}$ | Physical origin |
|----------|---:|---|
| $c_0$ (mean) | $+0.124$ | Secular eccentricity shift |
| $\cos\zeta$ | $+0.497$ | **Dominant** --- Keplerian ellipse |
| $\cos 2\zeta$ | $+0.025$ | Second harmonic (pericenter asymmetry) |
| $\sin k\zeta$ | $\approx 0$ | Absent by symmetry of $|h_{22}|$ |

The ansatz's $\xi_{\rm amp} = |h_{22}^{\rm ecc}| - 1$ contains **only $\cos k\zeta$ harmonics** --- the absolute value operation kills all $\sin k\zeta$ dependence. This is a structural gap that the residual must fill.

### 4.2 What the residual adds beyond the ansatz

#### (a) $\sin\zeta$ content (absent from ansatz)

The top 5 Ridge coefficients for $\delta\xi_{\rm amp}$ are **all $\sin\zeta$ terms**:

| Rank | Term | $c_{\rm amp}$ |
|---:|------|---:|
| 1 | $e^2\, x\, \nu\, \sin\zeta$ | $+100.7$ |
| 2 | $e\, x^2\, \chi_S\, \sin\zeta$ | $+90.6$ |
| 3 | $e\, x\, \nu\, \chi_S\, \sin\zeta$ | $+79.5$ |
| 4 | $e\, x^2\, \chi_A\, \sin\zeta$ | $+70.9$ |
| 5 | $e\, x\, \nu\, \chi_A\, \sin\zeta$ | $+69.9$ |

These $\sin\zeta$ terms arise from the **imaginary part of $h_{22}^{\rm ecc}$** --- specifically, the argument (phase) of the eccentric correction factor. In full PN theory, the (2,2) mode has the form:

$$h_{22} \propto e^{-2i\phi} \cdot h_{22}^{\rm ecc}(x, e, \zeta, \nu, \chi)$$

where $h_{22}^{\rm ecc}$ is complex. The ansatz uses only $|h_{22}^{\rm ecc}|$, discarding $\arg(h_{22}^{\rm ecc})$. The leading imaginary part at 0PN is:

$$\text{Im}(h_{22}^{\rm ecc}) \supset \frac{e\,(5 - 1)\sin\zeta}{4(1 - e^2)} = \frac{e\sin\zeta}{1 - e^2}$$

This is exactly the type of $e \cdot \sin\zeta$ structure that dominates the residual.

#### (b) Higher eccentricity orders at 1PN

The ansatz's 1PN term contains only $e^1$, $e^2$, $e^3$ in eccentricity. The residual's dominant 1PN terms include:

- $e^4 x\, \sin\zeta$ ($|c| = 32.8$)
- $e^5 x\, \sin\zeta$ ($|c| = 25.0$)

These represent higher-order eccentricity corrections at 1PN that are present in the full `EOB_modes.dat.m` expression (which goes to $O(e^{10})$ at 2.5PN) but truncated in the simplified ansatz.

#### (c) Spin-orbit corrections (1.5PN)

The leading spin-orbit effect in the (2,2) mode enters at $O(x^{3/2})$ in PN theory:

$$\delta h_{22}^{\rm SO} \sim x^{3/2} \cdot e \cdot \bigl(\chi_S\, f_S(\nu) + \chi_A\, \Delta\, f_A(\nu)\bigr) \cdot (\text{Fourier in } \zeta)$$

Since $x^{3/2}$ is not in our integer-power basis, the model approximates it via $x^1$ and $x^2$ terms:

$$x^{3/2} \approx c_1\, x + c_2\, x^2 \quad \text{(over the fitted range } x \in [0, 0.1])$$

This explains why the largest spin terms appear at both $x^1$ and $x^2$:
- $e\, x^2\, \chi_S\, \sin\zeta$: $c = +90.6$ (captures $x^{3/2}$ via $x^2$)
- $e\, x\, \nu\, \chi_S\, \sin\zeta$: $c = +79.5$ (captures $x^{3/2}$ via $x \cdot \nu$)

#### (d) 2PN and 3PN corrections

The full `EOB_modes.dat.m` contains 2PN ($x^2$) and 2.5PN ($x^{5/2}$) terms that are entirely absent from the implemented ansatz. The residual's $x^2$ and $x^3$ sectors absorb these:

- **2PN** ($x^2$): $e\, x^2\, \nu \cdot \cos\zeta$ and $\sin\zeta$ terms with $|c| \sim 20$--$60$
- **3PN** ($x^3$): $e\, x^3 \cdot \sin\zeta$ dominates with $|c| = 64$ --- largest single non-spin, non-$x^1$ term

The 3PN content is also partially absorbing tail effects (1.5PN, $\propto \pi x^{3/2}$) and their eccentricity-enhanced versions.

### 4.3 Amplitude--frequency coupling ratio ($c_\omega / c_{\rm amp}$)

For the 15 dominant terms, the ratio of frequency-to-amplitude coefficients is:

| Term | $c_\omega / c_{\rm amp}$ |
|------|---:|
| $e^2 x \nu \sin\zeta$ | 1.35 |
| $e x^2 \chi_S \sin\zeta$ | 1.54 |
| $e x \nu \chi_S \sin\zeta$ | 1.42 |
| $e x^2 \chi_A \sin\zeta$ | 1.56 |
| $e^2 x^2 \cos\zeta$ | 1.13 |
| $e x^3 \cos\zeta$ | 0.93 |
| $e x^3 \sin\zeta$ | 1.00 |

The gwNRXHME Relation III predicts $\xi_\omega = \xi_{\rm amp} / B$ with $B \approx 0.9$, i.e., $c_\omega/c_{\rm amp} \approx 1/0.9 = 1.11$. The measured ratios cluster around **1.1--1.6**, with spin-dependent terms showing larger ratios (~1.4--1.6) and the highest-PN non-spin terms ($x^3$) converging to $\sim 1.0$. This suggests:

- Relation III holds approximately for the non-spin, high-PN sector
- **Spin corrections violate Relation III**: the frequency modulation receives relatively larger spin corrections than the amplitude, with $B_{\rm spin} \approx 0.65$--$0.75$ rather than 0.9
- This is physically expected: spin-orbit coupling affects the orbital frequency more directly than the wave amplitude

---

## 5. Connecting to Known PN Expressions in `EOB_modes.dat.m`

### 5.1 Term-by-term identification

The full `hFactEccCorr[2,2]` in `EOB_modes.dat.m` (lines 13233--13812) contains five PN orders:

| PN order | Power of $x$ | Denominator | Content |
|----------|:---:|---|---|
| 0PN | $x^0$ | $(1 - e^2)^{-1}$ | Leading Keplerian modulation |
| 1PN | $x^1$ | $(1 - e^2)^{-2}$ | First relativistic correction, $\nu$-dependent |
| 1.5PN | $x^{3/2}$ | $(1 - e^2)^{-7/2}$ | Tail terms with $\pi$, $\log$ factors |
| 2PN | $x^2$ | $(1 - e^2)^{-3}$ | Second-order, $\nu^2$-dependent |
| 2.5PN | $x^{5/2}$ | $(1 - e^2)^{-7/2}$ | Higher tail, eccentricity up to $e^{10}$ |

The implemented ansatz only includes **0PN + 1PN** (lines 59--76 of `fit.py`). Therefore, the Ridge residual absorbs:

| What the residual captures | PN origin | Dominant Ridge terms |
|---|---|---|
| Complex phase of $h_{22}^{\rm ecc}$ | 0PN (imaginary part) | $e^a \sin\zeta$ at all $x$ powers |
| 1.5PN tail contributions | $x^{3/2} \cdot \pi$ | $e^a\, x\, \nu$ and $e^a\, x^2$ terms |
| 2PN corrections | $x^2 \cdot f(\nu, \nu^2)$ | $e\, x^2\, \nu$, $e^2\, x^2$ terms |
| 2.5PN corrections | $x^{5/2}$ | $e\, x^3$ terms (approximating $x^{5/2}$) |
| All spin-orbit (1.5PN) | $x^{3/2} \chi$ | $e^a\, x\, \chi_S$, $e^a\, x^2\, \chi_S$ |
| All spin-spin (2PN) | $x^2 \chi^2$ | $e^a\, x^2\, \chi_S \chi_A$ |
| Resummation effects | $(1 - e^2)^{-p}$ factors | Spread across all $e^a$ powers |

### 5.2 The $(1 - e^2)^{-p}$ resummation signature

The PN denominators $(1 - e^2)^{-1}$, $(1 - e^2)^{-2}$, etc., when Taylor-expanded in $e$:

$$\frac{1}{(1 - e^2)^p} = 1 + p\,e^2 + \frac{p(p+1)}{2}\,e^4 + \frac{p(p+1)(p+2)}{6}\,e^6 + \cdots$$

generate a specific pattern of coefficients at even powers of $e$. The Ridge model's polynomial basis must reconstruct these factors term by term, which is one reason the $e^4$ and $e^5$ coefficients are still substantial. This is a prime target for resummation (Section 6).

---

## 6. Strategies for Compactifying the 2955-Term Expression

There are two independent axes of compactification: (i) resum the Fourier series in each group, and (ii) collapse the power-law groups.

### 6.1 Fourier resummation via Hansen coefficients

The classic Hansen coefficient identity is:

$$\sum_{k=-\infty}^{\infty} X_k^{n,m}(e)\, e^{ik\zeta} = \frac{e^{im v}}{(r/a)^n}$$

where $v$ is the true anomaly and $r/a = (1 - e\cos u)$ with $u$ the eccentric anomaly. Truncated Fourier series in $\zeta$ with $e$-dependent coefficients therefore resum to closed forms involving:

- Powers of $(1 - e^2)^{-1/2}$
- Trigonometric functions of the true anomaly $v$ or eccentric anomaly $u$
- Bessel function envelopes: $\sqrt{\alpha_k^2 + \beta_k^2} \sim J_k(ke)$ or $(e/2)^k/k!$

**Diagnostic**: For each power-law group $g$, plot the Fourier amplitudes $A_k^{(g)} = \sqrt{(\alpha_k^{(g)})^2 + (\beta_k^{(g)})^2}$ vs $k$. From the coefficient analysis, the dominant group ($e^2 x \nu$) has:

$$A_1 = 115.8, \quad A_2 = 12.1, \quad A_3 = 1.9, \quad A_4 = 1.1, \quad \ldots$$

The ratio $A_2/A_1 \approx 0.10$ and $A_3/A_1 \approx 0.016$ show rapid decay --- consistent with a Bessel-like or geometric envelope. If confirmed across groups, each 15-term Fourier sum collapses to a closed form, reducing the expression from **2955 to ~197 terms**.

### 6.2 Pade in $w = e^{i\zeta}$

Convert each group's Fourier sum to a Laurent polynomial in $w = e^{i\zeta}$:

$$F_g(\zeta) = c_0 + \sum_{k=1}^{7}\bigl(\alpha_k \cos k\zeta + \beta_k \sin k\zeta\bigr) = c_0 + \sum_{k=1}^{7}\frac{1}{2}\bigl[(\alpha_k - i\beta_k)\,w^k + (\alpha_k + i\beta_k)\,w^{-k}\bigr]$$

then fit a Pade approximant $P_m(w)/Q_n(w)$. Since the k=1 harmonic dominates ($>60\%$), a [1,1] or [2,1] Pade should suffice for most groups, giving:

$$F_g(\zeta) \approx \frac{a + b\,e^{i\zeta}}{1 + c\,e^{i\zeta}} = \frac{A + B\cos\zeta + C\sin\zeta}{1 + D\cos\zeta + E\sin\zeta}$$

This reduces each group from 15 parameters to ~5, and the full expression from 2955 to ~985 parameters.

### 6.3 Collapsing power-law groups

#### Tensor product check

The 197 exponent tuples $(a, b, c, d_S, d_A)$ form a **truncated Cartesian product** constrained by $a + b + c + d_S + d_A \leq 8$. The full product $\{1..5\} \times \{0..3\} \times \{0..2\} \times \{0..1\}^2 = 240$ groups, of which 197 survive the constraint. This is close to factored, suggesting the expression approximately separates.

Test whether the coefficient tensor $C_{a,b,c,d_S,d_A,k}$ has **low Tucker rank**: perform a multilinear SVD and check how many singular values are needed in each mode.

#### $(1 - e^2)^{-p}$ factoring

Since PN denominators produce the pattern $e^0 : e^2 : e^4 = 1 : p : p(p+1)/2$, check whether the base coefficients $c_0^{(g)}$ for groups with the same $(b, c, d_S, d_A)$ but varying $a$ follow this ratio. If so, factor out $(1 - e^2)^{-p}$ from each $(b, c, d_S, d_A)$ sector:

$$\sum_{a=1}^{5} c_a^{(b,c,d_S,d_A)}\, e^a \;\longrightarrow\; \frac{e \cdot P(e^2)}{(1 - e^2)^{p(b)}}$$

where $P$ is a low-order polynomial. This is physically motivated: every PN order carries a $(1 - e^2)^{-p}$ enhancement.

#### Pade in physical variables

After resumming in $\zeta$, apply Pade approximation in each remaining variable:

1. **$\nu$ (mass ratio)**: Since $\nu \in (0, 1/4]$, and PN coefficients are always polynomials in $\nu$, a [1,0] or [1,1] Pade in $\nu$ should suffice (max power is $\nu^2$, already low).

2. **$x$ (frequency)**: Since $x \in [0, 0.1]$, the Taylor series converges well, but a [2,1] Pade in $x$ can capture the $(1 - 6x)^{-1}$ light-ring enhancement seen in EOB models.

3. **$e$ (eccentricity)**: The series in $e$ converges slowest. A Pade in $e^2$ can absorb the $(1 - e^2)^{-p}$ factors: $\sum c_n e^{2n} \to P_m(e^2)/Q_n(e^2)$.

### 6.4 Expected compactification path

| Step | Terms | Parameters | Strategy |
|------|------:|----------:|---------|
| Current | 2955 | 2955 | Raw polynomial $\times$ Fourier |
| After Fourier resummation | 197 groups | ~985 | Pade [2,1] in $w = e^{i\zeta}$ per group |
| After $(1-e^2)^{-p}$ factoring | ~50 groups | ~250 | Factor known PN denominators |
| After Pade in $e^2$ | ~25 terms | ~125 | Rational function in $e^2$ |
| After Pade in $x$ | ~10--15 terms | ~50--75 | Rational function in $x$ |

The final compact form would be:

$$\delta\xi_{\rm amp} = \sum_{i=1}^{N} \frac{e\, \mathcal{P}_i(\nu, \chi_S, \chi_A)}{(1-e^2)^{p_i}} \cdot \frac{R_i(x)}{S_i(x)} \cdot \frac{A_i + B_i\cos\zeta + C_i\sin\zeta}{1 + D_i\cos\zeta}$$

with $N \sim 10$--$15$ and $\mathcal{P}_i$ being low-order polynomials in the spin variables.

---

## 7. Physical Interpretation Summary

### What the PN ansatz captures

The 0PN + 1PN ansatz $|h_{22}^{\rm ecc}| - 1$ provides the **Keplerian backbone**: the dominant $e\cos\zeta/(1-e^2)$ modulation with 1PN relativistic corrections proportional to $x \cdot e \cdot f(\nu)$. This accounts for the fundamental oscillation of amplitude and frequency at the orbital period, driven by the varying separation in an elliptical orbit.

### What the Ridge residual adds

1. **$\sin\zeta$ phase content** (largest contribution, $\sim 40\%$ of residual weight): The imaginary part of $h_{22}^{\rm ecc}$ that is lost when taking $|h_{22}^{\rm ecc}| - 1$. Physically, this represents the **orbital phase modulation** --- the GW phase advances faster near pericenter and slower near apocenter, creating a $\sin\zeta$ asymmetry.

2. **Spin-orbit and spin-spin corrections** ($\sim 55\%$ of residual weight through $\chi_S, \chi_A$ terms): Enters at 1.5PN as $x^{3/2} \cdot \chi \cdot e \cdot \sin\zeta$ (approximated by $x \cdot \chi$ and $x^2 \cdot \chi$ in the integer-power basis). The spin-orbit force modifies both the orbital precession rate and the instantaneous radiation, with $\chi_S$ and $\chi_A$ contributing nearly equally.

3. **Higher PN orders** (2PN, 2.5PN, tail effects): The 2PN terms ($x^2$) bring $\nu^2$-dependent corrections and higher eccentricity harmonics. The 1.5PN tail contribution ($\propto \pi x^{3/2}$, containing logarithms and $\pi$) is absorbed into the $x^1$ and $x^2$ sectors. The 3PN-like $x^3$ terms capture 2.5PN corrections approximated via integer powers.

4. **Resummation of $(1-e^2)^{-p}$ factors**: The PN denominators enhanced by eccentricity are reconstructed term-by-term through the $e^3$, $e^4$, $e^5$ coefficients, which encode the Taylor expansion of factors like $(1 - e^2)^{-7/2}$ appearing at 1.5PN.

### Accuracy hierarchy

The model achieves validation $\mathcal{E} = 4.2 \times 10^{-4}$ (median) with phase correction, meaning the ansatz + residual together reproduce the SEOB waveform to sub-percent accuracy across the parameter space $q \in [1,10]$, $|\chi| \leq 0.5$, $e_0 \leq 0.5$.

---

## 8. Recommended Next Steps

1. **Extract and tabulate the Fourier amplitude envelopes** $A_k^{(g)}$ for each power-law group to identify Hansen/Bessel structure.

2. **Implement Pade resummation in $w = e^{i\zeta}$** for the top 30 groups (which carry ~80% of the total weight) and measure the accuracy loss.

3. **Factor out $(1 - e^2)^{-p}$** from each PN-order sector by fitting $p$ from the ratio of even-$e$ coefficients.

4. **Compare the non-spin $x^2$ and $x^3$ coefficients** directly against the 2PN and 2.5PN expressions in `EOB_modes.dat.m` to check quantitative agreement.

5. **Attempt a compact spin-augmented ansatz**: extend `h22_ecc_ansatz` to include the leading spin-orbit term at 1.5PN from the Mathematica supplementary files, which would absorb the largest spin residuals and potentially reduce the Ridge model to a pure non-spin correction.
