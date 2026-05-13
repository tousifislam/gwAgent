# gwAgent

**Discovery of Interpretable Surrogates via Agentic AI: Application to Gravitational Waves**

Tousif Islam, Digvijay Wadekar, Tejaswi Venumadhav, Matias Zaldarriaga, Ajit Kumar Mehta, Javier Roulet, Barak Zackay

**Paper:** [arXiv:2605.11280](https://arxiv.org/abs/2605.11280)

## Abstract

Fast surrogate models for expensive simulations are now essential across the sciences, yet they typically operate as black boxes. We present `GWAgent`, a large language model (LLM)-based workflow that constructs interpretable analytic surrogates directly from simulation data. Surrogate modeling is well suited to agentic workflows because candidate models can be quantitatively validated against ground-truth simulations at each iteration. As a demonstration, we build a surrogate for gravitational waveforms from eccentric binary black hole mergers. We show that providing the agent with a physics-informed domain ansatz substantially improves output model accuracy. The resulting analytic surrogate attains a median Advanced LIGO mismatch of 6.9x10^{-4} together with an ~8.4x speedup in waveform evaluation, surpassing both symbolic regression and conventional machine learning baselines. Beyond producing an accurate model, the workflow identifies compact physical structure from the learned representation. As an astrophysical application, we use `GWAgent` to analyze the eccentricity of GW200129 and infer e_{20Hz} = 0.099^{+0.063}_{-0.044}. These results show that validation-constrained agentic workflows can produce accurate, fast, and interpretable surrogates for scientific simulations and inference.

## Repository Structure

- **`dynamics_implementation/`** — EOB dynamics rewrite: Hamiltonian, flux, evolution equations, integrator, and eccentricity corrections
- **`modulation_learning/`** — Spin modulation learning pipeline: ridge regression fits, interpretability analysis, timing optimization, and model comparison
- **`global_workflow.md`** — High-level project workflow

## Citation

If you use this work, please cite:

```bibtex
@article{Islam:2026gwagent,
    author = {Islam, Tousif and Wadekar, Digvijay and Venumadhav, Tejaswi and Zaldarriaga, Matias and Mehta, Ajit Kumar and Roulet, Javier and Zackay, Barak},
    title = {Discovery of Interpretable Surrogates via Agentic AI: Application to Gravitational Waves},
    year = {2026},
    eprint = {2605.11280},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc}
}
```
