# PRISTINE Framework Overview

PRISTINE is a flexible and differentiable inference engine for phylogenetic and evolutionary modeling. It combines maximum likelihood estimation with robust uncertainty quantification and supports a wide range of biological and statistical analyses.

This overview outlines the types of questions and tasks you can perform with PRISTINE, along with links to the technical documentation for each component.

---

## Key Questions You Can Address

- **How have sequences evolved over a phylogenetic tree?**
  - Use the [Felsenstein Pruning Algorithm](felsenstein.md) to compute the likelihood of observed sequences given a substitution model and tree.

- **What substitution process best fits my data?**
  - Use the [GTR model](gtr_model.md) to estimate transition rates between nucleotides or amino acids.

- **When did evolutionary divergences occur?**
  - Fit branch times using [molecular clock models](molecular_clock.md), including both strict and relaxed clocks.

- **How do traits or latent states influence speciation and sampling rates?**
  - Fit state-dependent diversification processes using [birth-death-sampling models](bds_model.md), including trait-based linear models.

- **How confident am I in the inferred parameters?**
  - Estimate parameter uncertainty with [Laplace approximation](laplace.md) or robust [likelihood profiling](likelihood_profiler.md).

- **Can I visualize non-identifiability or parameter sloppiness?**
  - Use curvature diagnostics in [Laplace estimation](laplace.md) to assess identifiability.

- **How do I fit models robustly in complex loss landscapes?**
  - Use gradient-based [optimization with backtracking](optimization.md) to ensure convergence even when gradients are unstable.

---

## Modules and Their Roles

| Component | Functionality |
|----------|----------------|
| [Felsenstein Algorithm](felsenstein.md) | Efficient likelihood computation over a phylogenetic tree |
| [GTR Model](gtr_model.md) | Generalized time-reversible substitution process |
| [Molecular Clock Models](molecular_clock.md) | Models that link substitutions to branch time |
| [Birth-Death-Sampling Models](bds_model.md) | Diversification modeling over time and states |
| [Laplace Estimation](laplace.md) | Gaussian approximation of posterior uncertainty |
| [Likelihood Profiling](likelihood_profiler.md) | Profile-based confidence intervals |
| [Optimization](optimization.md) | Stable training via adaptive learning and backtracking |

---

## Inference Modes Supported

- **Maximum likelihood** estimation
- **Posterior variance approximation**
- **Trait-dependent diversification**
- **Empirical vs simulated likelihood comparison**
- **Parallel profiling and batched optimization**

---

## Ideal Use Cases

- Phylogenetic inference from DNA, RNA, or protein sequences
- Molecular dating of trees with uncertain divergence times
- Parameter sensitivity analysis and identifiability diagnostics
- Fitting state-dependent speciation models
- Simulation of evolutionary processes for benchmarking

---

For in-depth technical details, see the linked module documentation.
