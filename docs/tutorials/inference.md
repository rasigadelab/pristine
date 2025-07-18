# PRISTINE Tutorial: End-to-End Phylogenetic Inference

This tutorial introduces the PRISTINE framework through worked examples that guide you through simulation, model construction, and parameter inference for a variety of phylogenetic settings.

---

## 🔁 Sequence Evolution and Substitution Model Inference

Learn how to simulate sequences along a phylogenetic tree and recover GTR substitution parameters:

📄 [example_03_fpa_gtr.py](example_03_fpa_gtr.py)

Key steps:
- Simulate DNA sequences along a tree using a known GTR model
- Build a likelihood function using the Felsenstein pruning algorithm
- Optimize the GTR parameters to recover stationary frequencies and exchange rates

---

## 🕒 Molecular Clock Estimation

Estimate divergence times and substitution rates under different clock models:

### Continuous Additive Relaxed Clock (cARC)

📄 [example_01_carc.py](example_01_carc.py)

- Simulate distances using a relaxed molecular clock
- Fit node dates and evolutionary rate using maximum likelihood
- Compare estimated vs true node ages

### Conditional Error Clock (JC69-based)

📄 [example_02_cdclock.py](example_02_cdclock.py)

- Use a binomial model based on JC69 substitution probabilities
- Fit branch durations from simulated distances
- Suitable for shorter sequences or simpler models

---

## 🌳 Joint Estimation of Phylogeny and Divergence

Recover both substitution dynamics and divergence times:

📄 [example_04_fpa_dating.py](example_04_fpa_dating.py)

- Simulate sequences with a known GTR model
- Fit GTR parameters and internal node dates simultaneously
- Illustrates parameter entanglement and numerical optimization

---

## 🌱 Diversification Inference: Birth-Death-Sampling (BDS)

### Constant-Rate BDS

📄 [example_05_bds_constant.py](example_05_bds_constant.py)

- Simulate trees under a fixed birth and sampling process
- Estimate log-likelihood using analytic formulas
- Fit birth and sampling rates from tree shape

### State-Dependent BDS

📄 [example_06_bds_multistate.py](example_06_bds_multistate.py)

- Simulate sequences that encode hidden states
- Assign state-dependent birth rates
- Use ancestral state probabilities to compute likelihoods

### Linear Trait-Dependent BDS

📄 [example_07_bds_linear.py](example_07_bds_linear.py)

- Simulate sequences under a 2-state GTR model
- Let birth rate depend linearly on hidden traits
- Estimate trait effects via maximum likelihood

---

## 🧮 Optimization and Inference Tools

All examples rely on the robust Adam optimizer with backtracking:

- Gradient-based optimization
- Learning rate adaptation
- Safe fallback for numerical instability

The optimizer is defined in [`optimize.py`](/pristine/math/optimization/).

To assess uncertainty and parameter identifiability:
- Use [Laplace approximation](/pristine/math/laplace/) for posterior variance
- Use [Likelihood profiling](/pristine/math/likelihood_profiler/) for non-quadratic confidence intervals

---

## Next Steps

Explore the [overview page](/pristine/math/pristine_overview/) for conceptual organization and links to each module’s documentation.

All examples are designed to be runnable and modifiable. To explore further:
- Change the number of tips or states
- Add noise via clock dispersion
- Enable curvature diagnostics to test identifiability

