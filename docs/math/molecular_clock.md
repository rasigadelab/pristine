# Molecular Clock Models

## Overview

Molecular clock models describe how genetic divergence accumulates over time along the branches of a phylogenetic tree. They are essential for calibrating node times and estimating evolutionary rates. The PRISTINE framework implements two main clock models:

- `ConditionalErrorClock`: a binomial noise model based on JC69 substitution probabilities.
- `ContinuousAdditiveRelaxedClock`: a gamma-distributed model for rate heterogeneity.

---

## 1. Conditional Error Clock

This model assumes that the observed distance $d$ between two sequences is a noisy estimate of the true expected distance, which follows a Jukes-Cantor 1969 (JC69) model:

$$
p(l) = \frac{K - 1}{K} \left(1 - \exp\left(-\frac{K}{K-1} l\right)\right)
$$

The likelihood is modeled as a binomial draw:

$$
\log \Pr(D \mid l) = \log \binom{L}{L p(d)} + L p(d) \log p(l) + L (1 - p(d)) \log(1 - p(l))
$$

where:
- $L$ is the number of sites
- $p(d)$ is the empirical JC69 probability for observed distance $d$
- $p(l)$ is the expected JC69 probability at branch length $l = r \cdot t$ (rate × duration)

This model is used when the observed distance is fixed, and the goal is to estimate branch durations【124†source】.

---

## 2. Continuous Additive Relaxed Clock (cARC)

This model assumes that branch-specific substitution rates are gamma-distributed:

- Mean rate: $\mu = \exp(\text{log\_rate})$
- Dispersion: $\phi$
- Shape parameter: $\alpha = \mu \cdot L \cdot t \cdot r$
- Rate parameter: $\beta = \frac{1}{1 + \phi}$

Then the likelihood of observing a distance $d$ is:

$$
\log p(d \mid t) = \log \text{Gamma}(d \cdot L; \alpha, \beta)
$$

To ensure numerical stability, a barrier penalty is applied to the gamma shape:

$$
\text{barrier}(x) = \text{softplus}(-x, \text{strength}) \cdot \text{strength}
$$

This model generalizes the strict clock and allows rates to vary across lineages【124†source】.

---

## Simulation

The `simulate()` method in `ContinuousAdditiveRelaxedClock` draws distances:

$$
d \sim \text{Gamma}(\alpha, \beta), \quad d = \text{distance} / L
$$

This supports stochastic modeling of genetic divergence for inference or synthetic data generation.

---

## API

### ConditionalErrorClock
- `log_likelihood(durations, distances)`: binomial JC69 likelihood
- `rate()`: returns $\exp(\text{log\_rate})$

### ContinuousAdditiveRelaxedClock
- `log_likelihood(durations, distances)`: gamma log-likelihood
- `simulate(durations)`: draw distances from gamma
- `rate()`: mean substitution rate

---

## Initialization Utilities

- `new_conditional_error_clock(K, L)`: creates a `ConditionalErrorClock`
- `new_continuous_additive_relaxed_clock(L)`: creates a `ContinuousAdditiveRelaxedClock` with default parameters【124†source】

---

## References

- Jukes, T. H., & Cantor, C. R. (1969). Evolution of protein molecules.
- Drummond, A. J., & Suchard, M. A. (2010). Bayesian random local clocks.
- Yang, Z. (2014). *Molecular Evolution: A Statistical Approach*. Oxford University Press.

---

## Source

Defined in [`molecularclock.py`](../molecularclock.py)【124†source】.
