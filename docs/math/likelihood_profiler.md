\
# Likelihood Profiler

## Overview

The `LikelihoodProfiler` class estimates confidence intervals for model parameters using **likelihood profiling**. Unlike local curvature-based methods (e.g. Laplace approximation), it evaluates how the likelihood changes when a single parameter is perturbed, while re-optimizing all other parameters.

This approach is more robust for non-quadratic or asymmetric likelihood surfaces and is especially suited for **maximum likelihood inference** where parameters are interdependent or poorly constrained.

---

## Likelihood Profiling

Let $\hat{\theta}$ be the maximum likelihood estimate of a parameter, and let $\mathcal{L}(\theta)$ be the log-likelihood at that value. The **confidence interval** at level $1 - \alpha$ is estimated by solving:

$$
\mathcal{L}(\theta) \geq \mathcal{L}(\hat{\theta}) - \frac{1}{2} \chi^2_{1, 1-\alpha}
$$

This is done by fixing $\theta$, re-optimizing all other parameters, and checking the resulting log-likelihood.

---

## Bracketing and Optimization

The main method `profile__()` performs a root-finding search using Brent's method to identify interval bounds:

- Starts from the MLE and explores a range defined by a multiple of the parameter’s standard deviation.
- If no solution is found in the initial bracket, the search interval is **adaptively expanded** up to a configurable number of times【115†source】.
- Re-optimization is performed at each step using the `Optimizer` class.

Fallbacks are triggered when optimization fails or bounds do not diverge from the center. In these cases, a **Laplace-based interval** is returned【115†source】.

---

## Grid Profiling

The `profile()` method evaluates the likelihood on a fixed grid around the MLE:

1. Construct a linearly spaced grid.
2. For each grid point, re-optimizes the model.
3. Interpolates the log-likelihood profile to find the confidence bounds【115†source】.

This variant is slower but does not depend on root-finding convergence.

---

## Confidence Interval Estimation

- `estimate_confint()` returns a pair $(\theta_{\text{lower}}, \theta_{\text{upper}})$ for a given confidence level.
- The default uses 95% confidence, corresponding to a $\chi^2_1$ threshold of approximately 3.84.
- Internally computes:

$$
\Delta \log \mathcal{L} = \frac{1}{2} \chi^2_{1, 1 - \alpha}
$$

---

## Parallel and Batch Profiling

- `estimate_confint_all()`: loops over parameters sequentially.
- `estimate_confint_all_parallel()`: uses `ThreadPoolExecutor` to parallelize across CPU threads【115†source】.
- `estimate_confint_all_laplace()`: computes intervals using only the Laplace approximation for all parameters, with optional dense or Hutchinson inversion.

---

## Robustness and Fallbacks

If any profiling fails due to poor optimization convergence or lack of bracketing, a symmetric confidence interval is computed:

$$
\hat{\theta} \pm z_{\alpha/2} \cdot \sqrt{\text{Var}(\hat{\theta})}
$$

using the Laplace variance and normal quantiles【115†source】.

---

## References

- Pawitan, Y. (2001). *In All Likelihood: Statistical Modelling and Inference Using Likelihood*. Oxford.
- Bolker, B. (2008). *Ecological Models and Data in R*. Princeton University Press.
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.

---

## Source

Defined in [`likelihood_profiler.py`](../likelihood_profiler.py)【115†source】.
