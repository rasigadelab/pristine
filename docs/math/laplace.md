# Laplace Approximation for Parameter Uncertainty

## Overview

The `LaplaceEstimator` class implements a second-order approximation method for quantifying uncertainty in model parameters. Around a local optimum of the loss function (typically the negative log-likelihood), the function is approximated as quadratic, implying a multivariate Gaussian posterior distribution over parameters.

The Laplace approximation enables the estimation of:
- Marginal parameter variances
- Confidence intervals
- Local curvature (eigenvalue spectrum)

It supports both exact Hessian inversion for small models and a scalable approximation (Hutchinson’s method) for larger models.

---

## Mathematical Background

Let $\mathcal{L}(\boldsymbol{\theta})$ be the negative log-likelihood. Near the maximum likelihood estimate (MLE) $\hat{\boldsymbol{\theta}}$, we approximate:

$$
\mathcal{L}(\boldsymbol{\theta}) \approx \mathcal{L}(\hat{\boldsymbol{\theta}}) + \frac{1}{2} (\boldsymbol{\theta} - \hat{\boldsymbol{\theta}})^T H (\boldsymbol{\theta} - \hat{\boldsymbol{\theta}})
$$

where $H$ is the Hessian of the loss at $\hat{\boldsymbol{\theta}}$. The posterior is then:

$$
p(\boldsymbol{\theta} \mid \text{data}) \approx \mathcal{N}(\hat{\boldsymbol{\theta}}, H^{-1})
$$

### Marginal Variance

The marginal variance of parameter $\theta_i$ is:

$$
\text{Var}[\theta_i] \approx \left[H^{-1}\right]_{ii}
$$

Confidence intervals are estimated as:

$$
\theta_i \pm z_{\alpha/2} \cdot \sqrt{\left[H^{-1}\right]_{ii}}
$$

for a given confidence level (e.g., $z_{0.975} = 1.96$ for 95%).

---

## Estimation Methods

### Exact Dense Hessian

For small models, the full Hessian $H$ is computed and inverted:

```python
H = compute_dense_hessian()
Hinv = torch.linalg.inv(H)
```

The diagonal of $H^{-1}$ gives marginal variances【67†source】.

### Hutchinson’s Estimator (Large Models)

When $H$ is too large to invert, the estimator uses Hutchinson’s method:

1. Sample random vectors $v_i \sim \text{Rademacher}$
2. Solve $H x_i = v_i$ using conjugate gradient
3. Estimate:

$$
\text{diag}(H^{-1}) \approx rac{1}{M} \sum_{i=1}^M v_i \odot x_i
$$

This avoids computing or storing the full Hessian【67†source】.

---

## API Highlights

- `estimate_inv_hessian_diag()`: estimates the full diagonal of $H^{-1}$
- `estim_variance_by_name(name)`: returns variance for a named parameter
- `estim_all_confint_dict()`: returns 95% confidence intervals
- `curvature_report()`: eigenvalue analysis of $H$ to detect sloppy or non-identifiable directions【67†source】

---

## Curvature Analysis

To detect identifiability issues, eigenvalue bounds of $H$ are estimated:

- **Largest eigenvalue**: via power iteration
- **Smallest eigenvalue**: via inverse iteration

The flatness ratio is:

$$
\frac{\lambda_{\min}}{\lambda_{\max}}
$$

Interpretation:
- $\gtrsim 10^{-2}$: well-conditioned
- $10^{-6} \lesssim \cdot \lesssim 10^{-2}$: mild sloppiness
- $\ll 10^{-6}$: severe non-identifiability【67†source】

---

## Limitations

- Assumes the loss is locally quadratic
- Fails under flat or ill-posed likelihoods
- Hutchinson estimates are approximate and depend on sample size

---

## References

- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.

---

## Source

Defined in [`laplace_estimator.py`](../laplace_estimator.py)【67†source】.
