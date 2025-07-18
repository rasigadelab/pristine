# Optimization in PRISTINE

This document provides a mathematical and algorithmic breakdown of the optimizer used in the PRISTINE framework. The optimizer is designed for maximum-likelihood inference tasks where model parameters are fitted by minimizing a differentiable loss function, typically derived from the negative log-likelihood of observed data under a probabilistic model.

---

## Overview

The `Optimizer` class in `optimize.py` implements robust gradient-based parameter optimization using the **Adam** algorithm with adaptive learning rate control and **backtracking**. It targets the minimization of a differentiable loss function:

$$
\mathcal{L}(\theta) = - \log p(\text{data} \mid \theta)
$$

where $\theta \in \mathbb{R}^n$ is the vector of free model parameters. The objective is:

$$
\theta^* = \arg\min_\theta \mathcal{L}(\theta)
$$

The optimizer operates in discrete steps and includes:
- Adam update rule with momentum and adaptive variance normalization
- Learning rate backtracking when $\mathcal{L}(\theta_{t+1}) > \mathcal{L}(\theta_t) + \varepsilon$
- Early stopping on convergence or too-small learning rate

---

## Mathematical Formulation and Inner Workings

### 1. Objective

Given a differentiable model loss $\mathcal{L}(\theta)$, compute gradients:

$$
g_t = \nabla_\theta \mathcal{L}(\theta_t)
$$

### 2. Adam Update Rule

Maintain moving averages of gradient and squared gradient:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}
$$

Bias-corrected estimates:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parameter update:

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $\alpha$ is the learning rate.

### 3. Loss Evaluation and Backtracking

After updating parameters:

- Evaluate $\mathcal{L}(\theta_{t+1})$
- If $\mathcal{L}(\theta_{t+1}) > \mathcal{L}(\theta_t) + \varepsilon$, reduce learning rate $\alpha \leftarrow \rho \alpha$, revert update, retry

Repeat until:
- Improvement is sufficient
- Learning rate $< \alpha_{\min}$ → stop

---

## Practical Safeguards

- **Gradient sanity checks**: abort if NaNs or infinities are found in parameters or gradients
- **Adaptive LR recovery**: re-accelerates learning rate when updates stabilize
- **Stopping conditions**:
    - Max iterations
    - Loss convergence
    - Learning rate decay limit

---

## Python Implementation Notes

- Uses `torch.optim.Adam`
- Backtracking loop wraps optimizer step with retry logic
- Convergence threshold $\varepsilon$ and backtrack decay $\rho$ are configurable

---

## Applicability

This optimizer is ideal for:
- Fitting parameters in models with complex loss surfaces
- Phylogenetic likelihood inference
- Numerical stability in curved or sloppy landscapes

