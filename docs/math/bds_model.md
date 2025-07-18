# Birth-Death-Sampling (BDS) Models

## Overview

The `bds_model.py` module implements phylogenetic models based on **birth-death-sampling (BDS)** processes. These models describe how lineages speciate (birth), go extinct (death), and are sampled over evolutionary time. They are particularly suited for modeling time-calibrated trees under epidemiological or macroevolutionary scenarios.

The module includes:
- Time-homogeneous and state-dependent BDS likelihoods
- Analytical expressions for survival probabilities (`q(t)`) and extinction probabilities (`p₀(t)`)
- Simulation utilities for stochastic tree generation
- Extensions with linear state-dependent birth rates

---

## Core Quantities

Let $b$, $d$, and $s$ be the birth, death, and sampling rates. The **Stadler q-function**:

$$
q(t) = 2(1 - c^2) + e^{-ct}(1 - c)^2 + e^{ct}(1 + c)^2
$$

with:

$$
c = \frac{-(b - d - s)}{\sqrt{(b - d - s)^2 + 4bs}}
$$

governs the contribution of each branch to the tree likelihood【123†source】.

The **extinction probability** $p_0(t)$ is:

$$
p_0(t) = \frac{b + d + s + c \cdot \frac{e^{-ct}(1 - c) - (1 + c)}{e^{-ct}(1 - c) + (1 + c)}}{2b}
$$

---

## Tree Likelihood: Homogeneous Model

For a tree with $E$ edges and $T$ tips, the log-likelihood is:

$$
\log \mathcal{L} = \sum_{e \in \text{edges}} \log \frac{q(t_e^\text{child})}{q(t_e^\text{parent})} + (T-1)\log b + T \log s
$$

Implemented by:

```python
BirthDeathSamplingTreeWise.log_likelihood(treecal)
```

---

## State-Dependent BDS

Allows birth, death, and sampling to vary by hidden discrete state. Given per-edge parent state probabilities $p_k$, the likelihood term is:

$$
\sum_{e \in \text{edges}} \log \left( \sum_k p_k \cdot \frac{q_k(t_\text{child})}{q_k(t_\text{parent})} \cdot r_k \right)
$$

where $r_k$ is $b_k$ or $s_k$ depending on whether the child is internal or a tip.

Implemented by:

```python
StateDependentBirthDeathSampling.log_likelihood(treecal, ancestor_states)
```

---

## Simulation: BirthDeathSamplingSimulator

Stochastically grows a tree forward in time using specified BDS rates. Events:
- **Birth**: splits lineage into two children
- **Death**: terminates lineage
- **Sampling**: marks lineage as a tip

Simulation continues until a target number of tips is sampled【123†source】.

---

## Extensions: Linear Marker Birth Model

The `LinearMarkerBirthModel` allows the birth rate to depend linearly on hidden marker states:

$$
b_n = \exp\left(\beta_0 + \sum_{m,k>0} \beta_{mk} \cdot p_{n,m,k}\right)
$$

This enables modeling trait-dependent diversification. The log-likelihood aggregates over edges:

$$
\sum_e \log \left( \frac{q_{\text{child}}}{q_{\text{parent}}} \right) + \sum_{\text{birth events}} \log b + \sum_{\text{tips}} \log s
$$

---

## API Summary

### Likelihood Classes

- `BirthDeathSamplingTreeWise`: homogeneous log-likelihood
- `StateDependentBirthDeathSampling`: per-state likelihood with posterior weighting
- `LinearMarkerBirthModel`: linear trait-dependent birth rate

### Simulation

- `BirthDeathSamplingSimulator`: forward-in-time stochastic simulation
- `BirthDeathSamplingNodeData`: holds per-node rates

### Utilities

- `stadler_q(t, b, d, s)`: scalar q function
- `stadler_q_matrix`: vectorized q(t) over nodes and states
- `stadler_p0`: extinction probability

---

## References

- Stadler, T. (2010). Sampling-through-time in birth–death trees. JTB.
- Maddison, W. P., et al. (2007). Estimating a binary character’s effect on speciation.
- Rabosky, D. L. (2014). Automatic detection of key innovations.

---

## Source

Defined in [`bds_model.py`](../bds_model.py)【123†source】.
