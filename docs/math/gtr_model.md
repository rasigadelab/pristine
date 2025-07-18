# Generalized Time-Reversible (GTR) Model

## Overview

The GTR model is a flexible and widely used continuous-time Markov model (CTMM) for modeling sequence evolution in phylogenetics. It defines a stationary and time-reversible substitution process over a finite set of states (e.g., nucleotides or amino acids).

The `GeneralizedTimeReversibleModel` class in the PRISTINE framework provides a fully differentiable implementation of this model. It supports gradient-based estimation of evolutionary parameters directly from data and provides efficient routines for computing matrix exponentials needed in likelihood calculations.

This model is parameterized by:
- A stationary distribution $\boldsymbol{\pi}$ over the $K$ states
- A symmetric matrix of exchange rates $R$ between state pairs $(i, j)$

The rate matrix $Q$ is then derived from these components.

---

## Continuous-Time Markov Models (CTMM)

In a CTMM, the probability of transitioning from state $i$ to $j$ over time $t$ is described by the matrix exponential of a rate matrix $Q$:

$$
P(t) = \exp(Qt)
$$

The entries of $Q$ satisfy:
- $Q_{ij} \geq 0$ for $i \neq j$ (substitution rates)
- $\sum_{j} Q_{ij} = 0$ for all $i$ (rows sum to zero)

The diagonal entries are defined as:

$$
Q_{ii} = -\sum_{j \neq i} Q_{ij}
$$

---

## GTR Parameterization

The GTR model assumes time-reversibility, i.e., $\pi_i Q_{ij} = \pi_j Q_{ji}$.

It is parameterized via:
- $\boldsymbol{\pi}$: stationary distribution (a probability vector)
- $R$: symmetric exchangeability matrix

For $i \neq j$, the off-diagonal rate is:

$$
Q_{ij} = \pi_j R_{ij}
$$

Diagonal entries are set to ensure rows sum to zero:

$$
Q_{ii} = -\sum_{j \neq i} Q_{ij}
$$

This guarantees that $\boldsymbol{\pi}$ is the stationary distribution of $Q$.

### Stationary Distribution

Instead of directly optimizing $\pi$, the GTR model uses an unconstrained parameterization via logits $\boldsymbol{\ell} \in \mathbb{R}^{K-1}$. The first component of $\pi$ is pinned to act as reference:

$$
\pi_0 = \frac{1}{1 + \sum_{i=1}^{K-1} \exp(\ell_i)}, \quad
\pi_i = \frac{\exp(\ell_i)}{1 + \sum_{j=1}^{K-1} \exp(\ell_j)}, \quad i = 1,\dots,K-1
$$

This ensures $\pi$ is a valid probability distribution (non-negative and sums to one) while reducing redundancy.

### Exchangeability Matrix $R$

Only the upper triangle (excluding diagonal) of $R$ is parameterized. These values are exponentiated for positivity and symmetrically assigned:

$$
R_{ij} = R_{ji} = \exp(r_{ij}), \quad i < j
$$

Then the full $Q$ matrix is constructed as described above.

---

## Implementation Details

The `GeneralizedTimeReversibleModel` class supports:

- `stationary_dist()`: returns $\boldsymbol{\pi}$
- `rate_matrix_stationary_dist()`: returns the full rate matrix $Q$ and $\boldsymbol{\pi}$
- `average_rate()`: computes the mean substitution rate $\mu = -\sum_i \pi_i Q_{ii}$
- `compute_batch_matrix_exp(durations)`: efficiently computes $\exp(Qt)$ for a batch of durations

Internally, the model builds $Q$ via:

```python
Q = pi.unsqueeze(0) * R
Q.fill_diagonal_(0.0)
Q.diagonal().copy_(-Q.sum(dim=1))
```

which guarantees that $Q$ is valid and satisfies detailed balance.

---

## Practical Usage

This model is typically used with the Felsenstein pruning algorithm to compute likelihoods on phylogenetic trees. In the PRISTINE framework, it integrates with:

- `FelsensteinPruningAlgorithm` for sequence likelihoods
- `SequenceSimulationVisitor` for simulating sequence evolution
- Molecular clock models for dating trees

The GTR model is essential for inferring both phylogenies and substitution dynamics from sequence data.

---

## References

- Yang, Z. (2014). *Molecular Evolution: A Statistical Approach*. Oxford University Press.
- Felsenstein, J. (1981). *Evolutionary trees from DNA sequences: a maximum likelihood approach*. Journal of Molecular Evolution.

---

## Source

Defined in [`gtr.py`](../gtr.py)【26†source】.
