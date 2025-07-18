# Felsenstein Pruning Algorithm (FPA)

## Overview

The Felsenstein pruning algorithm (FPA) is a dynamic programming method to compute the likelihood of observed sequences at the tips of a phylogenetic tree under a substitution model. It works by traversing the tree from the tips to the root in post-order, propagating conditional likelihoods and applying transition probabilities computed from a rate matrix.

In the PRISTINE framework, the `FelsensteinPruningAlgorithm` class implements this procedure in a differentiable, TorchScript-compatible form. It supports both probability-space and log-space formulations, and handles multiple unique site patterns efficiently.

---

## Mathematical Foundations

Let:
- $L$ be the number of alignment sites
- $K$ be the number of possible states (e.g. 4 for DNA)
- $N$ be the number of nodes in the tree
- $P(t) = e^{Qt}$ be the transition matrix over duration $t$ with rate matrix $Q$

Let $f_{n\ell k}$ be the conditional likelihood that node $n$ emits state $k$ at site $\ell$. For leaves, this is a one-hot vector based on observed data. For internal nodes, we compute:

$$
f_{n\ell k} = \prod_{c \in 	ext{children}(n)} \sum_{j=1}^K P_{kj}(t_{nc}) f_{c\ell j}
$$

At the root, the marginal likelihood per site is:

$$
\mathcal{L}_\ell = \sum_{k=1}^K \pi_k f_{0\ell k}
$$

The overall log-likelihood is then:

$$
\log \mathcal{L} = \sum_{\ell=1}^L \log \mathcal{L}_\ell
$$

---

## Algorithm Details

The FPA implementation in PRISTINE uses the following approach:

### 1. Tree Recursion Structure

Edges are grouped into recursion levels using `get_postorder_edge_list`, ensuring that edges in each level do not depend on those in later levels【65†source】.

### 2. Forward Evaluation (Probability Space)

- **Initialization**: Tip nodes use observed sequences; internal nodes are initialized uniformly.
- **Edge messages**: For each edge, compute:

$$
	ext{msg}_{e,\ell,k} = \sum_j f_{c\ell j} P_{jk}(t_e)
$$

- **Accumulation**: Log-messages from children are accumulated at parents:

$$
\log f_{p\ell k} = \log f_{p\ell k} + \sum_{c} \log 	ext{msg}_{c 	o p}
$$

- **Normalization**: After accumulation, the conditional likelihood at each node is normalized and the scale factor is stored in `log_scaling`.

- **Final likelihood**: At the root, compute the total log-likelihood across all sites with pattern weights【65†source】.

### 3. Log-Space Variant

To improve numerical stability, the method `log_likelihood_logspace_` implements the full algorithm in log-space:

- Likelihoods are propagated as $\log f$
- Multiplications become additions
- Sums become log-sum-exp operations

This is more robust for long trees or many sites, but slightly slower【65†source】.

---

## Numerical Stability

To avoid underflow, both the probability-space and log-space implementations use per-site log-scaling terms. These scalings are stored and accumulated during recursion, ensuring that likelihood values remain in a computable range:

$$
	ext{CL}_{	ext{true}} = \exp(	ext{log\_scaling}) \cdot 	ext{node\_probs}
$$

---

## Output and Posterior States

The field `ancestor_states` stores the posterior state probabilities at internal nodes. These are available for downstream tasks such as:

- Ancestral reconstruction
- Trait correlation with divergence
- Structured birth-death likelihoods

---

## Practical Considerations

- **Vectorization**: All site computations are vectorized over alignment patterns.
- **Batch support**: Transition matrices are precomputed in batches for each edge.
- **Tree traversal**: Performed in level-by-level recursion groups for parallelization.

---

## References

- Felsenstein, J. (1981). *Evolutionary trees from DNA sequences: a maximum likelihood approach*. Journal of Molecular Evolution.
- Yang, Z. (2014). *Molecular Evolution: A Statistical Approach*. Oxford University Press.

---

## Source

Defined in [`felsenstein.py`](../felsenstein.py)【65†source】.
