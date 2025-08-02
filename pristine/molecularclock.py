# -----------------------------------------------------------------------------
# This file is part of the PRISTINE framework for statistical computing
#
# Copyright (C) Jean-Philippe Rasigade
# Hospices Civils de Lyon, and University of Lyon, Lyon, France
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing is available upon request. Please contact the author.
# -----------------------------------------------------------------------------

"""
Molecular Clock Models

Molecular clock models describe how genetic mutations accumulate over time
along the branches of a phylogenetic tree. By linking the expected number
of substitutions to elapsed time, these models provide a framework for
estimating divergence times from sequence data.

Simple models assume a constant substitution rate (a strict clock), while
more flexible variants allow rates to vary across lineages (relaxed clocks).
Such models are essential for combining phylogenetic structure with temporal
information, enabling both dating of ancestral nodes and inference of
evolutionary parameters.

ConditionalErrorClock

This class defines a simplified evolutionary clock model that links branch
durations to genetic distances using a probabilistic formulation based on
Jukes-Cantor (JC69) substitution probabilities. It assumes that observed
distances are noisy estimates of the expected substitution probabilities
determined by branch length. The likelihood is constructed by comparing
these expected probabilities to empirical ones using a binomial model,
effectively treating the number of observed substitutions as a noisy
binomial draw from an underlying continuous-time process. It is especially
useful for inference where observed distances are fixed and durations are
parameters to optimize.

ContinuousAdditiveRelaxedClock

This class implements a continuous-time additive relaxed clock model where
the substitution rate is not constant but follows a gamma distribution.
Durations (times) are scaled by a base rate and sequence length, and
the resulting expected distances follow a gamma distribution, capturing
extra variability across branches. A dispersion parameter governs how much
relaxation from a strict clock is allowed—zero dispersion recovers a
constant-rate clock. The model defines both simulation and likelihood
functions, and is well suited for Bayesian or maximum-likelihood inference
where evolutionary rates vary stochastically across lineages.

"""
import torch
import math
from typing import Optional, Tuple
from . import distribution as D
from .edgelist import TreeTimeCalibrator
from .qma import QMAGrid

#########################################################################
# CONDITIONAL CLOCK MODEL
#########################################################################

@torch.jit.script
class ConditionalErrorClock:
    def __init__(
        self,
        treecal: TreeTimeCalibrator,
        num_states: int,
        sequence_length: float,
        log_rate: Optional[torch.Tensor] = None
    ):
        """
        States and length are used as float in computations so we convert them
        once during initialization
        """
        self.treecal: TreeTimeCalibrator = treecal
        self.num_states: torch.Tensor = torch.tensor(num_states)
        self.sequence_length: torch.Tensor = torch.tensor(sequence_length)
        self.log_rate = log_rate if log_rate is not None else \
            torch.tensor(0.).requires_grad_(True)

    def rate(self)->torch.Tensor:
        return self.log_rate.exp()

    def log_likelihood(self)->torch.Tensor:
        r"""
        The log-likelihood of the waiting time l for a MLE of evolutionary distance (subt./site) d_hat
        estimated from a sequence of length L:
        
            log Binom[ L p(d) | L, p(l) ] where

            p(l) = (K-1)/K (1 - exp(-K/(K-1) * l)) is given by the JC69_probability() function
        
        Args:
            l (torch.Tensor): Tensor of candidate branch lengths.
            d_hat (torch.Tensor): Tensor of observed branch lengths (or proxies) that yield p(d_hat).
            L (float): A scalar parameter (e.g. number of sites).
            K (float): The number of states.
            eps (float): Small constant added for numerical stability.
            
        Returns:
            torch.Tensor: The loss value computed elementwise.
            
        Note:
            All operations are fully vectorized over the input tensors.
        """
        # Compute p(l) and p(d_hat)
        K = self.num_states
        L = self.sequence_length
        durations = self.treecal.durations()
        scaled_durations = durations.clamp(1e-8) * self.log_rate.exp()
        p_durations = D.JC69_probability(scaled_durations, K)
        p_distances = D.JC69_probability(self.treecal.distances, K)

        # Compute the log coefficient using lgamma (log Gamma)
        # Note: torch.lgamma(x) = log(Γ(x))
        log_coeff = torch.lgamma(L + 1.0) - \
                    torch.lgamma(L * p_distances + 1) - \
                    torch.lgamma(L * (1 - p_distances) + 1)
        
        # Compute the log likelihood (each term is computed elementwise)
        loglik = log_coeff + L * p_distances * torch.log(p_durations) + L * (1.0 - p_distances) * torch.log(1.0 - p_durations)
        
        return loglik - D.barrier_positive(durations).sum()
        
    def loss(self) -> torch.Tensor:
        return -self.log_likelihood().sum()



#########################################################################
# cARC MODEL
#########################################################################

@torch.jit.script
class ContinuousAdditiveRelaxedClock:
    def __init__(
        self,
        treecal: TreeTimeCalibrator,
        sequence_length: Optional[float] = None,
        log_rate: Optional[torch.Tensor] = None,
        dispersion: Optional[torch.Tensor] = None
    ):
        self.treecal: TreeTimeCalibrator = treecal
        self.sequence_length = sequence_length if sequence_length is not None else 1.0
        self.log_rate = log_rate if log_rate is not None else \
            torch.tensor(0.).requires_grad_(True)
        self.dispersion = dispersion if dispersion is not None else \
            torch.tensor(1.).requires_grad_(True)
    
    def rate(self)->torch.Tensor:
        return self.log_rate.exp()

    def _gamma_shape_rate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rate = 1.0 / (1.0 + self.dispersion)
        shape = self.rate() * self.sequence_length * self.treecal.durations() * rate
        return shape, rate

    def simulate(self) -> torch.Tensor:
        """
        Simulate genetic distances under the continuous additive relaxed clock model.
        """
        shape, rate = self._gamma_shape_rate()
        self.treecal.distances = torch._standard_gamma(shape) / rate / self.sequence_length
        return self.treecal.distances

    def log_likelihood(self)->torch.Tensor:
        """
        Log-likelihood of durations and distances under the continuous additive relaxed clock model.
        """
        shape, rate = self._gamma_shape_rate()
        scaled_distances = self.treecal.distances * self.sequence_length

        return (
            D.gamma_log_probability(scaled_distances, shape, rate) 
            - D.barrier_positive(shape)
        )
        
    def loss(self) -> torch.Tensor:
        return -self.log_likelihood().sum()
    
#########################################################################
# BASE CONSTANT CLOCK MODEL
#########################################################################

@torch.jit.script
class ConstantClock:
    def __init__(self, log_rate: Optional[torch.Tensor] = None):
        self.log_rate = log_rate if log_rate is not None else \
            torch.tensor(0.).requires_grad_(True)
        
    def rate(self):
        return self.log_rate.exp()
    
#########################################################################
# Relaxed Gamma clock model (uncorrelated lognormal for distances)
#########################################################################

@torch.jit.script
class RelaxedGammaClock:
    r"""
    A relaxed-clock model for phylogenetic branch lengths that places a Normal prior 
    on the log-substitution rate and uses Quantile Midpoint Approximation (QMA) to 
    marginalize over per-edge rate variation in a Gamma-distributed distance model.

    This class can both simulate genetic distances under the model and compute
    an approximate marginal log-likelihood for observed distances given a calibrated tree.

    Model summary
    -------------
    - Latent per-edge substitution rates mu_e are drawn (in simulation) from
         log mu_e ~ Normal(mean=log_rate_mean, std=exp(log_rate_log_std))
    - Genetic distances d_e are simulated via
         d_e = (1/num_markers) · Gamma(shape = mu_e · tau_e · num_markers, rate = 1)
      where tau_e is the edge duration from `treecal`.
    - In inference, the model marginalizes out mu_e by evaluating the provided QMA grid:
         theta_q = log_rate_mean + exp(log_rate_log_std) · z_q,
         mu_q    = exp(theta_q),
      where z_q and the corresponding weights are taken from `qma_grid`.
    - The per-edge likelihood is
         p(d_e) ≈ sum_q [ w_q · Gamma(d_e·num_markers | shape=mu_q·tau_e·num_markers, rate=1) ]
      where w_q are the weights from the QMA grid.  We compute a log-sum-exp over q
      for each edge and return one log-likelihood per edge.

    Parameters
    ----------
    treecal : TreeTimeCalibrator
        A calibrator object providing:
          - `.durations()` → tensor of branch durations tau_e, shape (E,)
          - mutable attribute `.distances` for simulation, shape (E,)
    num_markers : int
        Number of independent markers (e.g., sites) used in distance simulation.
    log_rate_mean : torch.Tensor, optional
        Initial value of the prior mean of log-rate (mu in log-space).  Scalar tensor
        with requires_grad=True to permit learning.
        Defaults to 0.0 (i.e. prior mean rate = 1.0).
    log_rate_log_std : torch.Tensor, optional
        Initial value of the log of the prior standard deviation of log-rate
        (sigma in log-space).  Scalar tensor with requires_grad=True.
        Defaults to -2.0 (i.e. prior sigma = exp(-2) in log-space).
    qma_grid : QMAGrid, optional
        A QMA grid object that provides:
          - `nodes`: the quantile points for theta = log-rate
          - `weights`: the integration weights for each node
        If None, a default symmetric 8-point grid is used.

    Methods
    -------
    simulate() → torch.Tensor
        Draws per-edge log-rates from the Normal prior, computes Gamma-distributed
        distances, writes them into `treecal.distances`, and returns that tensor.
    log_likelihood() → torch.Tensor
        Computes the QMA-marginalized log-likelihood vector of shape (E,).  For each
        edge e, it evaluates the Gamma log-probability at the Q quantile rates,
        then applies log-mean-exp across Q and subtracts any branch-duration barrier.
    loss() → torch.Tensor
        Returns the negative sum of the per-edge log-likelihoods, suitable for gradient‐
        based optimization.

    Example
    -------
    >>> # simulation
    >>> clock = RelaxedGammaClock(treecal, num_markers=100)
    >>> d = clock.simulate()     # d has shape (E,)
    >>>
    >>> # inference
    >>> clock = RelaxedGammaClock(treecal, num_markers=100)
    >>> loss = clock.loss()
    >>> loss.backward()
    >>> # optimize log_rate_mean and log_rate_log_std to maximize marginal likelihood
    """
    def __init__(self,
                treecal: TreeTimeCalibrator,
                num_markers: int,
                log_rate_mean: Optional[torch.Tensor] = None,
                log_rate_log_std: Optional[torch.Tensor] = None,
                qma_grid: Optional[QMAGrid] = None
                ):
        self.treecal: TreeTimeCalibrator = treecal
        self.log_rate_mean: torch.Tensor = log_rate_mean if log_rate_mean is not None else \
            torch.tensor(0.).requires_grad_(True)                # μ in log-space
        self.log_rate_log_std: torch.Tensor = log_rate_log_std if log_rate_log_std is not None else \
            torch.tensor(-2.).requires_grad_(True)                  # σ in log-space
        self.num_markers: int = num_markers
        self.qma_grid: QMAGrid = qma_grid if qma_grid is not None else QMAGrid(
            torch.tensor([-2.9306, -1.9817, -1.1572, -0.3812,  0.3812,  1.1572,  1.9817,  2.9306]),
            torch.tensor([1.1261e-04, 9.6352e-03, 1.1724e-01, 3.7301e-01, 3.7301e-01, 1.1724e-01, 9.6352e-03, 1.1261e-04])
        )

    def simulate(self) -> torch.Tensor:
        durations = self.treecal.durations()
        log_rates = torch.randn_like(durations) * self.log_rate_log_std.exp() + self.log_rate_mean
        shape = self.num_markers * log_rates.exp() * durations
        self.treecal.distances = torch._standard_gamma(shape) / self.num_markers
        return self.treecal.distances
    
    def log_likelihood(self) -> torch.Tensor:
        E = self.treecal.nedges()
        Q = self.qma_grid.num_nodes
        
        # Transform to log-rate: θ_q = μ + σ · z
        theta_q = self.log_rate_mean + self.log_rate_log_std.exp() * self.qma_grid.nodes  # shape (Q,)
        mu_q = theta_q.exp().view(Q, 1)                        # shape (Q, 1)

        # Compute Gamma shape = μ_q × τ_e × L
        durations = self.treecal.durations()
        shape_qe = mu_q \
            * durations.view(1, E) \
            * self.num_markers

        d = self.treecal.distances.view(1, E).expand(Q, E)
        logprob = D.gamma_log_probability(d * self.num_markers, shape_qe.clamp(1e-8), torch.tensor(1.))  # (Q, E)

        w = self.qma_grid.weights                                # (Q,)
        log_w = torch.log(w).view(Q, 1)                          # (Q,1)
        # Marginalize per edge
        log_lik_e = torch.logsumexp(logprob + log_w, dim=0)
        return log_lik_e - D.barrier_positive(durations)

    def loss(self) -> torch.Tensor:
        return -self.log_likelihood().sum()