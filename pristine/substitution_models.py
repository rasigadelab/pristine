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
Substitution Models for Phylogenetic Inference (PRISTINE Framework)
====================================================================

This module defines a family of continuous-time Markov models used to describe
substitution processes in molecular evolution. Each model constructs an instantaneous
rate matrix (Q) normalized to unit average substitution rate, used to compute transition
probabilities along the branches of a phylogenetic tree.

All models follow the general form:

    Q_ij = π_j × R_ij    for i ≠ j
    Q_ii = -∑_{j≠i} Q_ij

Where:
- R is a symmetric exchangeability matrix.
- π is the stationary distribution.
- Q is scaled so that the average rate under π is 1.

Core conventions:
-----------------
• Models may have fixed or learnable stationary distributions, typically parameterized
  with logits (first component fixed for identifiability).
• Exchangeability rates are either fixed or parameterized using log-rates with the first
  entry fixed to 1 to resolve scaling ambiguity.
• Matrix exponentials exp(Qt) are computed in batch using stable torch routines.
• All models expose a unified interface compatible with PRISTINE optimization.

Shared utilities:
-----------------
• `_make_normalized_Q(R, pi)` – builds and normalizes Q.
• `_compute_batch_matrix_exp(Q, durations)` – computes transition matrices exp(Qt).

Implemented DNA substitution models:
------------------------------------

• `JC69Model`
    - Jukes-Cantor model (1969)
    - Uniform stationary distribution
    - All off-diagonal rates equal and fixed
    - 0 free parameters

• `K80Model`
    - Kimura 2-parameter model (1980)
    - Uniform stationary distribution
    - Distinct rate for transitions (log-kappa), fixed transversion rate
    - 1 free parameter

• `HKYModel`
    - Hasegawa-Kishino-Yano model (1985)
    - Arbitrary stationary distribution via stationary logits
    - Distinct transition rate (log-kappa), fixed transversion rate
    - 1 free parameter

• `TN93Model`
    - Tamura-Nei model (1993)
    - Arbitrary stationary distribution via stationary logits
    - Two distinct transition rates (A↔G and C↔T), fixed transversion rate
    - 2 free parameters

• `GTRModel`
    - General Time-Reversible model
    - Arbitrary stationary distribution via stationary logits
    - Fully parameterized symmetric rate matrix (first rate fixed to 1)
    - (K·(K–1)/2 – 1) free parameters for K states

Each model provides the following interface:
--------------------------------------------
• `stationary_dist()` → returns π
• `rate_matrix_stationary_dist()` → returns (Q, π)
• `compute_batch_matrix_exp(durations)` → returns [exp(Q·t) for t in durations]
• `num_free_rates()` → number of trainable exchangeability parameters

These models are TorchScript-compatible and designed for differentiable maximum likelihood
or Bayesian inference within the PRISTINE framework.
"""


import torch
from typing import Tuple, Optional, Any
from . import distribution as D

@torch.jit.script
def _make_normalized_Q(
    R: torch.Tensor,
    pi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct normalized rate matrix Q from symmetric exchangeability matrix R and stationary dist pi.

    Args:
        R: symmetric matrix (K, K) with zeros on diagonal
        pi: stationary distribution (K,)

    Returns:
        Q: normalized rate matrix with unit average rate
        pi: returned unchanged
    """
    Q = pi.unsqueeze(0) * R  # Q_ij = pi_j * R_ij
    Q = Q - torch.diag_embed(Q.sum(dim=1))  # set diagonal to -row sums (no in-place op)
    Q = Q / (-torch.sum(pi * torch.diagonal(Q)))  # normalize to avg rate 1
    return Q, pi

@torch.jit.script
def _compute_batch_matrix_exp(Q: torch.Tensor, durations: torch.Tensor)->torch.Tensor:
    """
    Shared function to compute exp(Qt) for a batch of durations.

    Args:
        Q: substitution matrix of shape (K, K)
        durations: tensor of shape (E,) with branch durations

    Returns:
        P: tensor of shape (E, K, K) with transition matrices
    """    
    return torch.matrix_exp(Q.unsqueeze(0) * durations.view(-1, 1, 1))   


@torch.jit.script
class GTRModel:
    """
    Generalized Time-Reversible (GTR) model with fixed scale:

    - Free parameters: log exchange rates (first one fixed to 1) and stationary logits (first state fixed).
    - Constructs symmetric rate matrix R with upper triangle parametrized.
    - Returns normalized Q matrix with unit average substitution rate.
    """
    def __init__(self, K: int,
                 free_rates_log: Optional[torch.Tensor] = None,
                 stationary_logits: Optional[torch.Tensor] = None):
        self.K = K
        self.free_rates_log = (
            free_rates_log if free_rates_log is not None
            else torch.zeros(self.num_free_rates()).requires_grad_(True)
        )
        self.stationary_logits = (
            stationary_logits if stationary_logits is not None
            else torch.zeros(K - 1).requires_grad_(True)
        )

    def num_free_rates(self) -> int:
        """
        Number of free (log) rate parameters
        """
        K = self.K
        return K * (K - 1) // 2 - 1

    def stationary_dist(self) -> torch.Tensor:
        return D.softmax_with_fixed_zero(self.stationary_logits)

    def rate_matrix_stationary_dist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.stationary_dist()
        rates = torch.cat([torch.ones(1, device=self.free_rates_log.device), self.free_rates_log.exp()])
        idx = torch.triu_indices(self.K, self.K, offset=1)
        R = torch.zeros((self.K, self.K), dtype=rates.dtype, device=rates.device)
        R[idx[0], idx[1]] = rates
        R[idx[1], idx[0]] = rates
        return _make_normalized_Q(R, pi)

    def compute_batch_matrix_exp(self, durations: torch.Tensor) -> torch.Tensor:
        Q, _ = self.rate_matrix_stationary_dist()
        return _compute_batch_matrix_exp(Q, durations)

class HKYModel:
    """
    HKY model (Hasegawa-Kishino-Yano):

    - Fixed first transversion rate = 1 (e.g., A↔C).
    - Kappa (log-transformed) controls transition/transversion bias.
    - Stationary distribution is parametrized via logits with fixed first state.
    - Returns normalized Q with unit average rate.
    """
    def __init__(self,
                 kappa_log: Optional[torch.Tensor] = None,
                 stationary_logits: Optional[torch.Tensor] = None):
        self.K = 4
        self.kappa_log = kappa_log if kappa_log is not None else torch.tensor(0.0, requires_grad=True)
        self.stationary_logits = (
            stationary_logits if stationary_logits is not None
            else torch.zeros(3).requires_grad_(True)
        )

    def num_free_rates(self) -> int:
        """
        Number of free (log) rate parameters
        """
        return 1

    def stationary_dist(self) -> torch.Tensor:
        return D.softmax_with_fixed_zero(self.stationary_logits)

    def rate_matrix_stationary_dist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.stationary_dist()
        kappa = self.kappa_log.exp()

        R = torch.ones((4, 4), device=kappa.device)
        R.fill_diagonal_(0.0)
        R[0, 2] = R[2, 0] = kappa  # A<->G
        R[1, 3] = R[3, 1] = kappa  # C<->T
        R[0, 1] = R[1, 0] = 1.0    # A<->C (fixed to 1)

        return _make_normalized_Q(R, pi)

    def compute_batch_matrix_exp(self, durations: torch.Tensor) -> torch.Tensor:
        Q, _ = self.rate_matrix_stationary_dist()
        return _compute_batch_matrix_exp(Q, durations)

class K80Model:
    """
    Kimura 2-parameter (K80) model:

    - Uniform stationary distribution
    - Two rates: transition rate (kappa, log-transformed) and fixed transversion rate (1)
    """
    def __init__(self, kappa_log: Optional[torch.Tensor] = None):
        self.K = 4
        self.kappa_log = kappa_log if kappa_log is not None else torch.tensor(0.0, requires_grad=True)

    def num_free_rates(self) -> int:
        return 1

    def stationary_dist(self) -> torch.Tensor:
        return torch.full((self.K,), 0.25)

    def rate_matrix_stationary_dist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kappa = self.kappa_log.exp()
        R = torch.ones((4, 4), device=kappa.device)
        R.fill_diagonal_(0.0)
        R[0, 2] = R[2, 0] = kappa  # A<->G
        R[1, 3] = R[3, 1] = kappa  # C<->T
        return _make_normalized_Q(R, self.stationary_dist())

    def compute_batch_matrix_exp(self, durations: torch.Tensor) -> torch.Tensor:
        Q, _ = self.rate_matrix_stationary_dist()
        return _compute_batch_matrix_exp(Q, durations)

class TN93Model:
    """
    Tamura-Nei 1993 (TN93) model:

    - Unequal base frequencies
    - Two distinct transition rates (A<->G, C<->T)
    - One transversion rate (fixed to 1)
    """
    def __init__(self,
                 log_kappa1: Optional[torch.Tensor] = None,
                 log_kappa2: Optional[torch.Tensor] = None,
                 stationary_logits: Optional[torch.Tensor] = None):
        self.K = 4
        self.log_kappa1 = log_kappa1 or torch.tensor(0.0, requires_grad=True)  # A<->G
        self.log_kappa2 = log_kappa2 or torch.tensor(0.0, requires_grad=True)  # C<->T
        self.stationary_logits = stationary_logits or torch.zeros(3).requires_grad_(True)

    def num_free_rates(self) -> int:
        return 2

    def stationary_dist(self) -> torch.Tensor:
        return D.softmax_with_fixed_zero(self.stationary_logits)

    def rate_matrix_stationary_dist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.stationary_dist()
        k1 = self.log_kappa1.exp()
        k2 = self.log_kappa2.exp()

        R = torch.ones((4, 4), device=pi.device)
        R.fill_diagonal_(0.0)
        R[0, 2] = R[2, 0] = k1  # A<->G
        R[1, 3] = R[3, 1] = k2  # C<->T
        R[0, 1] = R[1, 0] = 1.0  # Transversions fixed
        return _make_normalized_Q(R, pi)

    def compute_batch_matrix_exp(self, durations: torch.Tensor) -> torch.Tensor:
        Q, _ = self.rate_matrix_stationary_dist()
        return _compute_batch_matrix_exp(Q, durations)

class JC69Model:
    """
    Jukes-Cantor (JC69) substitution model:

    - All off-diagonal rates equal (fixed to 1), uniform stationary distribution.
    - No free parameters.
    - Returns normalized Q matrix with unit average substitution rate.
    """
    def __init__(self, K: int = 4):
        self.K = K

    def num_free_rates(self) -> int:
        """
        Number of free (log) rate parameters
        """
        return 0
    
    def stationary_dist(self) -> torch.Tensor:
        return torch.full((self.K,), 1.0 / self.K)

    def rate_matrix_stationary_dist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.stationary_dist()
        R = torch.ones((self.K, self.K))
        R.fill_diagonal_(0.0)
        return _make_normalized_Q(R, pi)

    def compute_batch_matrix_exp(self, durations: torch.Tensor) -> torch.Tensor:
        Q, _ = self.rate_matrix_stationary_dist()
        return _compute_batch_matrix_exp(Q, durations)
