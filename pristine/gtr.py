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
GeneralizedTimeReversibleModel defines a flexible and parameter-efficient
class of continuous-time Markov models used in phylogenetics. It represents
the instantaneous rate matrix (Q) governing how characters (such as DNA or
amino acids) evolve over time.

This model captures two core ideas:

1. Stationary distribution: the long-term frequency of each character, 
   parameterized via unconstrained logits and converted to probabilities 
   with a custom softmax that pins the first category as reference.

2. Reversible exchange rates: off-diagonal rates between states i and j 
   are symmetric and modeled using a minimal set of parameters. These are 
   exponentiated and symmetrically placed in the matrix.

The full rate matrix Q is computed as Q[i,j] = π[j] * R[i,j] for i ≠ j,
where π is the stationary distribution and R is the symmetric rate matrix.
Diagonal entries Q[i,i] are defined to ensure rows sum to zero.

The model supports computing:

- The stationary distribution π
- The average substitution rate (μ)
- A batch of transition matrices exp(Q·t) for use in likelihood calculations

This class is fully differentiable, enabling gradient-based inference of
model parameters directly from observed data.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from . import distribution as D

@torch.jit.script
class GeneralizedTimeReversibleModel:
    """
    A clock-free GTR model for use with raw durations. Do not use with
    durations rescaled by a free clock parameters.
    """
    def __init__(self,
                 K: int,
                 rates_log: Optional[torch.Tensor] = None, 
                 stationary_logits: Optional[torch.Tensor] = None, 
                 ):
        self.K: int = K
        self.rates_log: torch.Tensor = (
            rates_log if rates_log is not None else torch.randn(int(K*(K-1) / 2))
        )
        self.stationary_logits: torch.Tensor = (
            stationary_logits if stationary_logits is not None else torch.randn(K-1)
        )

    def stationary_dist(self)->torch.Tensor:
        # return F.softmax(self.stationary_logits, dim=0)
        return D.softmax_with_fixed_zero(self.stationary_logits)
    
    def rate_matrix_stationary_dist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given steady-state logits and off-diagonal log–rates, compute the rate matrix Q.
        
        Q is computed as follows:
        1. Compute the stationary distribution π = softmax(steady_logits).
        2. Build a symmetric R matrix from log_rates (only for i < j).
        3. For i ≠ j, set Q[i,j] = π[j] * R[i,j] and Q[i,i] = -sum_{j≠i} Q[i,j].
        
        Returns:
            Q: Tensor of shape (K, K)
            pi: stationary distribution (tensor of shape (K,))
        """
        # pi = F.softmax(self.stationary_logits, dim=0)
        pi = self.stationary_dist()
        rates_upper = torch.exp(self.rates_log)

        indices = torch.triu_indices(self.K, self.K, offset=1)
        R = torch.zeros((self.K, self.K), dtype=pi.dtype)
        R[indices[0], indices[1]] = rates_upper
        R[indices[1], indices[0]] = rates_upper

        Q = pi.unsqueeze(0) * R
        Q.fill_diagonal_(0.0)
        Q.diagonal().copy_(-Q.sum(dim=1))
        return Q, pi

    def average_rate(self)->torch.Tensor:
        """
        Compute the average transition rate
        """
        Q, pi = self.rate_matrix_stationary_dist()
        mu = -torch.sum(pi * torch.diag(Q))
        return mu

    def compute_batch_matrix_exp(self, durations: torch.Tensor) -> torch.Tensor:
        """
        Compute a batch of matrix exponentials using torch.matrix_exp, which is 
        more robust than eigendecomposition in the presence of nearly degenerate eigenvalues.
        
        Args:            
            durations: Tensor of shape (E,), waiting times or durations.
        
        Returns:
            A tensor of shape (E, K, K) where the i-th slice is exp(Q * durations[i]).
        """
        Q, _ = self.rate_matrix_stationary_dist()
        E = durations.shape[0]
        # Create a batch of matrices Q*t for each edge.
        Q_batch = Q.unsqueeze(0).expand(E, -1, -1) * durations.view(E, 1, 1)
        # Use torch.matrix_exp which is based on scaling and squaring.
        P_batch = torch.matrix_exp(Q_batch)
        return P_batch
    
