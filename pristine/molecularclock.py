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
from typing import Optional
from . import distribution as D


#########################################################################
# CONDITIONAL CLOCK MODEL
#########################################################################

@torch.jit.script
class ConditionalErrorClock:
    def __init__(
        self,
        log_rate: torch.Tensor,
        num_states: torch.Tensor,
        sequence_length: torch.Tensor
    ):
        """
        States and length are used as float in computations so we convert them
        once during initialization
        """
        self.num_states: torch.Tensor = num_states
        self.log_rate: torch.Tensor = log_rate
        self.sequence_length: torch.Tensor = sequence_length

    def rate(self)->torch.Tensor:
        return self.log_rate.exp()

    def log_likelihood(self, durations: torch.Tensor, distances: torch.Tensor)->torch.Tensor:
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
        scaled_durations = durations * self.log_rate.exp()
        p_durations = D.JC69_probability(scaled_durations, K)
        p_distances = D.JC69_probability(distances, K)

        # Compute the log coefficient using lgamma (log Gamma)
        # Note: torch.lgamma(x) = log(Γ(x))
        log_coeff = torch.lgamma(L + 1.0) - \
                    torch.lgamma(L * p_distances + 1) - \
                    torch.lgamma(L * (1 - p_distances) + 1)
        
        # Compute the log likelihood (each term is computed elementwise)
        loglik = log_coeff + L * p_distances * torch.log(p_durations) + L * (1.0 - p_distances) * torch.log(1.0 - p_durations)
        
        return loglik

def new_conditional_error_clock(num_states: float, sequence_length: float)->ConditionalErrorClock:
    torch_log_rate = torch.tensor(0.0, requires_grad=True)
    return ConditionalErrorClock(
        log_rate=torch_log_rate,
        num_states=torch.tensor(num_states), 
        sequence_length=torch.tensor(sequence_length)
    )

#########################################################################
# cARC MODEL
#########################################################################

@torch.jit.script
class ContinuousAdditiveRelaxedClock:
    def __init__(
        self,
        log_rate: torch.Tensor,
        sequence_length: Optional[torch.Tensor] = None,
        dispersion: Optional[torch.Tensor] = None
    ):
        self.log_rate = log_rate
        self.sequence_length = sequence_length if sequence_length is not None else torch.tensor(1.0)
        self.dispersion = dispersion if dispersion is not None else torch.tensor(0.0)

    def rate(self)->torch.Tensor:
        return self.log_rate.exp()

    def _gamma_rate(self) -> torch.Tensor:
        return 1.0 / (1.0 + self.dispersion)

    def simulate(self, durations: torch.Tensor) -> torch.Tensor:
        """
        Simulate genetic distances under the continuous additive relaxed clock model.
        """
        rate = self._gamma_rate()
        shape = self.rate() * self.sequence_length * durations * rate
        return torch._standard_gamma(shape) * rate / self.sequence_length

    def log_likelihood(self, durations: torch.Tensor, distances: torch.Tensor)->torch.Tensor:
        """
        Log-likelihood of durations and distances under the continuous additive relaxed clock model.
        """
        rate = self._gamma_rate()
        shape = self.rate() * self.sequence_length * durations * rate        
        scaled_distances = distances * self.sequence_length

        return (
            D.gamma_log_probability(scaled_distances, shape, rate) 
            - D.barrier_positive(shape)
        )

def new_continuous_additive_relaxed_clock(sequence_length: int):
    torch_log_rate = torch.tensor(0.0, requires_grad=True)
    torch_dispersion = torch.tensor(0.0, requires_grad=True)
    torch_sequence_length = torch.tensor(sequence_length)
    return ContinuousAdditiveRelaxedClock(log_rate=torch_log_rate, sequence_length=torch_sequence_length, dispersion=torch_dispersion)