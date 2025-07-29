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
from typing import Optional, Tuple
from . import distribution as D
from .edgelist import TreeTimeCalibrator

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