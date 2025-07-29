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
A collection of torchscript distribution functions, barrier/penalty functions,
and basic Markov model functions
"""
import torch
import torch.nn.functional as F

@torch.jit.script
def barrier_positive(x: torch.Tensor, strength: float = 1e6):
    """
    Return a positive number growing quickly when x approaches zero or becomes negative.
    """
    return F.softplus(-x, strength) * strength

@torch.jit.script
def gamma_log_probability(x: torch.Tensor, shape:torch.Tensor, rate: torch.Tensor):
    return (
        (shape - 1.0) * torch.log(x)
        - x * rate
        - torch.lgamma(shape)
        + shape * torch.log(rate)
    )

@torch.jit.script
def JC69_probability(t: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Compute p(x) = (K-1)/K * (1 - exp(- (K/(K-1))*x))
    """
    return (K - 1.0) / K * (1.0 - torch.exp(- (K / (K - 1.0)) * t))

@torch.jit.script
def softmax_with_fixed_zero(logits_rest: torch.Tensor) -> torch.Tensor:
    """
    Computes a softmax over logits where the first category is fixed to 0.

    Args:
        logits_rest (Tensor): shape (K-1,), logit values for categories 1..K-1

    Returns:
        Tensor: shape (K,), probability vector with first prob tied to 0-logit
    """
    exp_rest = logits_rest.exp()
    denom = 1.0 + exp_rest.sum()
    probs = torch.cat([
        torch.tensor([1.0], device=logits_rest.device, dtype=logits_rest.dtype),
        exp_rest
    ]) / denom
    return probs