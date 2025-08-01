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
import torch.special

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

# @torch.jit.script
# def gamma_icdf(p: torch.Tensor, shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
#     """
#     TorchScript-compatible Gamma ICDF using gammaincinv.
#     Args:
#         alpha: shape parameter (scalar or broadcastable tensor)
#         beta: rate parameter
#         p: quantile levels ∈ (0, 1)

#     Returns:
#         Tensor of quantile values with shape of broadcast(alpha, beta, p)
#     """
#     return torch.special.gammaincinv(shape, p) / rate

from torch.special import gammainc

@torch.jit.script
def gamma_icdf(p: torch.Tensor, shape: torch.Tensor, rate: torch.Tensor, n_steps: int = 20):
    # x = shape / rate
    x = torch.max(torch.tensor(1e-8, device=p.device), (p * shape)**(1 / shape)) / rate
    for i in range(n_steps):
        cdf = gammainc(shape, rate * x)
        pdf = torch.exp((shape - 1)*torch.log(x) - rate*x - torch.lgamma(shape) + shape*torch.log(rate))
        x = x - (cdf - p) / (rate * pdf)
    return x

@torch.jit.script
def gamma_icdf_hybrid(p: torch.Tensor, shape: torch.Tensor, rate: torch.Tensor, 
                      n_bisect: int=10, n_newton: int=10):
    # Initial bracket: [low, high]
    low = torch.zeros_like(p)
    high = 10 * shape / rate + 10.0  # loose upper bound
    for _ in range(n_bisect):
        mid = (low + high) / 2
        cdf = gammainc(shape, rate * mid)
        high = torch.where(cdf > p, mid, high)
        low = torch.where(cdf <= p, mid, low)
    x = (low + high) / 2

    # Newton refinement
    for _ in range(n_newton):
        cdf = gammainc(shape, rate * x)
        pdf = torch.exp((shape - 1)*torch.log(x) - rate*x - torch.lgamma(shape) + shape*torch.log(rate))
        update = (cdf - p) / (rate * pdf)
        x = x - update
    return x


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