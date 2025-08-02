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
r"""
qma.py: Quantile Midpoint Approximation (QMA) utilities for PRISTINE
---------------------------------------------------------------------

This module provides tools to construct and hold quadrature grids for
approximately marginalizing out latent continuous parameters in hierarchical
models (e.g., per-branch substitution rates in relaxed-clock phylogenetics).

Quantile Midpoint Approximation (QMA) replaces intractable integrals of the form
    ∫ f(θ) p(θ) dθ
with a finite weighted sum
    ∑_{i=1}^Q w_i f(θ_i),
where {θ_i} are representative points (quantiles or nodes) and {w_i} are
appropriate quadrature weights.  By choosing different schemes for node
placement and weight computation, QMA balances accuracy, tail coverage,
and computational cost.

Classes
-------

QMAGrid (torch.jit.ScriptModule)
    A TorchScript-compatible holder for a fixed grid of evaluation points
    and their weights:
      - `nodes`: Tensor of shape (Q,) containing θ_i (quantiles or Hermite roots).
      - `weights`: Tensor of shape (Q,) containing w_i (sums to 1 for PDF-based
        grids, or ∑w_i=1/√π for Hermite-based Gaussian integrals).
    Method:
      - `get_grid() -> (nodes, weights)`

QMAFactory
    A Python factory (non-TorchScript) to generate `QMAGrid` instances via
    three common quadrature schemes:
      - `uniform()`: standard midpoint rule on [0,1], nodes = (k-0.5)/Q,
        weights = Δu = 1/Q.
      - `warped(gamma)`: power-law warp CDF(u)=0.5·(2u)^γ for u≤0.5,
        1-0.5·(2(1-u))^γ for u>0.5; nodes = warp(midpoints),
        weights = warp(u_{k})-warp(u_{k-1}).
      - `hermite()`: Gauss-Hermite quadrature for ∫ϕ(θ;0,1)f(θ)dθ,
        nodes = Hermite roots x_i, weights = w_i/√π.

Use cases
---------
- **Site-rate integration** in phylogenetics (Yang 1994), where per-site rates
  follow a Gamma prior.
- **Branch-rate marginalization** under log-normal priors (QMA for relaxed clocks).
- **Hierarchical priors** in any model where latent continuous variables
  must be integrated approximately but efficiently.

Rationale
---------
- QMA provides a deterministic, differentiable surrogate for marginalization,
  suitable for gradient-based optimization.
- Choice of grid governs bias-variance trade-off:
  uniform for simplicity, warped for tail emphasis, Hermite for Gaussian priors.
- Separation of grid construction (QMAFactory) from storage and use
  (QMAGrid) enables TorchScript compatibility and reuse.

Example
-------
>>> factory = QMAFactory(num_nodes=8)
>>> qma = factory.warped(gamma=0.5)
>>> nodes, weights = qma.get_grid()
>>> # In a model's likelihood:
>>> #   loglik = torch.logsumexp(logp_theta + torch.log(weights)[:,None], dim=0)

:contentReference[oaicite:2]{index=2}
"""

import math
from typing import Tuple

import numpy as np
import torch
# -------------------------------------------------------------------
# TorchScript-compatible holder of QMA nodes & weights
# -------------------------------------------------------------------
@torch.jit.script
class QMAGrid:

    def __init__(self, nodes: torch.Tensor, weights: torch.Tensor):
        """
        Holds fixed quadrature nodes and weights for QMA.

        Args:
            nodes   (Tensor[Q]): evaluation points, either quantiles or Hermite roots
            weights (Tensor[Q]): nonnegative weights summing to 1 (for probabilities) or ∑w_i=1/√π (for Hermite)
        """
        self.num_nodes = nodes.size(0)
        self.nodes: torch.Tensor = nodes
        self.weights: torch.Tensor = weights

    def get_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (nodes, weights)
        """
        return self.nodes, self.weights

# -------------------------------------------------------------------
# Non-TorchScript factory for building QMA grids
# -------------------------------------------------------------------
class QMAFactory:
    """
    Build QMAWeights using one of three schemes:
      - 'uniform' : equal-width bins in [0,1]
      - 'warped'   : power-law warped bins in [0,1] to emphasize tails
      - 'hermite' : Gauss-Hermite quadrature for Normal-prior integrals
    """
    def __init__(self, num_nodes: int = 8):
        self.num_nodes = num_nodes

    def uniform(self) -> QMAGrid:
        # Partition [0,1] into Q equal bins, use midpoints and bin-widths
        Q = self.num_nodes
        edges = torch.linspace(0.0, 1.0, Q + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        weights = edges[1:] - edges[:-1]
        quantiles = torch.special.ndtri(mids)  # inverse standard normal CDF
        return QMAGrid(quantiles, weights)

    def warped(self, gamma: float) -> QMAGrid:
        """
        Power-law warp quadrature for Normal(0,1) integrals:
            warp(u) = 0.5*(2u)^γ     for u <= 0.5
                    = 1 - 0.5*(2(1-u))^γ  for u > 0.5
        Nodes are warp(midpoints), weights are the differences in warp at bin edges.
        """
        Q = self.num_nodes
        u_edges = torch.linspace(0.0, 1.0, Q + 1)

        def warp(u: torch.Tensor) -> torch.Tensor:
            left = 0.5 * (2.0 * u).pow(gamma)
            right = 1.0 - 0.5 * (2.0 * (1.0 - u)).pow(gamma)
            return torch.where(u <= 0.5, left, right)

        warped_edges = warp(u_edges)
        mids = (warped_edges[:-1] + warped_edges[1:]) / 2
        weights = warped_edges[1:] - warped_edges[:-1]
        quantiles = torch.special.ndtri(mids)
        return QMAGrid(quantiles, weights)

    def hermite(self) -> QMAGrid:
        """
        Gauss-Hermite quadrature for Normal(0,1) integrals:
          ∫ e^{-x^2} g(x) dx ≈ ∑ w_i g(x_i)
        We absorb the 1/√π factor into weights for ∫ φ(θ;μ,σ^2) f(θ) dθ.
        """
        # compute with NumPy, then convert
        x_np, w_np = np.polynomial.hermite.hermgauss(self.num_nodes)
        nodes = torch.from_numpy(x_np.astype(np.float32))
        weights = torch.from_numpy((w_np / math.sqrt(math.pi)).astype(np.float32))
        return QMAGrid(nodes, weights)