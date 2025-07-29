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
This module implements the Felsenstein pruning algorithm (FPA) using 
TorchScript for efficient, differentiable computation of phylogenetic 
likelihoods on trees.

At its core, the FelsensteinPruningAlgorithm class computes the likelihood 
of observing genetic sequences at the tips of a tree, given a substitution 
model and a tree with branch lengths. The algorithm works by propagating 
conditional likelihoods from the leaves toward the root (post-order traversal), 
combining evidence from child nodes using transition probabilities derived 
from a rate matrix.

Conceptually, the method uses dynamic programming to avoid redundant 
computations and exploits the independence across sites in the sequence 
alignment. It maintains per-node, per-site conditional likelihoods across 
states, and updates them recursively from child to parent by computing 
probability-weighted transitions.

Numerical scaling is applied to avoid underflow. The final likelihood is 
computed at the root by marginalizing over all root states weighted by 
their stationary frequencies.

The optional logspace implementation provides a numerically stable 
alternative by performing all operations in logarithmic form, at the 
cost of additional computational overhead.

The helper function get_postorder_edge_list organizes tree edges into 
recursion levels such that edges in earlier levels never depend on 
edges in later ones. This enables level-by-level vectorized traversal 
of the tree.

This implementation allows for exact and scalable likelihood calculations 
for complex models in a differentiable, GPU-compatible framework.
"""

from __future__ import annotations # For forward references within class definition
import sys;import os;sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from typing import List, Any, Optional

from .edgelist import TreeTimeCalibrator
from .gtr import GeneralizedTimeReversibleModel
from .sequence import CollapsedConditionalLikelihood
from .molecularclock import ConstantClock

@torch.jit.script
def get_postorder_edge_list(treecal: TreeTimeCalibrator):
    """
    Traverse the tree from tips to root (postorder) and assign edges to level groups so that
    an edge in group i<j does not contain any ancestor of any edge in group j. This is a costly
    function that should not be called repeatedly.
    """
    # N = torch_edges.shape[0] + 1 # No. of nodes and tips in tree = no. of edges + the root
    N = treecal.ntips() + treecal.nnodes()
    E = treecal.nedges()
    depth = torch.zeros(N, dtype=torch.long)
    for i in range(E):
        parent = int(treecal.parents[i].item())
        child = int(treecal.children[i].item())
        depth[child] = depth[parent] + 1

    tree_recursion_structure: List[List[torch.Tensor]] = []

    max_depth = int(depth.max().item())
    for d in range(max_depth, 0, -1):

        # Select edges whose child is at depth d (so the parent is at depth d-1).
        mask = (depth[treecal.children] == d)
        if mask.sum() == 0:
            continue
        edge_idx = torch.nonzero(mask).squeeze(1)
        # Get the parent and child indices for these edges.
        tree_recursion_structure.append([edge_idx, treecal.parents[edge_idx], treecal.children[edge_idx]])

    return tree_recursion_structure

@torch.jit.script
class FelsensteinPruningAlgorithm:
    """
    TorchScript class for FPA.
    """
    def __init__(self,
                 substitution_model: GeneralizedTimeReversibleModel, 
                 markers: CollapsedConditionalLikelihood,
                 treecal: TreeTimeCalibrator,
                 clock: Optional[ConstantClock] = None):
        self.eps = 1e-16
        self.clock: ConstantClock = clock if clock is not None else ConstantClock()
        self.substitution_model: GeneralizedTimeReversibleModel = substitution_model
        self.markers: CollapsedConditionalLikelihood = markers
        self.ancestor_states: torch.Tensor = markers.unique_patterns.clone().detach()
        self.treecal: TreeTimeCalibrator = treecal
        # Recursion object used for parallel FPA
        self.recursion_structure: List[List[torch.Tensor]] = get_postorder_edge_list(treecal)

    def log_likelihood(self) -> torch.Tensor:                      # ── main entry
        """
        Probability‑space Felsenstein pruning.

        * `node_probs[n, l, k]`  : normalised CL for node n, site l, state k
        * `log_scaling[n, l]`    : accumulated log‑scale for node n, site l
        The true (unnormalised) CL is  
            L_true = exp(log_scaling) * node_probs
        """
        # ------------------------------------------------------------------
        N, L, K = self.markers.unique_patterns.shape
        # eps      = 1e-16                                            # numerical safety
        device   = self.markers.unique_patterns.device
        # ------------------------------------------------------------------
        # Transition matrices for every edge  -----------------------------
        durations = self.treecal.durations().clamp_min_(0.)
        scaled_durations = durations * self.clock.rate()
        P_batch   = self.substitution_model.compute_batch_matrix_exp(scaled_durations)  # (E,K,K)

        # ------------------------------------------------------------------
        # Initialise per‑node conditional likelihoods ---------------------
        # • tips  : one‑hot (already in unique_patterns)
        # • internals: uniform (1/K) so that Σ_k CL = 1
        node_probs = self.markers.unique_patterns.clone().detach()              # (N,L,K)
        internal_mask = node_probs.sum(dim=-1, keepdim=True).eq(0)              # (N,L,1)
        node_probs[internal_mask.expand_as(node_probs)] = 1.0 / K

        # Per‑site log‑scaling accumulators
        log_scaling = torch.zeros((N, L), dtype=torch.float32, device=device)   # (N,L)

        # Temporary tensor for log message accumulation (re‑allocated each level)
        # (allocating per level avoids an O(N·L·K) zero‑fill each iteration)
        acc_log = torch.zeros((N, L, K), dtype=torch.float32, device=device)
        # ------------------------------------------------------------------
        for edge_idx, parents, children in self.recursion_structure:            # post‑order

            # ----- 1. message from each child up its edge ------------------
            P_edges     = P_batch[edge_idx]                                     # (E,K,K)
            child_prob  = node_probs[children]                                  # (E,L,K)
            msg_base    = torch.bmm(child_prob, P_edges.transpose(1, 2))        # (E,L,K)

            # True (unnormalised) message = msg_base * exp(log_scaling_child)
            log_msg = msg_base.clamp_min(self.eps).log() +                           \
                    log_scaling[children].unsqueeze(-1)                       # (E,L,K)

            # ----- 2. accumulate log‑messages per parent -------------------
            #    log(product over children) = Σ log(message)
            acc_log.index_add_(0, parents, log_msg)                             # add logs

            unique_parents = torch.unique(parents)                              # 1…M_parents

            # ----- 3. update each parent’s conditional likelihood -----------
            # parent_log   = (node_probs[unique_parents] + eps).log()             # (M,L,K)
            parent_log   = node_probs[unique_parents].log()             # (M,L,K)
            log_val      = parent_log + acc_log[unique_parents]                 # combine
            norm         = torch.logsumexp(log_val, dim=-1, keepdim=True)       # (M,L,1)

            # store normalised CL and absorb scale factor --------------------
            node_probs[unique_parents]  = torch.exp(log_val - norm)             # (M,L,K)
            log_scaling[unique_parents] += norm.squeeze(-1)                     # (M,L)

        # ------------------------------------------------------------------
        # Persist ancestor‑state posteriors for downstream use --------------
        self.ancestor_states[self.treecal.node_indices] = \
            node_probs[self.treecal.node_indices].detach()

        # ------------------------------------------------------------------
        # Likelihood at the root -------------------------------------------
        root_true   = node_probs[0] * log_scaling[0].exp().unsqueeze(-1)        # (L,K)
        pi          = self.substitution_model.stationary_dist()                # (K,)
        site_like   = (root_true * pi).sum(dim=-1).clamp_min(self.eps)              # (L,)
        log_L_sites = site_like.log() + log_scaling[0]                         # (L,)

        # ------------------------------------------------------------------
        # Pattern weights  --------------------------------------------------
        log_likelihood = (log_L_sites * self.markers.pattern_counts).sum()
        return log_likelihood
        
    def loss(self) -> torch.Tensor:
        return -self.log_likelihood().sum()