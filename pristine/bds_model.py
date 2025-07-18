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
BirthDeathSamplingNodeData models the behavior of a lineage through time.
It assigns each node three rates: birth (speciation), death (extinction),
and sampling (observation). These parameters govern the stochastic events
that shape the branching structure of the phylogenetic tree.

BirthDeathSamplingSimulator grows a binary tree forward in time using a
stochastic birth-death-sampling process. Each lineage can give rise to
two offspring (birth), disappear (death), or be sampled (marked as an
observed tip). Simulation continues until a fixed number of samples
(tips) are collected, then prunes unresolved branches.

stadler_q, stadler_q_matrix, and stadler_p0 implement analytical
functions for evaluating birth-death-sampling probabilities under a
time-homogeneous model. These are used to compute likelihoods of trees
given continuous-time parameters and node ages.

BirthDeathSamplingTreeWise evaluates the log-likelihood of a dated tree
under a time-homogeneous birth-death-sampling process. It uses analytical
expressions to compute the probability of the full tree as a product
of q-ratios across branches and rate-dependent contributions at tips
and internal nodes.

StateDependentBirthDeathSampling extends the above model to allow rates
to vary according to hidden discrete states at each node. It computes a
weighted likelihood by summing over the probabilities of each state,
modulated by q-ratios and per-event rates. This is typically used in
conjunction with ancestor state inference, such as from a Felsenstein
pruning algorithm.
"""

#########################################################################
# BIRTH-DEATH-SAMPLING MODEL SIMULATION
#########################################################################

from .edgelist import TreeTimeCalibrator
from .binarytree import BinaryTreeNode
from typing import Any, Tuple, Optional
import random
import heapq

class BirthDeathSamplingNodeData:
    """
    Base class for BinaryTreeNode.data used in birth/death/sampling models.

    Fields:
        birth_rate: the rate of lineage division
        death_rate: the rate of lineage extinction
        sampling_rate: the rate of lineage detection
    """
    def __init__(self, birth_rate: float = 0.0, death_rate: float = 0.0, sampling_rate: float = 0.0, **kwargs):
        self.birth_rate: float=birth_rate
        self.death_rate: float=death_rate
        self.sampling_rate: float=sampling_rate
        super().__init__(**kwargs)

    def total_rate(self)->float:
        return (
            self.birth_rate +
            self.death_rate +
            self.sampling_rate
        )

class BirthDeathSamplingNode(BinaryTreeNode):
    def __init__(self, id: int = 0, time: float = 0, parent: Optional[BinaryTreeNode] = None):
        super().__init__(id, time, parent)

class BirthDeathSamplingSimulator:
    """
    Simulate growth of a phylogenetic tree forward in time
    according to a piecewise-constant birth-death-sampling model.
    """
    def __init__(self,
                 root: BinaryTreeNode,
                 model: Any):
        self.root: BinaryTreeNode = root
        self.model = model

    def simulate(
            self,
            n: int
        ) -> BinaryTreeNode:

        node_counter = 1
        event_queue = []
        active_lineages = {}
        n_sampled = 0

        # Initialize root node
        self.model.visit(self.root)
        active_lineages[self.root.id] = self.root

        def schedule_event(node: BinaryTreeNode, datanode: BinaryTreeNode | None = None):
            """Schedule the next event (birth, death, or sampling) for a node."""
            if datanode is None:
                datanode = node                
            total_rate = datanode.data.total_rate() # type: ignore
            if total_rate == 0:
                return  # No events possible
            wait_time = random.expovariate(total_rate)
            event_time = datanode.time + wait_time
            r = random.uniform(0, total_rate)
            if r < datanode.data.birth_rate:
                event_type = "birth"
            elif r < datanode.data.birth_rate + datanode.data.death_rate:
                event_type = "death"
            else:
                event_type = "sampling"
            heapq.heappush(event_queue, (event_time, node, event_type))

        # Schedule initial event for root
        schedule_event(self.root)

        while n_sampled < n and event_queue:
            event_time, node, event_type = heapq.heappop(event_queue)

            node.time = event_time
            self.model.visit(node)

            if node.id not in active_lineages:
                continue  # Skip if already removed

            if event_type == "birth":
                # Create two child nodes (bifurcation)
                left = BinaryTreeNode(node_counter, node.time, node)
                node.left = left
                node_counter += 1

                right = BinaryTreeNode(node_counter, node.time, node)
                node.right = right
                node_counter += 1

                active_lineages[left.id] = left
                active_lineages[right.id] = right

                schedule_event(left, node)
                schedule_event(right, node)

                node.tip = False
                node.sampled = False

                del active_lineages[node.id]  # Internal node no longer active

            elif event_type == "death":
                node.tip = True
                node.sampled = False
                del active_lineages[node.id]

            elif event_type == "sampling":
                node.tip = True
                node.sampled = True
                del active_lineages[node.id]
                n_sampled += 1


        if n_sampled < n:
            return self.simulate(n)

        # Kill remaining lineages (prune unresolved nodes)
        for remaining_node in list(active_lineages.values()):
            self.root = self.root.drop_tip(remaining_node)

        self.root.reindex()
        return self.root 

#########################################################################
# BIRTH-DEATH-SAMPLING SUBROUTINES
#########################################################################
import torch

@torch.jit.script
def stadler_q(age, birth, death, sampling):
    bdsdiff = birth - death - sampling
    c1 = torch.sqrt(torch.square(bdsdiff) + 4. * birth * sampling)
    c2 = -bdsdiff / c1
    q = 2.*(1.-torch.square(c2)) + torch.exp(-c1 * age) * torch.square(1. - c2) + torch.exp( c1 * age) * torch.square(1. + c2)
    return q

@torch.jit.script
def stadler_q_matrix(age: torch.Tensor, birth: torch.Tensor, death: torch.Tensor, sampling: torch.Tensor) -> torch.Tensor:
    """
    Vectorized q values
    """
    # age: [N], birth, death, sampling: [K]
    # Expand for broadcasting
    age = age.unsqueeze(0)         # [1, N]
    birth = birth.unsqueeze(1)     # [K, 1]
    death = death.unsqueeze(1)     # [K, 1]
    sampling = sampling.unsqueeze(1)  # [K, 1]

    bdsdiff = birth - death - sampling        # [K, 1]
    c1 = torch.sqrt(bdsdiff ** 2 + 4. * birth * sampling)  # [K, 1]
    c2 = -bdsdiff / c1                         # [K, 1]

    exp_neg = torch.exp(-c1 * age)             # [K, N]
    exp_pos = torch.exp(c1 * age)              # [K, N]

    q = (
        2. * (1. - c2**2)                      # [K, 1]
        + exp_neg * (1. - c2)**2               # [K, N]
        + exp_pos * (1. + c2)**2               # [K, N]
    )

    # return q  # shape [K, N]
    return q.transpose(0, 1)  # shape [N, K]

@torch.jit.script
def stadler_p0(age, birth, death, sampling):
    bdsdiff = birth - death - sampling
    c1 = torch.sqrt(torch.square(bdsdiff) + 4. * birth * sampling)
    c2 = -bdsdiff / c1
    c3 = torch.exp(-c1 * age) * (1. - c2)
    p0 = (birth + death + sampling + c1 * (c3 - (1. + c2)) / (c3 + (1. + c2))) / (2. * birth)
    return p0

#########################################################################
# BIRTH-DEATH-SAMPLING MODEL EVALUATION
#########################################################################

@torch.jit.script
class BirthDeathSamplingTreeWise:

    def __init__(self,
                 birth_log: torch.Tensor,
                 death_log: torch.Tensor,
                 sampling_log: torch.Tensor
                 ):
        self.birth_log: torch.Tensor = birth_log
        self.death_log: torch.Tensor = death_log
        self.sampling_log: torch.Tensor = sampling_log

    def birth_death_sampling_rates(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.birth_log.exp(), self.death_log.exp(), self.sampling_log.exp()
    
    def stadler_q(self, ages)->torch.Tensor:
        birth, death, sampling = self.birth_death_sampling_rates()
        bdsdiff = birth - death - sampling
        c1 = torch.sqrt(torch.square(bdsdiff) + 4. * birth * sampling)
        c2 = -bdsdiff / c1
        q = 2.*(1.-torch.square(c2)) + torch.exp(-c1 * ages) * torch.square(1. - c2) + torch.exp( c1 * ages) * torch.square(1. + c2)
        return q
    
    def log_likelihood(self, treecal: TreeTimeCalibrator):
        # Q-ratio terms
        q_terms = self.stadler_q(treecal.ages())
        q_ratios = q_terms[treecal.children] / q_terms[treecal.parents]
        # Treewise-constant BDS term        
        bds_log_likelihood = (
            # Edges terms
              torch.log( q_ratios ).sum() 
            # Node terms (birth events)
            + (treecal.ntips() - 1.) * self.birth_log
            # Tip terms (sampling events)
            + treecal.ntips() * self.sampling_log
            )
        return bds_log_likelihood
    
@torch.jit.script
class StateDependentBirthDeathSampling:
    """
    BDS parameters are constructed along the states
    """
    def __init__(self,
                 num_states: int,
                 birth_log: Optional[torch.Tensor] = None,
                 death_log: Optional[torch.Tensor] = None,
                 sampling_log: Optional[torch.Tensor] = None,
                 ):
        self.num_states: int = num_states
        self.birth_log: torch.Tensor = birth_log if birth_log is not None else torch.zeros(num_states)
        self.death_log: torch.Tensor = death_log if death_log is not None else torch.zeros(num_states)
        self.sampling_log: torch.Tensor = sampling_log if sampling_log is not None else torch.zeros(num_states)
    
    def birth(self)->torch.Tensor:
        return self.birth_log.exp().expand(self.num_states)

    def death(self)->torch.Tensor:
        return self.death_log.exp().expand(self.num_states)

    def sampling(self)->torch.Tensor:
        return self.sampling_log.exp().expand(self.num_states)

    def log_likelihood(self, treecal: TreeTimeCalibrator, ancestor_states: torch.Tensor)->torch.Tensor:

        birth = self.birth_log.exp().expand(self.num_states)
        death = self.death_log.exp().expand(self.num_states)
        sampling = self.sampling_log.exp().expand(self.num_states)

        ages = treecal.ages()
        qmat = stadler_q_matrix(ages, birth, death, sampling) # shape (N, K)
        """
        q-ratio at each edge
        """
        q_ratio = qmat[treecal.children] / qmat[treecal.parents]
        """
        Probability at each parent, tensor [E, K]. Beware that ancestor markers has shape [N L K]
        with L=1 in our case. Leaving this 3D form will introduce subtle, silent bug so don't forget
        to call .squeeze(1).
        """
        parent_state_prob = ancestor_states[treecal.parents].squeeze(1)
        """
        Event probabilities, as a [E, K] tensor with a row equal to 'sampling' if
        the edge child is a tip, or equal to 'birth' otherwise. FIXME this assumes that
        a node is either a birth or a sampling event.
        """
        # Step 1: identify tip children
        edge_child_is_tip = torch.isin(treecal.children, treecal.tip_indices)    # [E], dtype=bool

        # Step 2: create a mask for broadcasting: [E, 1]
        mask = edge_child_is_tip.unsqueeze(1)  # [E, 1]

        # Step 3: broadcast birth and sampling to [E, K]
        birth_rows = birth.unsqueeze(0).expand(treecal.nedges(), self.num_states)      # [E, K]
        sampling_rows = sampling.unsqueeze(0).expand(treecal.nedges(), self.num_states)  # [E, K]

        # Step 4: use the mask to select between them
        sampling_or_birth = torch.where(mask, sampling_rows, birth_rows)  # [E, K]
        """
        Take the weighted sum of q-ratios x sampling/birth parameters weighted by the
        probability that parent has state k
        """
        weighted_qmat = q_ratio * sampling_or_birth * parent_state_prob
        weighted_qmat_rowsumlogs = weighted_qmat.sum(-1).log()
        bds_log_likelihood = weighted_qmat_rowsumlogs
        return bds_log_likelihood
    
@torch.jit.script
def stadler_q_general(ages: torch.Tensor,
                      birth: torch.Tensor,
                      death: torch.Tensor,
                      sampling: torch.Tensor) -> torch.Tensor:
    """
    Generalized Stadler q(t) computation for node-specific birth rates.

    Args:
        ages:      [N] (node ages)
        birth:     scalar or [N] (birth rate per node)
        death:     scalar or [N] (death rate)
        sampling:  scalar or [N] (sampling rate)

    Returns:
        q: [N]
    """
    N = ages.shape[0]

    if birth.numel() == 1:
        birth = birth.expand(N)
    if death.numel() == 1:
        death = death.expand(N)
    if sampling.numel() == 1:
        sampling = sampling.expand(N)

    bdsdiff = birth - death - sampling
    c1 = torch.sqrt((bdsdiff ** 2 + 4. * birth * sampling).clamp_min(1e-8))  # [N]
    c2 = -bdsdiff / c1                                                       # [N]

    q = (
        2. * (1. - c2**2) +
        torch.exp(-c1 * ages) * (1. - c2)**2 +
        torch.exp(c1 * ages) * (1. + c2)**2
    )
    return q

@torch.jit.script
class LinearMarkerBirthModel:
    """
    Birth-death-sampling model with birth rate modeled as a linear function
    of marker posterior probabilities, using K-1 coefficients per marker
    (state 0 is the reference).

    Attributes:
        intercept (Tensor): scalar intercept term.
        coeffs (Tensor): shape (L, K-1), coefficients for each marker and non-reference state.
        death_log (Tensor): log-death rate (scalar).
        sampling_log (Tensor): log-sampling rate (scalar).
    """
    def __init__(self,
                 intercept: torch.Tensor,
                 coeffs: torch.Tensor,  # shape (L, K-1)
                 death_log: torch.Tensor,
                 sampling_log: torch.Tensor):

        self.intercept = intercept  # scalar
        self.coeffs = coeffs        # (L, K-1)
        self.death_log = death_log
        self.sampling_log = sampling_log

    def log_likelihood(self,
                       treecal: TreeTimeCalibrator,
                       ancestor_states: torch.Tensor  # shape (N, L, K)
                      ) -> torch.Tensor:
        """
        Compute log-likelihood under BDS with linear birth model.

        Args:
            treecal: calibrated tree structure
            ancestor_states: tensor of shape (N, M, K) with marker posteriors

        Returns:
            log-likelihood scalar
        """
        # Step 1: Compute expected birth rate at each node
        probs_nonref = ancestor_states[:, :, 1:].clone()  # (N, L, K-1)
        birth_expect = self.intercept + torch.einsum("nmk,mk->n", probs_nonref, self.coeffs)  # (N,)
        birth_expect = torch.clamp(birth_expect, min=1e-8)  # Ensure positivity

        # Step 2: Extract birth rate at parent of each edge
        birth_edge = birth_expect[treecal.parents]  # (E,)

        # Step 3: Constant death/sampling (scalar)
        death = self.death_log.exp()
        sampling = self.sampling_log.exp()

        # Step 4: Use stable q-ratio computation via general q function
        ages = treecal.ages()  # (N,)
        q = stadler_q_general(ages, birth_expect, death, sampling)  # (N,)
        q_ratio = q[treecal.children] / q[treecal.parents]  # (E,)

        # Step 5: Likelihood components
        logL = (
            torch.log(q_ratio).sum() +
            (treecal.ntips() - 1.) * torch.log(birth_edge).sum() / birth_edge.numel() +
            treecal.ntips() * self.sampling_log
        )

        return logL
