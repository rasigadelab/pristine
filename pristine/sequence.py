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
This module provides tools for simulating, storing, and analyzing molecular 
sequence data on a phylogenetic tree using a continuous-time Markov model.

Conceptually, sequence evolution is modeled by assigning a sequence of discrete 
states (e.g., nucleotides or amino acids) to the root of a binary tree, sampled 
from the stationary distribution of a substitution model. As we traverse the tree 
from the root to the tips, each branch propagates this sequence using a transition 
probability matrix derived from the rate matrix and the elapsed time.

The `SequenceSimulationVisitor` class performs this simulation in a tree traversal, 
generating synthetic sequences at every node. Each node's data is stored as 
a one-hot encoded tensor.

The `SequenceCollector` class traverses the tree to collect all node sequences 
into a single tensor. This is especially useful for downstream likelihood 
calculation, as it collapses identical site patterns across nodes for 
computational efficiency.

The `CollapsedConditionalLikelihood` class stores these collapsed sequence 
patterns and their counts, reducing redundancy for likelihood evaluation. 
This representation is central to efficient implementations of the Felsenstein 
pruning algorithm.

Together, these classes support the simulation of realistic sequence evolution 
along a phylogeny, as well as the efficient preprocessing of sequence data 
for maximum likelihood inference.
"""
from __future__ import annotations # For forward references within class definition
import sys;import os;sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
from typing import Dict

from .binarytree import BinaryTreeNode
# from .gtr import GeneralizedTimeReversibleModel
from .molecularclock import ConstantClock
from .substitution_models import GTRModel

#########################################################################
# SEQUENCE SIMULATION (PYTHON SIDE)
#########################################################################

class SequenceNodeData:
    """
    Holds sequence data in BinaryTreeNode.data field
    """
    def __init__(self, **kwargs):
        self.sequence: torch.Tensor = None
        super().__init__(**kwargs)

    def sequence_length(self):        
        return self.sequence.shape[0]
    
    def num_states(self):
        return self.sequence.shape[1]
    
    def state_indices(self)->torch.Tensor:
        """Return a sequence of the indices of states with maximum probability"""
        return self.sequence.argmax(dim=-1)

class SequenceSimulationVisitor(BinaryTreeNode.Visitor):
    """
    Simulate a sequence of markers. Currently supports a constant rate.
    """
    def __init__(self,
                 clock: ConstantClock,
                 transition_model: GTRModel, 
                 sequence_length: int = 1):
        self.clock = clock
        self.sequence_length: int = sequence_length
        self.transition_model: GTRModel = transition_model
        Q, pi = transition_model.rate_matrix_stationary_dist()
        self.transition_matrix: torch.Tensor = Q
        self.stationary_distribution: torch.Tensor = pi
        self.clock_rate = clock.rate()

    def visit(self, node: BinaryTreeNode)->None:
        if node.data is None:
            node.data = SequenceNodeData()

        if node.parent is None:
            # Sample one state per site (resulting tensor has shape (L,))
            root_states = torch.multinomial(
                self.stationary_distribution, 
                num_samples=self.sequence_length, 
                replacement=True)
            # Convert to one-hot encoding (result shape: (L, K))
            node.data.sequence = F.one_hot(root_states, num_classes=self.transition_model.K)
        else:            
            # Compute the transition probability matrix P(t)
            waiting_time = node.time - node.parent.time
            P = torch.matrix_exp(
                self.transition_matrix * waiting_time * self.clock_rate
                )  # shape: (K, K)
            
            # For each site, find the parent's state index
            # (since parent_seq is one-hot, argmax gives the state index)
            # parent_indices = node.parent.data.sequence.argmax(dim=-1)  # shape: (L,)
            parent_indices = node.parent.data.state_indices()

            # For each site, pick the row corresponding to the parent's state,
            # so that transition_probs has shape (L, K)
            transition_probs = P[parent_indices]
            
            # Sample a new state for each site from its transition probability vector.
            # torch.multinomial applied to a 2D tensor samples one index per row.
            child_states = torch.multinomial(transition_probs, num_samples=1, replacement=True).squeeze(1)
            
            # Convert the sampled states to one-hot encoding
            node.data.sequence = F.one_hot(child_states, num_classes=self.transition_model.K)
        pass

#########################################################################
# SEQUENCE RETRIEVAL (TORCH SIDE)
#########################################################################

@torch.jit.script
class CollapsedConditionalLikelihood:
    """
    Holds tensor of shape (N, U, K) where N is the number of nodes and tips, U
    is the number of unique site patterns, K is the number of states, 
    """
    def __init__(self, seq_tensor: torch.Tensor):
        seq_tensor = seq_tensor.contiguous()
        unique_patterns, pattern_counts = torch.unique(seq_tensor, dim=1, return_counts=True)       
        self.unique_patterns: torch.Tensor = unique_patterns
        self.pattern_counts: torch.Tensor = pattern_counts
        self.num_nodes: int = self.unique_patterns.shape[0]
        self.sequence_length: int = seq_tensor.shape[1]
        self.num_states: int = self.unique_patterns.shape[2]
        self.num_unique_patterns: int = self.unique_patterns.shape[1]

class SequenceCollector:
    """
    Traverse a tree to collect sequences and return a unique site pattern object
    """
    def __init__(self, root: BinaryTreeNode):
        self.root: BinaryTreeNode = root
        self.num_nodes: int = 0
        self.sequence_length: int = 0
        self.num_states: int = 0
        self.sequences: torch.Tensor = None
        

    class SequenceCollectionVisitor(BinaryTreeNode.Visitor):
        """
        Populates a dictionary of sequence tensor references
        """
        def __init__(self, root: BinaryTreeNode):
            self.sequence_length: int = root.data.sequence.shape[0]
            self.num_states: int = root.data.sequence.shape[1]
            self.sequence_dict: Dict[int, torch.Tensor] = {}            

        def visit(self, node: BinaryTreeNode):
            self.sequence_dict[node.id] = node.data.sequence

    def collect(self)->SequenceCollector:
        # Populate a dictionary of sequences
        visitor = SequenceCollector.SequenceCollectionVisitor(self.root)
        self.root.bfs(visitor)
        # print(visitor.sequence_dict)
        
        self.sequence_length = visitor.sequence_length
        self.num_states = visitor.num_states
        self.num_nodes = len(visitor.sequence_dict)

        self.sequences = torch.ones([self.num_nodes, self.sequence_length, self.num_states]) # (N, L, K) in math notation

        for node_id, sequence in sorted(visitor.sequence_dict.items()):
            self.sequences[node_id,:,:] = sequence

        return self

    def erase_node_sequences(self)->SequenceCollector:
        """
        Put all node sequences in default state, a vector of all 1's.
        Used in simulation context to mimick observing only tip sequences.
        """
        for node in self.root.nodelist():
            if node.left is not None or node.right is not None:
                self.sequences[node.id,:,:] = torch.ones([self.sequence_length, self.num_states])
        return self