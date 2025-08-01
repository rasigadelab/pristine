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
TreeTimeCalibrator

This class represents a phylogenetic tree in temporal coordinates. It
encodes the parent–child relationships between nodes as tensors and assigns
dates to both internal nodes and tip leaves. These dates may be fixed
(observed tip dates) or learnable (inferred node dates). From these, the
class computes edge durations (i.e., branch lengths in time) and also
provides ages measured backward from the latest sampling time. This
structure is optimized for differentiable computations such as likelihood
evaluation or molecular clock calibration in time-scaled phylogenies.

EdgeList

This class provides a minimal representation of a phylogenetic tree as a
list of edges and corresponding branch lengths. It supports key operations
such as identifying tips, computing root-to-leaf distances, generating
Newick strings for visualization, and converting to tensor format for
modeling. It can also build initial date assignments consistent with
observed tip dates. Together, these utilities enable conversion of
phylogenetic structures into time-aware models for use in likelihood
calculations and optimization workflows.
"""

from __future__ import annotations # For forward references within class definition
from collections import deque
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree
from typing import Tuple, List, Dict
import torch

@torch.jit.script
class TreeTimeCalibrator:
    def __init__(
        self, 
        parents: torch.Tensor,
        children: torch.Tensor,
        distances: torch.Tensor,
        tip_indices: torch.Tensor, 
        node_indices: torch.Tensor,
        tip_dates: torch.Tensor,
        node_dates: torch.Tensor
        ):
        self.parents: torch.Tensor = parents
        self.children: torch.Tensor = children
        self.distances: torch.Tensor = distances
        self.tip_indices: torch.Tensor = tip_indices
        self.tip_dates: torch.Tensor = tip_dates
        self.node_indices: torch.Tensor = node_indices
        self.node_dates: torch.Tensor = node_dates

    def ntips(self)->int:
        return len(self.tip_indices)
    
    def nnodes(self)->int:
        return len(self.node_indices)
    
    def nedges(self)->int:
        return self.parents.shape[0]

    def dates(self)->torch.Tensor:
        """Dates of nodes and tips, forward in time"""
        dates = torch.zeros(self.nnodes() + self.ntips())
        dates[self.tip_indices] = self.tip_dates
        dates[self.node_indices] = self.node_dates
        return dates
    
    def ages(self)->torch.Tensor:
        """Ages of nodes and tips, backward in time 
        before the last sampled tip"""
        last_date = self.tip_dates.max()
        ages = torch.zeros(self.nnodes() + self.ntips())
        ages[self.tip_indices] = last_date - self.tip_dates
        ages[self.node_indices] = last_date - self.node_dates
        return ages
    
    def durations(self)->torch.Tensor:
        dates = self.dates()
        durations = dates[self.children] - dates[self.parents]
        return durations
    
    def internal_edge_indices(self) -> torch.Tensor:
        """
        Return the indices of edges whose child is NOT a tip (i.e., internal→internal).
        """
        mask = ~torch.isin(self.children, self.tip_indices)
        return torch.nonzero(mask).squeeze(1)

    def terminal_edge_indices(self) -> torch.Tensor:
        """
        Return the indices of edges whose child is a tip (i.e., internal→tip).
        """
        mask = torch.isin(self.children, self.tip_indices)
        return torch.nonzero(mask).squeeze(1)

    def parent_node_index_lookup(self) -> torch.Tensor:
        """
        Return a lookup tensor mapping each edge's parent (global node index)
        to its local index in the internal node tensor (`node_dates` or any [N_nodes]-length parameter).

        This is needed when parameters (e.g., birth rates) are stored only for internal nodes,
        and must be indexed per edge using the global parent node IDs.

        Returns:
            Tensor of shape [E], where E = number of edges. Each value is an index into
            the [N_nodes]-length internal node vector corresponding to self.node_indices.
        """
        N = self.nnodes() + self.ntips()
        node_id_to_local = torch.full((N,), -1, dtype=torch.long)
        node_id_to_local[self.node_indices] = torch.arange(self.nnodes(), dtype=torch.long)
        return node_id_to_local[self.parents]

    def child_node_index_lookup(self) -> torch.Tensor:
        """
        Return a lookup tensor mapping each edge's child (global node index)
        to its local index in the internal node tensor (`node_dates` or similar).

        This is useful when child-specific parameters are stored only for internal nodes,
        and you need to map edge children to those.

        Returns:
            Tensor of shape [E], where E = number of edges. Each value is an index into
            the [N_nodes]-length internal node vector, or -1 if the child is a tip.
        """
        N = self.nnodes() + self.ntips()
        node_id_to_local = torch.full((N,), -1, dtype=torch.long)
        node_id_to_local[self.node_indices] = torch.arange(self.nnodes(), dtype=torch.long)
        return node_id_to_local[self.children]


class EdgeList:
    """
    Minimal R-like structure for phylogenetic trees with a list of edges and a list of edge lengths
    """
    def __init__(self, edges: List[Tuple[int, int]], edge_lengths: List[float], tip_names: Dict[int, str]={}):
        self.edges: List[Tuple[int, int]] = edges
        self.edge_lengths: List[float] = edge_lengths
        self.tip_names: Dict[int, str] = tip_names

    def parent_and_children_sets(self)->Tuple[set, set]:
        """
        Return sets of parent and child indices
        """
        parents = set()
        children = set()
        
        for parent, child in self.edges:
            parents.add(parent)
            children.add(child)

        return parents, children
    
    def nodes_and_tips(self)->Tuple[List[int], List[int]]:
        """
        Return a list of nodes and a list of tips
        """        
        parents, children = self.parent_and_children_sets()

        # Tips are nodes that appear as children but not as parents
        tips = children - parents
        return list(parents), list(tips)
    
    def root(self)->int:
        """
        Return the root index (assuming a single root)
        """
        parents, children = self.parent_and_children_sets()

        root_set = parents - children
        assert len(root_set) == 1
        return next(iter(root_set))


    def newick(self)->str:
        """
        Build a Newick format string from the tree edges and branch lengths.
        
        :param edges: List of tuples representing parent-child relationships.
        :param branch_lengths: Dictionary mapping edges to branch lengths.
        :return: Newick format string representing the tree.

        WARNING by default the Newick string will contain scientific notation to handle small branches. Should change that
        or make it optional if other software complain.
        """
        child_map = {}
        
        # for parent, child in self(edges):
        for e in range(len(self.edges)):
            parent, child = self.edges[e]
            if parent not in child_map:
                child_map[parent] = []
            child_map[parent].append((child, self.edge_lengths[e]))
        
        def recursive_newick(node_id):
            if node_id not in child_map:
                return f"{node_id}"  # Leaf node
            return f"({','.join(f'{recursive_newick(child)}:{length:.6e}' for child, length in child_map[node_id])}){node_id}"
        
        return recursive_newick(self.root()) + ";"
    
    def as_Phylo(self):
        from Bio import Phylo
        from io import StringIO
        newick_str = self.newick()
        handle = StringIO(newick_str)
        tree = Phylo.read(handle, "newick")
        return tree

    def plot(self)->None:
        from Bio import Phylo
        tree = self.as_Phylo()
        Phylo.draw(tree, do_show=False)

    def as_torch(self, device = torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the edge list and durations as torch.Tensors
        """
        return (torch.tensor(self.edges, device=device, dtype=torch.int32), torch.tensor(self.edge_lengths, device=device, dtype=torch.float64))
    
    def distances_from_root(self)->List[float]:
        """
        Compute the distance from the root to each node or leaf in the tree.

        :return: A list indexed by node ID containing distances from the root.
        """
        # Create a dictionary to store distances
        root_id = self.root()
        distances = {root_id: 0.0}  # Distance from root to itself is 0
        
        # Store edge lengths by child index and convert edges to adjacency list
        length_by_child = {}
        child_map = {}
        for edge_index in range(len(self.edges)):
            parent, child = self.edges[edge_index]
            if parent not in child_map:
                child_map[parent] = []
            child_map[parent].append(child)
            length_by_child[child] = self.edge_lengths[edge_index]
        
        # Use a stack for depth-first traversal
        stack = [root_id]
        while stack:
            parent = stack.pop()
            if parent in child_map:
                for child in child_map[parent]:
                    edge = (parent, child)
                    distances[child] = distances[parent] + length_by_child[child]
                    stack.append(child)
        
        # Convert distances dictionary to a list indexed by node ID
        max_node_id = max(distances.keys())
        distance_list = [0.0] * (max_node_id + 1)
        for node_id, dist in distances.items():
            distance_list[node_id] = dist
        
        return distance_list
    
    def tip_distances_from_root(self)->Dict[int, float]:
        """
        Return a dictionary of tip distances from root indexed by tip index
        """
        distances = self.distances_from_root()
        _, tips = self.nodes_and_tips()
        return {id: distances[id] for id in tips}
    
    def propose_feasible_dates(self, dates: Dict[int, float], eps = 1.0)->Dict[int, float]:
        """
        Estimate a feasible set of dates given fixed tip dates
        """
        dates = dict(dates)
        for (parent, child) in reversed(self.edges):
            assert child in dates.keys()
            if parent in dates.keys():
                dates[parent] = min(dates[parent], dates[child] - eps)
            else:
                dates[parent] = dates[child] - eps
        return dates
    
    def get_tree_time_calibrator(self, tip_dates: Dict[int, float])->TreeTimeCalibrator:
        """
        Obtain a set of tensors representing tip and nodes indices and dates,
        given a dictionary of tip dates
        """
        starting_dates = self.propose_feasible_dates(tip_dates)
        torch_dates = torch.tensor([date for (_, date) in sorted(starting_dates.items()) ])
        nodes, tips = self.nodes_and_tips()
        
        torch_edges = torch.tensor(self.edges, dtype=torch.int64)

        torch_tips = torch.tensor(tips, dtype=torch.int64)
        torch_nodes = torch.tensor(nodes, dtype=torch.int64)

        torch_tip_dates = torch_dates[torch_tips].clone().detach()
        torch_node_dates = torch_dates[torch_nodes].clone().detach().requires_grad_(True)

        return TreeTimeCalibrator(
            parents=torch_edges[:,0],
            children=torch_edges[:,1],
            distances=torch.tensor(self.edge_lengths),
            tip_indices=torch_tips,
            tip_dates=torch_tip_dates,
            node_indices=torch_nodes,
            node_dates=torch_node_dates
        )
    
    def get_tree_time_calibrator_fixed(self)->TreeTimeCalibrator:
        """
        Obtain a set of tensors representing tip and nodes indices and dates,
        for when dates are considered fixed.
        """
        torch_dates = torch.tensor(self.distances_from_root())
        nodes, tips = self.nodes_and_tips()
        
        torch_edges = torch.tensor(self.edges, dtype=torch.int64)

        torch_tips = torch.tensor(tips, dtype=torch.int64)
        torch_nodes = torch.tensor(nodes, dtype=torch.int64)

        torch_tip_dates = torch_dates[torch_tips].clone().detach()
        torch_node_dates = torch_dates[torch_nodes].clone().detach()

        return TreeTimeCalibrator(
            parents=torch_edges[:,0],
            children=torch_edges[:,1],
            distances=torch.tensor(self.edge_lengths),
            tip_indices=torch_tips,
            tip_dates=torch_tip_dates,
            node_indices=torch_nodes,
            node_dates=torch_node_dates
        )
    
    def new_from_Phylo(tree: Tree)->EdgeList:
        """
        Given a Biopython Tree object, returns:
        - edges: list of (parent_index, child_index) tuples
        - edge_lengths: list of branch lengths (float)
        - tip_labels: dict mapping tip index to tip name
        """
        node_index = {}
        edges = []
        edge_lengths = []
        tip_labels = {}

        index_counter = 0
        queue = deque([(None, tree.root)])  # (parent, current_clade)

        while queue:
            parent, clade = queue.popleft()

            # Assign unique index if new
            if clade not in node_index:
                node_index[clade] = index_counter
                index_counter += 1

            clade_idx = node_index[clade]

            # Record tip label
            if clade.is_terminal():
                tip_labels[clade_idx] = clade.name

            for child in clade.clades:
                queue.append((clade, child))

                if child not in node_index:
                    node_index[child] = index_counter
                    index_counter += 1

                child_idx = node_index[child]
                edges.append((clade_idx, child_idx))

                length = child.branch_length if child.branch_length is not None else 0.0
                edge_lengths.append(length)

        return EdgeList(edges, edge_lengths, tip_labels)
    
    def new_from_TreeTimeCalibrator(treecal: TreeTimeCalibrator)->EdgeList:
        edges = [e for e in zip(treecal.parents.tolist(), treecal.children.tolist())]
        edge_lengths = treecal.durations().tolist()
        return EdgeList(edges, edge_lengths)

    def add_edge(self, parent: int, child: int, length: float = 0.0):
        """
        Add an edge (parent, child) with specified branch length.
        """
        self.edges.append((parent, child))
        self.edge_lengths.append(length)

    def remove_edge(self, index: int):
        """
        Remove the edge at the given index from the edge list and its branch length.

        Args:
            index: integer index of the edge to remove
        """
        if 0 <= index < len(self.edges):
            del self.edges[index]
            del self.edge_lengths[index]
        else:
            raise IndexError(f"Edge index {index} is out of bounds for removal.")

    def remove_edge_by_nodes(self, parent: int, child: int):
        """
        Remove the edge (parent, child) if it exists.

        Raises:
            ValueError if the edge is not found.
        """
        for i, (p, c) in enumerate(self.edges):
            if p == parent and c == child:
                self.remove_edge(i)
                return
        raise ValueError(f"Edge ({parent}, {child}) not found in the tree.")

    # UNTESTED
    def reroot_along_edge(self, edge_index: int, alpha: float) -> "EdgeList":
        """
        Return a new EdgeList rerooted at fractional position alpha along a given edge.

        Args:
            edge_index: index of edge to split
            alpha: fraction along edge (0 < alpha < 1)

        Returns:
            New EdgeList with root inserted along specified edge
        """
        assert 0 < alpha < 1, "alpha must be strictly between 0 and 1"

        import copy
        newtree = copy.deepcopy(self)
        parents, children = zip(*newtree.edges)
        L = parents[edge_index]
        R = children[edge_index]
        w = newtree.edge_lengths[edge_index]
        wL, wR = alpha * w, (1 - alpha) * w

        # Create a new internal node to serve as the root
        root = max(parents + children) + 1

        # Remove the original edge by marking it detached
        self.remove_edge(edge_index)

        # Add edges L → root and root → R
        newtree.add_edge(root, L, length=wL)
        newtree.add_edge(root, R, length=wR)

        return newtree

