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
BinaryTreeNode represents a rooted binary tree where each node tracks its parent,
children, and time. This class is central to phylogenetic models where trees encode
ancestral relationships, branching times, and evolutionary structure.

Each node has pointers to its left and right child, and its 'time' field represents
the node's position along the temporal axis—typically interpreted as the divergence
time from its ancestor. The tree can be traversed or modified using methods like
bfs(), drop_tip(), or grow_aldous_tree().

Functionally, BinaryTreeNode supports both structural operations and data-driven
tasks. For example, edgelist() extracts edges and branch durations; tip_indices()
returns the leaves of the tree; and newick() builds a Newick-formatted string for
compatibility with external tools.

The class also enables forward-time simulation of trees via grow_aldous_tree(),
and supports visualization and reindexing to maintain consistent node identifiers.
It is designed to integrate easily with statistical and computational models by
storing arbitrary data at each node, including sequences, rates, or latent states.
"""

from __future__ import annotations # For forward references within class definition
from collections import deque
import random
from typing import List, Tuple, Optional
import torch
from .edgelist import EdgeList

class BinaryTreeNode:
    """
    A recursive binary tree implementation. 
    """
    def __init__(self, id: int=0, time: float=0.0, parent: Optional[BinaryTreeNode] = None):
        self.id = id
        self.time = time
        self.parent = parent
        self.left = None
        self.right = None
        self.data = None
        self.b = None
        self.d = None
        self.s = None
        self.mu = None

    def edgelist(self)->EdgeList:
        """
        Traverse the induced tree and extract edges and durations in Tensor format.
        """
        edges: List[Tuple[int, int]] = []
        edge_lengths: List[float] = []
        queue = deque([self])
        
        while queue:
            node = queue.popleft()
            # Process the left child if it exists.
            if node.left is not None:
                edges.append((node.id, node.left.id))
                edge_lengths.append(node.left.time - node.time)
                queue.append(node.left)
            # Process the right child if it exists.
            if node.right is not None:
                edges.append((node.id, node.right.id))
                edge_lengths.append(node.right.time - node.time)
                queue.append(node.right)
        
        # return torch.Tensor(edges), torch.Tensor(edge_lengths)
        return EdgeList(edges, edge_lengths)

    class Visitor:
        """
        Base class for node visitors, useful for type hints
        """
        def visit(self, node: BinaryTreeNode)->None:
            raise NotImplementedError("Vistor.visit() should not be called directly. Common cause of this error is " \
            "a missing .visit(node) method in the callback visitor class")

    def bfs(self: BinaryTreeNode, callback: BinaryTreeNode.Visitor)->BinaryTreeNode:
        """
        Breadth-first search the tree and apply a callback.
        The callback object must expose a .visit(node) method
        """
        queue = deque([self])
        
        while queue:
            current = queue.popleft()
            callback.visit(current)
            if current.left is not None:
                queue.append(current.left)
            if current.right is not None:
                queue.append(current.right)
        return self

    def edges_with_length(self):
        """
        DEPRECATED
        Traverse the tree in BFS order starting at 'root' and extract:
            - A list of edges as (parent_id, child_id) tuples.
            - A dictionary mapping each edge to its branch length (child.time - parent.time).

        :param root: The root BinaryTreeNode of the tree.
        :return: A tuple (edges, edge_lengths).
                    edges is a list of tuples (parent_id, child_id).
                    edge_lengths is a dictionary with keys (parent_id, child_id) and values as branch lengths.
        """
        if self.left is None and self.right is None:
            return [], {}
        
        edges = []
        edge_lengths = {}
        queue = deque([self])
        
        while queue:
            node = queue.popleft()
            # Process the left child if it exists.
            if node.left is not None:
                edges.append((node.id, node.left.id))
                # Compute branch length as difference in time.
                edge_lengths[(node.id, node.left.id)] = node.left.time - node.time
                queue.append(node.left)
            # Process the right child if it exists.
            if node.right is not None:
                edges.append((node.id, node.right.id))
                edge_lengths[(node.id, node.right.id)] = node.right.time - node.time
                queue.append(node.right)
        
        return edges, edge_lengths

    def reindex(self)->BinaryTreeNode:
        """
        Reindex the nodes in BFS order
        """
        q = deque()
        q.append(self)
        nodeid = 0
        while len(q) > 0:
            node = q.popleft()
            node.id = nodeid
            nodeid += 1
            if node.left is not None: q.append(node.left)
            if node.right is not None: q.append(node.right)
        return self

    def nodelist(self)->list[BinaryTreeNode]:
        """
        List of nodes indexed by node index.

        Perform a breadth-first search (BFS) starting from the root and return a list of all 
        TreeNode objects such that the index in the list is the node.id attribute. That is,
        result[node.id] == node.

        :param root: The root TreeNode of the tree.
        :return: A list of TreeNode objects indexed by their node.id.
        """
                # Use BFS to collect nodes and track the maximum id.
        nodes_by_id = {}
        max_id = 0
        queue = deque([self])
        
        while queue:
            current = queue.popleft()
            nodes_by_id[current.id] = current
            if current.id > max_id:
                max_id = current.id
            if current.left is not None:
                queue.append(current.left)
            if current.right is not None:
                queue.append(current.right)
        
        # Create a list where each index corresponds to a node.id.
        result = [None] * (max_id + 1)
        for node_id, node in nodes_by_id.items():
            result[node_id] = node

        return result

    def drop_tip(self, tip: BinaryTreeNode)->BinaryTreeNode:
        """
        Drop a tip and its parent and reconnect the grandparent and the tip's sibling
        """
        parent = tip.parent
        if parent is None:
            return None
        
        if tip.id == parent.left.id:
            # Graft right sibling to grandparent
            sibling = parent.right
        elif tip.id == parent.right.id:
            # Graft left sibling to grandparent
            sibling = parent.left
        else:
            raise RuntimeError("BinaryTreeNode is malformed")
        
        grandparent = parent.parent
        if grandparent is None:
            # raise RuntimeError("Removing a root's child would break binary splits")
            # The sibling becomes the root
            sibling.parent = None
            return sibling
        else:
            sibling.parent = grandparent
            if grandparent.left.id == parent.id:
                grandparent.left = sibling
            else:
                grandparent.right = sibling
            return self

    def grow_aldous_tree(self, n):
        """
        Grow a random binary tree with exactly n leaves using Aldous' model.
        
        :param n: Number of leaves in the generated tree.        
        :return: self (the root of the newly grown tree)
        """
        leaves = [self]
        node_count = 1

        while len(leaves) < n:
            # Pick a random leaf to split
            node = random.choice(leaves)
            
            # Replace the selected leaf with an internal node having two children
            node.left = BinaryTreeNode(node_count, node.time + random.expovariate(1.0), node)
            node_count += 1
            node.right = BinaryTreeNode(node_count, node.time + random.expovariate(1.0), node)
            node_count += 1

            # Remove the old leaf and add two new leaves
            leaves.remove(node)
            leaves.append(node.left)
            leaves.append(node.right)
        
        return self

    def tip_indices(self)->list[int]:
        """List of indices of tips"""
        q = deque()
        q.append(self)
        tips = []
        while len(q) > 0:
            node = q.popleft()
            if node.left is None and node.right is None:
                tips.append(node.id)
                continue
            if node.left is not None: q.append(node.left)
            if node.right is not None: q.append(node.right)
        return tips
    
    def newick(self)->str:
        """
        Build a Newick format string from the tree edges and branch lengths.
        :return: Newick format string representing the tree.
        """
        edges, edge_lengths = self.edges_with_length()
        child_map = {}
    
        for parent, child in edges:
            if parent not in child_map:
                child_map[parent] = []
            child_map[parent].append((child, edge_lengths[(parent, child)]))
        
        def recursive_newick(node_id):
            if node_id not in child_map:
                return f"{node_id}"  # Leaf node
            return f"({','.join(f'{recursive_newick(child)}:{length:.4f}' for child, length in child_map[node_id])}){node_id}"
        
        return recursive_newick(0) + ";"
    
    def visualize(self):
        from Bio import Phylo
        from io import StringIO
        newick_str = self.newick()
        handle = StringIO(newick_str)
        tree = Phylo.read(handle, "newick")
        Phylo.draw(tree, do_show=False)