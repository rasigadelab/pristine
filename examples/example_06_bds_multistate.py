#%%
"""
State-dependent BDS.
"""
import torch
#########################################################################
# SIMULATE
#########################################################################
print("Simulating data...")

from pristine.binarytree import BinaryTreeNode
from pristine.bds_model import BirthDeathSamplingNodeData, BirthDeathSamplingSimulator
from pristine.sequence import SequenceNodeData, SequenceSimulationVisitor
from pristine.gtr import GeneralizedTimeReversibleModel

# Combine BDS and sequence data in a single forward simulator
class StatefulBDSNodeData(BirthDeathSamplingNodeData, SequenceNodeData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# A forward model object exposes a simulate(node) function to generate
# data or parameters during forward simulation. Here we use the 
# BirthDeathSamplingNodeData base class to provide constant BDS parameters.
class StatefulBDSSimulationVisitor:
    """
    Implement functions to generate data and BDS parameters
    along a tree so that data can influence tree growth.
    """
    def __init__(self, seq_visitor: SequenceSimulationVisitor):
        self.seq_visitor: SequenceSimulationVisitor = seq_visitor

    # Signature must be enforced
    def visit(self, node: BinaryTreeNode)->None:
        if node.parent is None and node.data is not None:
            return None # Skip the root if data is provided

        if node.data is None:
            node.data = StatefulBDSNodeData()
        
        # Generate sequence
        self.seq_visitor.visit(node)
        # State-dependent parameters
        state = node.data.state_indices()[0].item()

        node.data.birth_rate = 4.0 if state == 1 else 1.0
        node.data.sampling_rate = 0.5

# Generate a random 2-state GTR model
num_states = 2
sequence_length = 1
gtr = GeneralizedTimeReversibleModel(num_states)
gtr.rates_log -= 2.0 # Default rates are too fast, signal saturates

sequence_visitor = SequenceSimulationVisitor(gtr)

# Start with root in state zero. Preserve dimension (K, L) where L is 1
root: BinaryTreeNode=BinaryTreeNode()
root.data = StatefulBDSNodeData()
root.data.sequence = torch.tensor([1., 0.]).unsqueeze(0) 
root.data.birth_rate = 1.0

model = StatefulBDSSimulationVisitor(sequence_visitor)
bdsforward = BirthDeathSamplingSimulator(root, model)

n = 1600
root = bdsforward.simulate(n)
# root.visualize()

"""
Plot tree states
"""
# import toytree

nodes = root.nodelist()
# tree = toytree.tree(root.newick(), internal_labels="name")
# tree_ordered_nodes = [nodes[ int(node.name) ] for node in tree.get_nodes()]

# tree.set_node_data(feature="state", data =[node.data.state_indices().item() for node in tree_ordered_nodes], default=0, inplace=True)
# tree.set_node_data(feature="birth", data =[node.data.birth_rate for node in tree_ordered_nodes], default=0, inplace=True)
# tree.draw(
#     node_labels="state", node_sizes=15, node_mask=False, node_colors=("state", "BlueRed"),
#     edge_colors="white", edge_widths=1,
#     tip_labels_style={"font-size": 10, "anchor-shift": 20},
#     tip_labels_colors="white",
#     scale_bar=True,
#     width = 600
# )

"""
Check that branch lengths differ sufficiently depending on marker state
"""
import matplotlib.pyplot as plt

edges, edge_lengths = root.edges_with_length()

states = [n.data.state_indices().item() for n in nodes]

len_0 = [edge_lengths[(parent, child)] for (parent, child) in edges if states[parent] == 0]
len_1 = [edge_lengths[(parent, child)] for (parent, child) in edges if states[parent] == 1]

# Create the boxplot
plt.boxplot([len_0, len_1], tick_labels=['State 0', 'State 1'])
plt.title('Branch length depending on parent state')
plt.ylabel('Branch length (y)')
# plt.grid(True)
# plt.show()

#########################################################################
# MODEL ENERGY FUNCTION
#########################################################################
"""
Support as many parameters as there are states

Ancestor states stored in FPA.ancestor_states, shape (N, U, K)
"""

from pristine.felsenstein import FelsensteinPruningAlgorithm
from pristine.bds_model import StateDependentBirthDeathSampling

class Model:
    def __init__(self, 
                 fpa: FelsensteinPruningAlgorithm,
                 bds: StateDependentBirthDeathSampling
                 ):
        self.fpa: FelsensteinPruningAlgorithm = fpa
        self.bds: StateDependentBirthDeathSampling = bds

    def loss(self)->torch.Tensor:

        fpa_log_likelihood = self.fpa.log_likelihood().sum()
        bds_log_likelihood = self.bds.log_likelihood(
            ancestor_states=self.fpa.ancestor_states
        ).sum()

        return -fpa_log_likelihood - bds_log_likelihood
    
#########################################################################
# PREPARE INITIAL GUESS
#########################################################################
print("Preparing starting conditions...")
from pristine.sequence import SequenceCollector, CollapsedConditionalLikelihood

### Collate unique sites into a sequence structure. Erase ancestor sequences
# to mimic application.
collector = SequenceCollector(root).collect().erase_node_sequences()
markers = CollapsedConditionalLikelihood(collector.sequences)

# Consolidated tree structure with edges etc. The _fixed suffix implies
# that edge lengths are fixed (known)
treecal = root.edgelist().get_tree_time_calibrator_fixed()

# Random starting point for GTR model. Use .requires_grad_(True) on
# parameters to make them trainable.
gtr_optim = GeneralizedTimeReversibleModel(num_states)
gtr_optim.stationary_logits.requires_grad_(True)
gtr_optim.rates_log -= 4 # Don't start too far from the objective
gtr_optim.rates_log.requires_grad_(True)
fpa = FelsensteinPruningAlgorithm(gtr_optim, markers, treecal)

# BDS with a single sampling parameter
bds = StateDependentBirthDeathSampling(treecal, num_states)
bds.birth_log = torch.zeros(num_states, requires_grad=True)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################
print("Launching optimizer...")
import pristine.optimize
import time

model = Model(fpa=fpa, bds=bds)
loss_init = model.loss().item()
optim = pristine.optimize.Optimizer(model)

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={model.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"No. of iterations: {optim.num_iter}")

print(f"Estimated state-dependent birth rate: {bds.birth()[0]: .3f}, {bds.birth()[1]: .3f}")
print(f"Estimated state-dependent sampling rate: {bds.sampling()[0]: .3f}")


# %%
