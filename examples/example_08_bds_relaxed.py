#%%
"""
Relaxed BDS for exploratory analysis. Each parent node exhibits its own
birth rate, independent of state or lineage.
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

n = 800
root = bdsforward.simulate(n)
# root.visualize()

"""
Plot tree states
"""
# import toytree

nodes = root.nodelist()

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
plt.draw()

#########################################################################
# MODEL ENERGY FUNCTION
#########################################################################
"""
FPA not mandatory but useful here to recover ancestral states for display
purpose.
"""

from pristine.felsenstein import FelsensteinPruningAlgorithm
from pristine.bds_model import RelaxedBirthModel

class Model:
    def __init__(self, 
                 fpa: FelsensteinPruningAlgorithm,
                 bds: RelaxedBirthModel
                 ):
        self.fpa: FelsensteinPruningAlgorithm = fpa
        self.bds: RelaxedBirthModel = bds

    def loss(self)->torch.Tensor:

        fpa_log_likelihood = self.fpa.log_likelihood().sum()
        bds_log_likelihood = self.bds.log_likelihood().sum()

        return -fpa_log_likelihood -bds_log_likelihood
    
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

# Relaxed BDS
bds = RelaxedBirthModel(treecal)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################
print("Launching optimizer...")
import pristine.optimize
import time
import matplotlib.pyplot as plt

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

def hist_birth_by_state(model, ancestor_states: torch.Tensor):
    """
    Plots log birth rates by most probable state (state 0 vs. 1) for internal nodes only.

    Parameters:
        model: RelaxedBirthModel (must have .log_birth_nodes and .treecal.node_indices)
        ancestor_states: tensor of shape [N_total, L=1, K]
    """
    # Step 1: Restrict to internal nodes
    node_ids = model.treecal.node_indices  # shape: [N_nodes]
    anc_states_internal = ancestor_states[node_ids]  # shape: [N_nodes, 1, K]

    # Step 2: Get MAP state for each node
    state_assignments = anc_states_internal.squeeze(1).argmax(dim=-1)  # shape: [N_nodes]

    # Step 3: Split log birth rates
    log_birth_nodes = model.log_birth_nodes.detach().cpu()
    log_rates_0 = log_birth_nodes[state_assignments == 0].numpy()
    log_rates_1 = log_birth_nodes[state_assignments == 1].numpy()

    # Step 4: Plot
    plt.figure(figsize=(7, 4))
    plt.hist(log_rates_0, bins=20, alpha=0.6, label="State 0", edgecolor="black")
    plt.hist(log_rates_1, bins=20, alpha=0.6, label="State 1", edgecolor="black")
    plt.xlabel("Log Birth Rate")
    plt.ylabel("Count")
    plt.title("Log Birth Rate by Most Probable State (Internal Nodes)")
    plt.legend()
    plt.tight_layout()
    plt.draw()

hist_birth_by_state(bds, fpa.ancestor_states)

def boxplot_birth_by_state(model, ancestor_states: torch.Tensor):
    """
    Shows box plots of log birth rates by most probable state (state 0 vs. 1) for internal nodes.

    Parameters:
        model: RelaxedBirthModel (must have .log_birth_nodes and .treecal.node_indices)
        ancestor_states: tensor of shape [N_total, L=1, K]
    """
    # Step 1: Restrict to internal nodes
    node_ids = model.treecal.node_indices  # [N_nodes]
    anc_states_internal = ancestor_states[node_ids]  # [N_nodes, 1, K]

    # Step 2: Get most probable state
    state_assignments = anc_states_internal.squeeze(1).argmax(dim=-1)  # [N_nodes]

    # Step 3: Extract log birth rates
    log_birth_nodes = model.log_birth_nodes.detach().cpu()
    log_rates_0 = log_birth_nodes[state_assignments == 0].numpy()
    log_rates_1 = log_birth_nodes[state_assignments == 1].numpy()

    # Step 4: Box plot
    plt.figure(figsize=(6, 4))
    plt.boxplot([log_rates_0, log_rates_1],
                labels=["State 0", "State 1"],
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="black"),
                medianprops=dict(color="darkblue"))
    plt.ylabel("Log Birth Rate")
    plt.title("Log Birth Rate by Most Probable State (Internal Nodes)")
    plt.tight_layout()
    plt.draw()

boxplot_birth_by_state(bds, fpa.ancestor_states)

# %%
