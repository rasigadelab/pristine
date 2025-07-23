#%%
import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
"""
Linear model state-dependent BDS.
"""
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
        self.birth_intercept = 3.0
        self.birth_coeff = [-1.5, +2.0]

    # Signature must be enforced
    def visit(self, node: BinaryTreeNode)->None:
        if node.parent is None and node.data is not None:
            return None # Skip the root if data is provided

        if node.data is None:
            node.data = StatefulBDSNodeData()
        
        # Generate sequence
        self.seq_visitor.visit(node)
        # State-dependent parameters
        states = node.data.state_indices().tolist()

        node.data.birth_rate = \
            self.birth_intercept + \
            self.birth_coeff[0] * states[0] + \
            self.birth_coeff[1] * states[1]
        node.data.sampling_rate = 0.5

# Generate a random 2-state GTR model
num_states = 2
sequence_length = 2
gtr = GeneralizedTimeReversibleModel(num_states)
gtr.rates_log -= 2.0 # Default rates are too fast, signal saturates

sequence_visitor = SequenceSimulationVisitor(gtr)

# Start with root in state zero. Preserve dimension (K, L) where L is 1
root: BinaryTreeNode=BinaryTreeNode()
root.data = StatefulBDSNodeData()
root.data.sequence = F.one_hot(torch.zeros(sequence_length, dtype=torch.int64), num_classes=num_states)
root.data.birth_rate = 1.0

model = StatefulBDSSimulationVisitor(sequence_visitor)
bdsforward = BirthDeathSamplingSimulator(root, model)

n = 1600
root = bdsforward.simulate(n)
nodes = root.nodelist()

#########################################################################
# MODEL ENERGY FUNCTION
#########################################################################
from pristine.felsenstein import FelsensteinPruningAlgorithm
from pristine.bds_model import LinearMarkerBirthModel

class Model:
    def __init__(self, 
                 fpa: FelsensteinPruningAlgorithm,
                 bds: LinearMarkerBirthModel
                 ):
        self.fpa: FelsensteinPruningAlgorithm = fpa
        self.bds: LinearMarkerBirthModel = bds

    def loss(self)->torch.Tensor:

        fpa_log_likelihood = self.fpa.log_likelihood().sum()
        bds_log_likelihood = self.bds.log_likelihood(
            # treecal=self.fpa.treecal,
            # ancestor_states=self.fpa.ancestor_states
        ).sum()

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

# BDS with a single sampling parameter
bds = LinearMarkerBirthModel(treecal=treecal,
                             ancestor_states=fpa.ancestor_states,
                            intercept=torch.tensor(1.0, requires_grad=True),
                             coeffs=torch.zeros([sequence_length, num_states-1], requires_grad=True),
                             death_log=torch.tensor(float("-inf")),
                             sampling_log=torch.tensor(0., requires_grad=True)
                             )

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

print(f"Estimated sampling rate: {bds.sampling_log.exp().item(): .3f}")
print(f"Estimated birth intercept: {bds.intercept.item(): .3f}")
print(f"Estimated beta_0: {bds.coeffs[0].item(): .3f}, beta_1: {bds.coeffs[1].item(): .3f}")

#%%
# Curvature report
from pristine.laplace_estimator import LaplaceEstimator

laplace = LaplaceEstimator(model)
laplace.curvature_report()

# %%
