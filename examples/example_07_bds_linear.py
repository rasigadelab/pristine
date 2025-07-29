#%%
import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
import math
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
# from pristine.gtr import GeneralizedTimeReversibleModel
from pristine.substitution_models import GTRModel
from pristine.molecularclock import ConstantClock

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
        self.birth_intercept = 2.
        self.birth_coeff = [-1., +1.]

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

        node.data.birth_rate = math.exp(
            self.birth_intercept + 
            self.birth_coeff[0] * states[0] +
            self.birth_coeff[1] * states[1]
            )
        node.data.death_rate = 0.0
        node.data.sampling_rate = 0.5

# Generate a random 2-state GTR model
num_states = 2
sequence_length = 2

subst_model_ref = GTRModel(num_states)
clock_ref = ConstantClock(torch.tensor(-2.))
sequence_visitor = SequenceSimulationVisitor(clock_ref, subst_model_ref)

# Start with root in state zero. Preserve dimension (K, L) where L is 1
root: BinaryTreeNode=BinaryTreeNode()
root.data = StatefulBDSNodeData()
root.data.sequence = F.one_hot(torch.zeros(sequence_length, dtype=torch.int64), num_classes=num_states)
root.data.birth_rate = 1.0

model = StatefulBDSSimulationVisitor(sequence_visitor)
bdsforward = BirthDeathSamplingSimulator(root, model)

n = 800
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

# Define clock and substitution models
clock = ConstantClock()
subst_model_test = GTRModel(num_states)
fpa = FelsensteinPruningAlgorithm(subst_model_test, markers, treecal, clock)

# BDS with a single sampling parameter
bds = LinearMarkerBirthModel(treecal=treecal,
                             ancestor_states=fpa.ancestor_states)

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

print(f"Estimated sampling rate: {bds.sampling().item(): .3f}")
print(f"Estimated birth intercept: {bds.intercept.item(): .3f}")
print(f"Estimated beta_0: {bds.coeffs[0].item(): .3f}, beta_1: {bds.coeffs[1].item(): .3f}")

