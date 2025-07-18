# %%
"""
Example of modular design for tree dating and molecular clock estimation.
Uses the original error clock model.
"""
import time
import torch
import pristine.optimize
from pristine.binarytree import BinaryTreeNode
from pristine.edgelist import EdgeList, TreeTimeCalibrator
from pristine.molecularclock import ContinuousAdditiveRelaxedClock

#########################################################################
# SIMULATE
#########################################################################

# Simulate a tree with n leaves
n_leaves = 100
timetree_root = BinaryTreeNode().grow_aldous_tree(n_leaves)

### SIMULATE BRANCH LENGTHS
# Transform the branch lengths into genetic distances
evolutionary_rate = 0.1
sequence_length = 1000
rate_dispersion = 0.0 # Dispersion of the evolutionary rate, controls the noise

refclock = ContinuousAdditiveRelaxedClock(log_rate=torch.tensor(evolutionary_rate).log(), 
                      sequence_length=torch.tensor(sequence_length), 
                      dispersion=torch.tensor(rate_dispersion))

phylotime = timetree_root.edgelist()
torch_edges, torch_durations = phylotime.as_torch()
torch_distances = refclock.simulate(torch_durations)

# EdgeList object with genetic distances (the observed tree)
phylodist = EdgeList(phylotime.edges, torch_distances.tolist())
phylodist.plot()

#########################################################################
# MODEL ENERGY FUNCTION - CONDITIONAL ERROR CLOCK TIP DATING
#########################################################################
from pristine.molecularclock import ConditionalErrorClock
import pristine.distribution as D

class Model:

    def __init__(self, 
                 clock: ConditionalErrorClock, 
                 treecal: TreeTimeCalibrator
                 ):
        self.clock: ConditionalErrorClock = clock
        self.treecal: TreeTimeCalibrator = treecal

    def loss(self)->torch.Tensor:
        durations = self.treecal.durations()
        penalty = D.barrier_positive(durations).sum()
        return -self.clock.log_likelihood(durations, self.treecal.distances).sum() + penalty

#########################################################################
# PREPARE INITIAL GUESS
#########################################################################
from pristine.molecularclock import new_conditional_error_clock

# Recover tip dates from the initial simulated time tree
tip_dates = phylotime.tip_distances_from_root()
# Model structure for tree calibration
treecal: TreeTimeCalibrator = phylodist.get_tree_time_calibrator(tip_dates)
# Basic molecular clock
num_states: float = 4.0
clock: ConditionalErrorClock = new_conditional_error_clock(num_states, sequence_length)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################

model = Model(clock=clock, treecal=treecal)
loss_init = model.loss().item()
optim = pristine.optimize.Optimizer(model)

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={model.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"Evolutionary rate ={clock.rate().item():.3f}, ground truth={evolutionary_rate:.3f}")

node_ages_ref = torch.tensor(phylotime.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

import pristine.plot
pristine.plot.plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")
