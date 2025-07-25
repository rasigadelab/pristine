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
sequence_length = 100
rate_dispersion = 1.0 # Dispersion of the evolutionary rate, controls the noise

phylotime = timetree_root.edgelist()
torch_edges, torch_durations = phylotime.as_torch()
treecal_time = phylotime.get_tree_time_calibrator_fixed()

refclock = ContinuousAdditiveRelaxedClock(
    treecal=treecal_time,
    sequence_length=sequence_length, 
    log_rate=torch.tensor(evolutionary_rate).log(), 
    dispersion=torch.tensor(rate_dispersion))

torch_distances = refclock.simulate()

# EdgeList object with genetic distances (the observed tree)
phylodist = EdgeList(phylotime.edges, torch_distances.tolist())
phylodist.plot()

# import pristine.plot
# pristine.plot.plot_compare(torch_durations, torch_distances / evolutionary_rate, "Node dates, estimated")

#########################################################################
# PREPARE INITIAL GUESS - CONDITIONAL ERROR CLOCK TIP DATING
#########################################################################
from pristine.molecularclock import ConditionalErrorClock

# Recover tip dates from the initial simulated time tree
tip_dates = phylotime.tip_distances_from_root()
# Model structure for tree calibration
treecal: TreeTimeCalibrator = phylodist.get_tree_time_calibrator(tip_dates)
# Basic molecular clock
num_states: float = 4.0
clock = ConditionalErrorClock(
    treecal=treecal,
    num_states=num_states,
    sequence_length=sequence_length
)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################

loss_init = clock.loss().item()
optim = pristine.optimize.Optimizer(clock)

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={clock.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"Evolutionary rate ={clock.rate().item():.3f}, ground truth={evolutionary_rate:.3f}")

node_ages_ref = torch.tensor(phylotime.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

import pristine.plot
pristine.plot.plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")
