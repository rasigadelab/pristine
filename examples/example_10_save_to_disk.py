#%%
import time
import torch
import pristine
import pristine.optimize
import pristine.plot
from pristine.binarytree import BinaryTreeNode
from pristine.molecularclock import ContinuousAdditiveRelaxedClock
from pristine.edgelist import EdgeList, TreeTimeCalibrator
from pristine.io_tools import IOTools
#########################################################################
# SIMULATE
#########################################################################

# Simulate a tree with n leaves
n_leaves = 100
timetree_root = BinaryTreeNode().grow_aldous_tree(n_leaves)

### SIMULATE BRANCH LENGTHS
# Transform the branch lengths into genetic distances
evolutionary_rate = 3.
sequence_length = 1000
rate_dispersion = 3. # Dispersion of the evolutionary rate, controls the noise

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

pristine.plot.plot_compare(torch_durations, torch_distances / evolutionary_rate, "Node dates, estimated")

#########################################################################
# PREPARE INITIAL GUESS - cARC TIP DATING
#########################################################################

# Recover tip dates from the initial simulated time tree
tip_dates = phylotime.tip_distances_from_root()
# Model structure for tree calibration
treecal: TreeTimeCalibrator = phylodist.get_tree_time_calibrator(tip_dates)
# Basic molecular clock
clock = ContinuousAdditiveRelaxedClock(
    treecal=treecal,
    sequence_length=sequence_length)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################

io = IOTools(clock, "clock.pt", trainable_only=False)
io.save()

# model = Model(clock=clock, treecal=treecal)
loss_init = clock.loss().item()
optim = pristine.optimize.Optimizer(clock)
# optim.reset_interval = 100

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={clock.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"Evolutionary rate ={clock.rate().item():.3f}, ground truth={evolutionary_rate:.3f}")
print(f"Dispersion rate   ={clock.dispersion.item():.3f}, ground truth={rate_dispersion:.3f}")

node_ages_ref = torch.tensor(phylotime.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

import pristine.plot
pristine.plot.plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")

assert clock.loss().item() < loss_init # Optimized state
io.load()
assert clock.loss().item() == loss_init # Restored original state