# %%
"""
Example of modular design for tree dating and molecular clock estimation.
Uses the cARC model of Didelot et al. Mol Biol Evol 2021.
"""
import time
import torch
import pristine
import pristine.optimize
from pristine.binarytree import BinaryTreeNode
from pristine.molecularclock import ContinuousAdditiveRelaxedClock, new_continuous_additive_relaxed_clock
from pristine.edgelist import EdgeList, TreeTimeCalibrator

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
# MODEL ENERGY FUNCTION - cARC TIP DATING
#########################################################################

class Model:

    def __init__(self, 
                 clock: ContinuousAdditiveRelaxedClock, 
                 treecal: TreeTimeCalibrator
                 ):
        self.clock: ContinuousAdditiveRelaxedClock = clock
        self.treecal: TreeTimeCalibrator = treecal

    def loss(self)->torch.Tensor:
        durations = self.treecal.durations()        
        return -self.clock.log_likelihood(durations, self.treecal.distances).sum() #+ penalty

#########################################################################
# PREPARE INITIAL GUESS
#########################################################################

# Recover tip dates from the initial simulated time tree
tip_dates = phylotime.tip_distances_from_root()
# Model structure for tree calibration
treecal: TreeTimeCalibrator = phylodist.get_tree_time_calibrator(tip_dates)
# Basic molecular clock
clock: ContinuousAdditiveRelaxedClock = new_continuous_additive_relaxed_clock(sequence_length)

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
print(f"Dispersion rate   ={clock.dispersion.item():.3f}, ground truth={rate_dispersion:.3f}")

node_ages_ref = torch.tensor(phylotime.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

import pristine.plot
pristine.plot.plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")
