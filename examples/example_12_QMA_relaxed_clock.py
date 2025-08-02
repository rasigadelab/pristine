# %%
"""
Generic QMA integration for the Gamma clock model. This example compares the
cARC model, which relaxes the Gamma(rate*duration, 1) strict clock with a
dispersion parameter, with the QMA-relaxed clock that integrates over the
rate at each edge, using a LogNormal prior
"""
import time
import torch
import pristine
import pristine.optimize
from pristine.binarytree import BinaryTreeNode
from pristine.molecularclock import ContinuousAdditiveRelaxedClock, RelaxedGammaClock
from pristine.edgelist import EdgeList, TreeTimeCalibrator
from pristine.qma import QMAFactory

#########################################################################
# SIMULATE WITH RELAXED GAMMA
#########################################################################

# Simulate a tree with n leaves
n_leaves = 200
timetree_root = BinaryTreeNode().grow_aldous_tree(n_leaves)

### SIMULATE BRANCH LENGTHS
# Transform the branch lengths into genetic distances
evolutionary_rate = 0.7
num_markers = 100
log_rate_std = 0.1 # Dispersion of the evolutionary rate, controls the noise

phylotime = timetree_root.edgelist()
torch_edges, torch_durations = phylotime.as_torch()
treecal_time = phylotime.get_tree_time_calibrator_fixed()

refclock = RelaxedGammaClock(
    treecal=treecal_time,
    num_markers=num_markers, 
    log_rate_mean=torch.tensor(evolutionary_rate).log(), 
    log_rate_log_std=torch.tensor(log_rate_std).log())

torch_distances = refclock.simulate()

# EdgeList object with genetic distances (the observed tree)
phylodist = EdgeList(phylotime.edges, torch_distances.tolist())
# phylodist.plot()

# pristine.plot.plot_compare(torch_durations, torch_distances / evolutionary_rate, "Node dates, estimated")
#########################################################################
# cARC TIP DATING for reference
#########################################################################

# Recover tip dates from the initial simulated time tree
tip_dates = phylotime.tip_distances_from_root()
# Model structure for tree calibration
treecal: TreeTimeCalibrator = phylodist.get_tree_time_calibrator(tip_dates)
# Basic molecular clock
cARC_clock = ContinuousAdditiveRelaxedClock(
    treecal=treecal,
    sequence_length=num_markers)

loss_init = cARC_clock.loss().item()
optim = pristine.optimize.Optimizer(cARC_clock)
# optim.reset_interval = 100

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("cARC Model:")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={cARC_clock.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"Evolutionary rate ={cARC_clock.rate().item():.3f}, ground truth={evolutionary_rate:.3f}")
print(f"Dispersion rate   ={cARC_clock.dispersion.item():.3f}")

node_ages_ref = torch.tensor(phylotime.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

import pristine.plot
# pristine.plot.plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")

optim.plot_diagnostics()

#########################################################################
# RELAXED GAMMA TIP DATING
#########################################################################

# 1. Plot node positions and weights for illustration
import matplotlib.pyplot as plt

# Generate grids
factory = QMAFactory(num_nodes=16)
uni = factory.uniform()
pw = factory.warped(gamma=6)
gh = factory.hermite()

uni_nodes, uni_weights = uni.get_grid()
pw_nodes, pw_weights = pw.get_grid()
gh_nodes, gh_weights = gh.get_grid()

# Plot all QMA grids using consistent style
plt.figure(figsize=(8, 3))
plt.stem(uni_nodes.numpy(), uni_weights.numpy(), linefmt='C0-', markerfmt='C0o', basefmt=' ', label="Uniform QMA")
plt.stem(pw_nodes.numpy(), pw_weights.numpy(), linefmt='C1-', markerfmt='C1s', basefmt=' ', label="Power warp (γ=6)")
plt.stem(gh_nodes.numpy(), gh_weights.numpy(), linefmt='C2-', markerfmt='C2^', basefmt=' ', label="Gauss-Hermite QMA")

plt.title("QMA Node Positions and Weights (16 nodes)")
plt.xlabel("Node position")
plt.ylabel("Weight")
plt.legend()
plt.tight_layout()
plt.draw()

# 2. Rebuild model components and use the relaxed gamma clock

# Model structure for tree calibration
treecal: TreeTimeCalibrator = phylodist.get_tree_time_calibrator(tip_dates)

# Basic molecular clock
qma_clock = RelaxedGammaClock(
    treecal=treecal,
    num_markers=num_markers,
    qma_grid=QMAFactory(16).hermite()
    )

loss_init = qma_clock.loss().item()
optim = pristine.optimize.Optimizer(qma_clock)
# optim.reset_interval = 100

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("QMA/Gamma model:")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={qma_clock.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"Average rate = {qma_clock.log_rate_mean.exp():.3f}, ground truth={evolutionary_rate:.3f}")
print(f"Log std = {qma_clock.log_rate_log_std.exp():.3f}, ground truth={log_rate_std:.3f}")

node_ages_ref = torch.tensor(phylotime.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

import pristine.plot
# pristine.plot.plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")

optim.plot_diagnostics()
# %%
