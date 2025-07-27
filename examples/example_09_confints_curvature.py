
# %%
import time
import torch
import pandas as pd
from pristine.binarytree import BinaryTreeNode
from pristine.molecularclock import ContinuousAdditiveRelaxedClock, ConditionalErrorClock
from pristine.edgelist import EdgeList
from pristine.optimize import Optimizer
from pristine.likelihood_profiler import LikelihoodProfiler
from pristine.hessian_tools import HessianTools
from pristine.plot import plot_compare_error_bars

#########################################################################
# Simulate phylogenetic tree and branch distances
#########################################################################
n_leaves = 20
tree = BinaryTreeNode().grow_aldous_tree(n_leaves)
edgelist = tree.edgelist()
torch_edges, torch_durations = edgelist.as_torch()
treecal = edgelist.get_tree_time_calibrator_fixed()

evolutionary_rate = 1.0
rate_dispersion = 0.0
sequence_length = 10000

true_clock = ContinuousAdditiveRelaxedClock(
    treecal=treecal,
    sequence_length=sequence_length,
    log_rate=torch.tensor(evolutionary_rate).log(),
    dispersion=torch.tensor(rate_dispersion))

torch_distances = true_clock.simulate()
phylodist = EdgeList(edgelist.edges, torch_distances.tolist())
tip_dates = edgelist.tip_distances_from_root()

treecal = phylodist.get_tree_time_calibrator(tip_dates)
clock = ConditionalErrorClock(treecal=treecal, num_states=4, sequence_length=sequence_length)

#########################################################################
# Optimize model
#########################################################################
opt = Optimizer(clock)
opt.optimize()

#########################################################################
# Confidence Intervals: Laplace
#########################################################################
profiler = LikelihoodProfiler(clock)

L_lap, U_lap = profiler.estimate_confint_scalar_laplace("log_rate")
L_prof, U_prof = profiler.estimate_confint_scalar_profile("log_rate")

print(f"\nOptimized log_rate = {clock.log_rate.item(): .2e}")
print(f"Log rate CI, Laplace, [{L_lap: .2e}, {U_lap: .2e}]")
print(f"Log rate CI, profile, [{L_prof: .2e}, {U_prof: .2e}]")

profiler.estimate_confints("log_rate", method="laplace")
# profiler.estimate_µconfints(["log_rate"], method="profile")

profiler.estimate_confints_parallel(method="laplace")
# profiler.estimate_all_confints_laplace(dense=False, num_samples=100)

profiler.estimate_all_confints_laplace()

#########################################################################
# Curvature report
#########################################################################

_ = profiler.curvature_report(top_k=3)

# %%
