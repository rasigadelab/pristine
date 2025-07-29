#%%
"""
Simulate DNA sequences along a random tree then recover GTR parameters and node dates.
Practically, discard fixed edge lengths in 'treecal' object and make ages learnable

Regarding torch.compile:
cmd /k C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat

import os
os.environ["CXX"] = "cl"
"""
from __future__ import annotations # For forward references within class definition
import torch

#########################################################################
# SIMULATE
#########################################################################
print("Simulating data...")

# Grow a random tree
from pristine.binarytree import BinaryTreeNode

n = 40
root: BinaryTreeNode=BinaryTreeNode().grow_aldous_tree(n)

# Simulate DNA sequence along the random tree
from pristine.sequence import SequenceSimulationVisitor
from pristine.molecularclock import ConstantClock
from pristine.substitution_models import GTRModel

# Generate a random 4-state GTR model
num_states = 4
subst_model_ref = GTRModel(num_states)
subst_model_ref.free_rates_log = torch.randn(subst_model_ref.num_free_rates())
subst_model_ref.stationary_logits = torch.randn(num_states - 1)

clock_ref = ConstantClock(torch.tensor(-8.))

# Prepare sequence simulation visitor and apply on all descendents of the root
sequence_length = 100000
visitor = SequenceSimulationVisitor(clock_ref, subst_model_ref, sequence_length)
root.bfs(visitor)
       
#########################################################################
# PREPARE INITIAL GUESS INCLUDING AGES
#########################################################################
print("Preparing starting conditions...")
from pristine.sequence import SequenceCollector, CollapsedConditionalLikelihood
from pristine.edgelist import TreeTimeCalibrator
from pristine.felsenstein import FelsensteinPruningAlgorithm

### Collate unique sites into a sequence structure. Erase ancestor sequences
# to mimic application.
collector = SequenceCollector(root).collect().erase_node_sequences()
markers = CollapsedConditionalLikelihood(collector.sequences)
print("Site patterns:", torch.tensor(markers.unique_patterns.shape).tolist())

# Consolidated tree structure with edges etc. Collect tip dates
# then construct a 'treecal' object with learnable ages
edgelist = root.edgelist()
tip_dates = edgelist.tip_distances_from_root()
treecal: TreeTimeCalibrator = edgelist.get_tree_time_calibrator(tip_dates)

# Define clock and substitution models
clock = ConstantClock()
clock.log_rate = torch.tensor(-5.).requires_grad_()
subst_model_test = GTRModel(num_states)

fpa = FelsensteinPruningAlgorithm(subst_model_test, markers, treecal, clock)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################
print("Launching optimizer...")
import pristine.optimize
import time

loss_init = fpa.loss().item()
optim = pristine.optimize.Optimizer(fpa)
optim.print_interval = 10
optim.max_iterations = 10000

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={fpa.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"No. of iterations: {optim.num_iter}")

Q, pi = subst_model_ref.rate_matrix_stationary_dist()
Q_hat, pi_hat = subst_model_test.rate_matrix_stationary_dist()

from pristine.plot import plot_compare
plot_compare(pi.tolist(), pi_hat.tolist(), "Steady state distribution")
plot_compare(subst_model_ref.free_rates_log.tolist(), subst_model_test.free_rates_log.tolist(), "Exchange rates")

# Display node ages
node_ages_ref = torch.tensor(edgelist.distances_from_root())[treecal.node_indices]
node_ages_estim = treecal.node_dates.tolist()

plot_compare(node_ages_ref, node_ages_estim, "Node dates, estimated")
