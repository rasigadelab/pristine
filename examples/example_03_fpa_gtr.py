#%%
"""
Simulate DNA sequences along a random tree then recover GTR parameters
"""
from __future__ import annotations # For forward references within class definition
import torch

#########################################################################
# SIMULATE
#########################################################################

# Grow a random tree
from pristine.binarytree import BinaryTreeNode

n = 60
root: BinaryTreeNode=BinaryTreeNode().grow_aldous_tree(n)

# Simulate DNA sequence along the random tree
from pristine.gtr import GeneralizedTimeReversibleModel
from pristine.sequence import SequenceSimulationVisitor

# Generate a random 4-state GTR model
num_states = 4
gtr = GeneralizedTimeReversibleModel(num_states)
gtr.rates_log -= 8.0 # Default rates are too fast, signal saturates

# Prepare sequence simulation visitor and apply on all descendents of the root
sequence_length = 100000
visitor = SequenceSimulationVisitor(gtr, sequence_length)
root.bfs(visitor)
       
#########################################################################
# PREPARE INITIAL GUESS - FELSENSTEIN PRUNING ALGORITHM
#########################################################################
from pristine.felsenstein import FelsensteinPruningAlgorithm
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

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################
import pristine.optimize
import time

print("Site patterns:", torch.tensor(markers.unique_patterns.shape).tolist())

# model = Model(fpa=fpa)
loss_init = fpa.loss().item()
optim = pristine.optimize.Optimizer(fpa)
optim.print_interval = 10

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={fpa.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")
print(f"No. of iterations: {optim.num_iter}")

Q, pi = gtr.rate_matrix_stationary_dist()
Q_hat, pi_hat = gtr_optim.rate_matrix_stationary_dist()

# print(pi.tolist())
# print(pi_hat.tolist())

# print(Q)
# print(Q_hat)

from pristine.plot import plot_compare
plot_compare(pi.tolist(), pi_hat.tolist(), "Steady state distribution")
plot_compare(gtr.rates_log.tolist(), gtr_optim.rates_log.tolist(), "Exchange rates")
