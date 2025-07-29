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
# from pristine.gtr import GeneralizedTimeReversibleModel
from pristine.sequence import SequenceSimulationVisitor
from pristine.substitution_models import JC69Model, GTRModel, HKYModel
from pristine.molecularclock import ConstantClock

modeltype = "HKY"

# Generate a random 4-state substitution model
num_states = 4

subst_model_ref = None
if modeltype == "HKY":
    subst_model_ref = HKYModel()
    subst_model_ref.stationary_logits = torch.randn(num_states - 1)
elif modeltype == "JC69":
    subst_model_ref = JC69Model(num_states)
elif modeltype == "GTR":
    subst_model_ref = GTRModel(num_states)
    subst_model_ref.free_rates_log = torch.randn(subst_model_ref.num_free_rates())
    subst_model_ref.stationary_logits = torch.randn(num_states - 1)

clock_ref = ConstantClock(torch.tensor(-8.))

# Prepare sequence simulation visitor and apply on all descendents of the root
sequence_length = 100000
visitor = SequenceSimulationVisitor(clock_ref, subst_model_ref, sequence_length)
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

# Define clock and substitution models
clock = ConstantClock()

subst_model_test = None
if modeltype == "HKY":
    subst_model_test = HKYModel()
elif modeltype == "JC69":
    subst_model_test = JC69Model(num_states)
elif modeltype == "GTR":
    subst_model_test = GTRModel(num_states)

fpa = FelsensteinPruningAlgorithm(subst_model_test, markers, treecal, clock)

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

Q, pi = subst_model_ref.rate_matrix_stationary_dist()
Q_hat, pi_hat = subst_model_test.rate_matrix_stationary_dist()

from pristine.plot import plot_compare
plot_compare(pi.tolist(), pi_hat.tolist(), "Steady state distribution")

def off_diagonal_elements(A: torch.Tensor) -> torch.Tensor:
    """
    Extract all off-diagonal elements of a square matrix as a 1D vector.

    Args:
        A: tensor of shape (K, K), assumed to be square

    Returns:
        Tensor of shape (K*(K-1),) containing all elements except the diagonal
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Input must be a square matrix"
    K = A.shape[0]
    row, col = torch.meshgrid(torch.arange(K), torch.arange(K), indexing="ij")
    mask = row != col
    return A[mask]

plot_compare(
    off_diagonal_elements(Q).tolist(), off_diagonal_elements(Q_hat).tolist(), 
    "Exchange rates")
