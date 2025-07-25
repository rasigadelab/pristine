#%%
"""
Tree-wise BDS
"""
import torch

#########################################################################
# SIMULATE
#########################################################################
print("Simulating data...")
from pristine.binarytree import BinaryTreeNode
from pristine.bds_model import BirthDeathSamplingNodeData, BirthDeathSamplingSimulator

# A forward model object exposes a simulate(node) function to generate
# data or parameters during forward simulation. Here we use the 
# BirthDeathSamplingNodeData base class to provide constant BDS parameters.
class ConstantBDSVisitor:
    """
    Implement functions to generate data and BDS parameters
    along a tree so that data can influence tree growth.
    """
    def __init__(self):
        pass
    
    def visit(self, node: BinaryTreeNode)->None:
        node.data = BirthDeathSamplingNodeData(2.0, 0.0, 1.0)

root: BinaryTreeNode=BinaryTreeNode()
bdsforward = BirthDeathSamplingSimulator(root, ConstantBDSVisitor())

n = 100
root = bdsforward.simulate(n)
root.visualize()

#########################################################################
# PREPARE INITIAL GUESS
#########################################################################
from pristine.edgelist import TreeTimeCalibrator
from pristine.bds_model import ConstantBirthDeathSamplingModel
treecal = root.edgelist().get_tree_time_calibrator_fixed()
bds = ConstantBirthDeathSamplingModel(treecal)

#########################################################################
# OPTIMIZE AND PRINT RESULTS
#########################################################################

import pristine.optimize
import time

loss_init = bds.loss().item()
optim = pristine.optimize.Optimizer(bds)

start = time.perf_counter()
optim.optimize()
stop = time.perf_counter()

print("")
print(f"Initial loss: {loss_init: .3e}")
print(f"Final loss={bds.loss().item():.3e}")
print(f"Elapsed time: {stop - start:.3f}s")

print("")
print(f"Birth rate    = {bds.birth().item(): 0.2f}, ground truth = {root.data.birth_rate}")
print(f"Death rate    = {bds.death().item(): 0.2f}, ground truth = {root.data.death_rate}")
print(f"Sampling rate = {bds.sampling().item(): 0.2f}, ground truth = {root.data.sampling_rate}")
# %%
