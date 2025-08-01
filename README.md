# <span style="color:red">PRISTINE</span> phylodynamics

**PRISTINE** (PRofile-based Inference for STatIstical NEtworks) is a high-performance, differentiable framework for statistical inference on phylogenetic trees. It emphasizes robust likelihood computation, flexible parameter control, dynamic parameter freezing/unfreezing, uncertainty quantification, and model identifiability analysis.

See [PRISTINE documentation](https://rasigadelab.github.io/pristine/) for more details.

**/!\\** The framework is under active development. API may change without warning. **/!\\**

---

## Goals and Features

PRISTINE is designed to:

* Enable efficient likelihood evaluation using Felsenstein's pruning algorithm.
* Support time-calibrated phylogenetic inference via birth-death-sampling (BDS) models and molecular clocks.
* Simulate and preprocess molecular sequence data on trees.
* Handle JC69, HKY, TN93, and GTR substitution models with consistent parameter normalization.
* Offer robust optimization with adaptive backtracking.
* Provide confidence intervals through Laplace approximation or full likelihood profiling.
* Analyze model curvature and identifiability with Hessian eigenvalue diagnostics.
* Support flexible and state-dependent BDS models: constant, multistate, linear, or fully relaxed.
* Serialize and restore model parameters for reproducibility or checkpointing.

---

## Installation

> **Requirements:** Python ≥ 3.10, PyTorch ≥ 2.6

To install PRISTINE and run the examples:

```bash
git clone https://github.com/your-org/pristine.git
cd pristine
pip install .
python examples/run_all.py
```

---

## Example Usage

### 1. Simulate a Tree and Sequences

```python
from pristine.binarytree import BinaryTreeNode
from pristine.sequence import SequenceSimulationVisitor, SequenceCollector, CollapsedConditionalLikelihood
from pristine.substitution_models import GTRModel
from pristine.molecularclock import ConstantClock

tree = BinaryTreeNode().grow_aldous_tree(n=10).reindex()
clock = ConstantClock()
model = GTRModel(4)
visitor = SequenceSimulationVisitor(clock, model, sequence_length=100)
tree.bfs(visitor)

collector = SequenceCollector(tree).collect().erase_node_sequences()
collapsed = CollapsedConditionalLikelihood(collector)
```

### 2. Compute Likelihood with Felsenstein Pruning

```python
from pristine.felsenstein import FelsensteinPruningAlgorithm

treecal = tree.edgelist().get_tree_time_calibrator_fixed()
fpa = FelsensteinPruningAlgorithm(model, collapsed, treecal, clock)
logL = fpa.log_likelihood()
```

### 3. Optimize Model Parameters

```python
from pristine.optimize import Optimizer

opt = Optimizer(fpa)
opt.optimize()
opt.plot_diagnostics()
```

### 4. Estimate Confidence Intervals and Assess Parameter Identifiability

```python
from pristine.likelihood_profiler import LikelihoodProfiler

profiler = LikelihoodProfiler(fpa)
profiler.estimate_confints(["stationary_logits[1]", "free_rates_log[0]"], method="profile")
profiler.curvature_report(top_k=3)
```

---

## Structure

* `felsenstein.py`: Differentiable pruning algorithm for likelihood evaluation
* `sequence.py`: Sequence simulation and pattern collapsing
* `substitution_models.py`: JC69, HKY, TN93, and GTR substitution models
* `molecularclock.py`: Constant and relaxed molecular clock models
* `bds_model.py`: Constant, state-dependent, and relaxed birth-death-sampling models
* `parameter_tools.py`: Parameter flattening, aliasing, and indexing tools
* `optimize.py`: Adaptive optimizer with backtracking and convergence restarts
* `likelihood_profiler.py`: Profile and Laplace-based confidence intervals
* `hessian_tools.py`: Curvature diagnostics and Laplace variance estimators
* `io_tools.py`: Save/load model state to disk
* `distribution.py`: Distributions, barrier functions, JC69 formula
* `binarytree.py`: Tree construction, simulation, and traversal
* `plot.py`: Visualization of parameter calibration and confidence intervals

---

## License

PRISTINE is distributed under the **GNU Affero General Public License v3.0**. For commercial licensing, please contact the author.

---

## Citation

If you use PRISTINE in your research, please cite the framework and acknowledge Jean-Philippe Rasigade, Hospices Civils de Lyon and the University of Lyon.
