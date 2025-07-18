# <span style="color:red">PRISTINE</span> phylodynamics

**PRISTINE** (PRofile-based Inference for STatIstical NEtworks) is a high-performance, differentiable framework for statistical inference on phylogenetic trees. It emphasizes robust likelihood computation, flexible parameter control, changing fixed and free parameters, uncertainty quantification, and model identifiability analysis.

See [PRISTINE documentation](https://rasigadelab.github.io/pristine/) for more details.

---

## Goals and Features

PRISTINE is designed to:

- Enable efficient and differentiable likelihood evaluation using Felsenstein's pruning algorithm.
- Support birth-death-sampling models and molecular clocks for time-calibrated phylogenetic inference.
- Simulate and preprocess molecular sequence data on trees with support for state compression.
- Offer robust parameter optimization with adaptive backtracking to avoid divergence.
- Provide confidence intervals via Laplace approximation and likelihood profiling.
- Analyze model curvature and identifiability via Hessian eigenvalue diagnostics.
- Model evolution using Generalized Time Reversible (GTR) and JC69 substitution matrices.

---

## Installation

> **Requirements:** Python ≥ 3.10, PyTorch ≥ 2.6

You can install PRISTINE locally in development mode:

```bash
git clone https://github.com/your-org/pristine.git
cd pristine
pip install -e .
```

Or via `requirements.txt` (if provided):

```bash
pip install -r requirements.txt
```

---

## Example Usage

### 1. Simulate a Tree and Sequences

```python
from pristine.binarytree import BinaryTreeNode
from pristine.sequence import SequenceSimulationVisitor, SequenceCollector
from pristine.gtr import GeneralizedTimeReversibleModel

tree = BinaryTreeNode().grow_aldous_tree(n=10).reindex()
gtr = GeneralizedTimeReversibleModel(K=4)
sim = SequenceSimulationVisitor(gtr, sequence_length=100)
tree.bfs(sim)

collector = SequenceCollector(tree).collect()
collapsed = collector.erase_node_sequences()
```

### 2. Compute Likelihood with Felsenstein Pruning

```python
from pristine.felsenstein import FelsensteinPruningAlgorithm

fpa = FelsensteinPruningAlgorithm(gtr, collapsed, tree.edgelist().get_tree_time_calibrator_fixed())
logL = fpa.log_likelihood()
```

### 3. Optimize Model Parameters

```python
from pristine.optimize import Optimizer

opt = Optimizer(gtr)
opt.optimize()
```

### 4. Compute Confidence Intervals

```python
from pristine.likelihood_profiler import LikelihoodProfiler

profiler = LikelihoodProfiler(gtr, "stationary_logits[1]")
ci_lower, ci_upper = profiler.estimate_confint()
```

---

## Structure

- `felsenstein.py`: Core differentiable pruning algorithm
- `sequence.py`: Sequence simulation and pattern collapsing
- `laplace_estimator.py`: Uncertainty quantification via curvature
- `likelihood_profiler.py`: Confidence interval estimation by profiling
- `gtr.py`, `molecularclock.py`: Substitution and clock models
- `bds_model.py`: Birth-death-sampling models and extensions
- `parameter_tools.py`: Flat indexing and tensor introspection
- `optimize.py`: Robust optimizer with learning rate backtracking
- `binarytree.py`: General binary tree simulation and traversal
- `distribution.py`: Distribution functions and barrier terms

---

## License

PRISTINE is distributed under the **GNU Affero General Public License v3.0**. For commercial licensing, please contact the author.

---

## Citation

If you use PRISTINE in your research, please cite the framework and acknowledge Jean-Philippe Rasigade, Hospices Civils de Lyon and the University of Lyon.
