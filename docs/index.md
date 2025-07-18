# PRISTINE: Profile-based Inference for Statistical Networks

PRISTINE is a modular framework for phylogenetic likelihood inference, built in PyTorch ≥ 2.6 and Python ≥ 3.10.

It supports:
- Efficient likelihood computation with Felsenstein's pruning algorithm
- Model-based simulation and inference (e.g., GTR, BDS)
- Parameter profiling and Laplace approximation
- Relaxed and strict molecular clocks
- Transparent and flexible parameter management

---

## 📌 Quick Start

1. Install locally:

```bash
pip install -e .
```

2. Run a minimal inference:
```python
from pristine import optimize, binarytree, gtr
...
```

3. See [tutorials](tutorials/inference.md) and [covered scientific questions](math/pristine_overview.md)

---

## 📦 Components

| Module                   | Description                                 |
|--------------------------|---------------------------------------------|
| `felsenstein.py`         | FPA-based log-likelihood engine             |
| `gtr.py`                 | GTR substitution model                      |
| `sequence.py`            | Sequence simulation and pattern collapse    |
| `molecularclock.py`      | Strict and relaxed clock models             |
| `laplace_estimator.py`   | Laplace approximation of uncertainty        |
| `likelihood_profiler.py` | Confidence intervals via profiling          |
| `binarytree.py`          | Tree structure and simulation               |
| `edgelist.py`            | Time-aware representation of trees          |
