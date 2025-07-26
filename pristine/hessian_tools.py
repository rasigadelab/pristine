# -----------------------------------------------------------------------------
# This file is part of the PRISTINE framework for statistical computing
#
# Copyright (C) Jean-Philippe Rasigade
# Hospices Civils de Lyon, and University of Lyon, Lyon, France
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing is available upon request. Please contact the author.
# -----------------------------------------------------------------------------

"""
The `LaplaceEstimator` approximates parameter uncertainty in a differentiable 
model using the Laplace approximation. Mathematically, the idea is that near 
the optimum of a well-behaved loss function (typically a negative log-likelihood), 
the function can be locally approximated by a quadratic. In this regime, the 
posterior distribution over parameters is approximately Gaussian, centered at 
the optimum, with a covariance matrix given by the inverse of the Hessian of 
the loss function.

The main goal of this class is to compute approximate confidence intervals for 
model parameters using this Gaussian approximation. Specifically, it estimates 
the diagonal of the inverse Hessian matrix, which corresponds to the marginal 
variances of individual parameters.

For small models, the class can compute the full Hessian and invert it directly. 
For larger models, this is infeasible due to memory constraints, so the estimator 
falls back to a stochastic method known as Hutchinson’s estimator. In this method, 
the diagonal of the inverse Hessian is approximated using randomized projections 
combined with iterative linear solvers (specifically, conjugate gradient). This 
approach avoids forming the full Hessian but still gives a meaningful estimate 
of uncertainty.

Regularization is included in the dense case to ensure the Hessian is positive 
definite and invertible. In the Hutchinson case, numerical stability and 
convergence are handled through careful control of the conjugate gradient solver.

In summary, this class provides a practical way to compute local uncertainty 
estimates around a model's optimum, supporting both exact and approximate 
second-order methods depending on the size and structure of the model.
"""

import torch
from torch.autograd import grad
from typing import List, Dict, Any, Tuple
from .parameter_tools import ParameterTools

class HessianTools:
    """
    Estimates parameter variance using the Laplace approximation and the diagonal of the inverse Hessian.
    Uses Hutchinson's method combined with conjugate gradient inversion of Hessian-vector products.
    """

    def __init__(self, model: Any):
        self.model = model
        self.pt = ParameterTools(model)  # Holds names, indices, shapes, etc.
        self.params = self.pt.params
        self.dense = False
        self.hutchinson_num_samples = 10
        self.cg_residual_tol = 1e-10
        self.verbose = False

    def hvp(self, loss: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes Hessian-vector product: H @ vec
        where H is the Hessian of the loss w.r.t. model parameters.
        """
        grad1 = grad(loss, self.params, create_graph=True)
        grad1_flat = ParameterTools.flatten_tensor_list(grad1)
        return ParameterTools.flatten_tensor_list(
            grad(grad1_flat @ vec, self.params, retain_graph=True)
        )

    def cg_solve_adaptive(self, loss_fn,
                        vec: torch.Tensor,
                        max_iters: int = 500,
                        batch_size: int = 10,
                        rel_tol: float = 1e-2,
                        abs_tol: float = 1e-3,
                        stall_threshold: float = 1e-6,
                        max_stall_batches: int = 3) -> torch.Tensor:
        """
        Conjugate Gradient with full-vector adaptive convergence based on relative and absolute RMS changes.

        Stops early if:
        - Relative RMS change < rel_tol
        - Absolute RMS change < abs_tol
        - Vector norm remains tiny for max_stall_batches

        Args:
            loss_fn: closure returning scalar loss
            vec: RHS vector v in Hx = v
            max_iters: total iteration cap
            batch_size: iterations per convergence check
            rel_tol: relative RMS change threshold
            abs_tol: absolute RMS change threshold
            stall_threshold: consider x stalled if RMS < this
            max_stall_batches: number of consecutive stalls allowed
            verbose: whether to print diagnostics
        """
        x = torch.zeros_like(vec)
        r = vec.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        last_x = x.clone()
        num_iters = 0
        stall_counter = 0

        while num_iters < max_iters:
            for _ in range(batch_size):
                H_p = self.hvp(loss_fn(), p)
                alpha = rs_old / (torch.dot(p, H_p) + 1e-10)
                x += alpha * p
                r -= alpha * H_p
                rs_new = torch.dot(r, r)
                if torch.sqrt(rs_new) < 1e-10:
                    return x
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
                num_iters += 1
                if num_iters >= max_iters:
                    break

            delta = x - last_x
            rms_change = torch.norm(delta) / (delta.numel() ** 0.5)
            rms_current = torch.norm(x) / (x.numel() ** 0.5)
            rel_change = rms_change / (rms_current + 1e-12)

            if self.verbose:
                print(f".Relative change in CG: {rel_change:.3e}   (rms_change={rms_change:.3e}, rms_current={rms_current:.3e})")

            if rms_current < stall_threshold:
                stall_counter += 1
            else:
                stall_counter = 0

            if stall_counter >= max_stall_batches:
                if self.verbose:
                    print(f"[Warning] CG stagnated: rms_current < {stall_threshold:.1e} for {stall_counter} batches")
                break

            if rms_change < abs_tol or rel_change < rel_tol:
                break

            last_x = x.clone()

        return x

    def compute_dense_hessian(self) -> torch.Tensor:
        loss = self.model.loss()
        grad1 = grad(loss, self.params, create_graph=True)
        grad_flat = self.pt.flatten_tensor_list(grad1)
        dim = grad_flat.numel()
        H = torch.zeros((dim, dim), dtype=grad_flat.dtype, device=grad_flat.device)
        for i in range(dim):
            row_grads = grad(grad_flat[i], self.params, retain_graph=True)
            H[i] = self.pt.flatten_tensor_list(row_grads)
        return H

    def estimate_inv_hessian_diag(self) -> torch.Tensor:
        """
        Estimates the full diagonal of the inverse Hessian
        """
        if self.dense is True:
            return self.estimate_inv_hessian_diag_dense()
        else:
            return self.estimate_inv_hessian_diag_hutchinson()

    def estimate_inv_hessian_diag_dense(self,
                                        lambda_max: float = 1.0,
                                        lambda_step: float = 0.05) -> torch.Tensor:
        """
        Estimate the inverse Hessian diagonal using dense matrix inversion,
        with adaptive regularization to ensure positive-definiteness.

        Args:            
            lambda_max: maximum allowable regularization
            lambda_step: additive increase for lambda

        Returns:
            Diagonal of the inverse of the regularized Hessian
        """
        H = self.compute_dense_hessian()
        diag = torch.diag(H)
        H_diag = torch.diag(diag)

        lam = 0.0

        while lam <= lambda_max:
            H_reg = (1. - lam) * H + lam * H_diag
            try:
                # Try Cholesky to confirm positive-definite
                torch.linalg.cholesky(H_reg)
                Hinv = torch.linalg.inv(H_reg)
                if self.verbose and lam > 0.0:
                    print(f"[Info] Regularization applied: lambda = {lam:.1e}")
                return Hinv.diag()
            except RuntimeError as e:
                if "cholesky" not in str(e).lower():
                    raise  # Something else went wrong
                if self.verbose:
                    print(f"[Warning] Cholesky failed at λ={lam:.1e}, retrying...")
                lam += lambda_step

        raise RuntimeError(f"Failed to regularize Hessian to PD at λ={lam:.1e}")

    def estimate_inv_hessian_diag_hutchinson(self) -> torch.Tensor:
        """
        Estimates the full diagonal of the inverse Hessian using Hutchinson's method.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

        loss = self.model.loss()  # Required; do not remove
        flat_params = self.pt.flatten_parameters()
        dim = flat_params.numel()
        diag_estimate = torch.zeros_like(flat_params)

        for _ in range(self.hutchinson_num_samples):
            v = torch.randint(0, 2, (dim,), dtype=torch.float32, device=flat_params.device) * 2 - 1  # Rademacher
            Hinv_v = self.cg_solve_adaptive(lambda: self.model.loss(), v)
            diag_estimate += v * Hinv_v

        diag_estimate /= self.hutchinson_num_samples
        return diag_estimate

    def estim_all_variances_list(self) -> List[torch.Tensor]:
        """
        Returns unflattened list of Laplace variances corresponding to each model parameter.
        """
        diag_flat = self.estimate_inv_hessian_diag()
        return self.pt.unflatten_vector(diag_flat)

    def estim_all_variances_dict(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary mapping parameter names to their Laplace variances.
        """
        diag_flat = self.estimate_inv_hessian_diag()
        variances = self.pt.unflatten_vector(diag_flat)
        return {name: var for (name, _), var in zip(self.pt.named_params, variances)}

    def estim_variance_by_name(self, name: str) -> torch.Tensor:
        """
        Estimate Laplace variance for a full parameter (returns tensor) or a specific element (returns scalar tensor).
        """
        base, idx = self.pt.parse_name(name)

        flat_params = self.pt.flatten_parameters()
        dim = flat_params.numel()
        v = torch.zeros(dim, dtype=torch.float32, device=self.params[0].device)

        if idx is not None:
            # Scalar or indexed element
            indexed_name = f"{base}[{idx}]"
            full_index = self.pt.get_named_index(indexed_name)
            if full_index >= v.numel():
                raise IndexError(f"Index {full_index} out of range for flattened vector of size {v.numel()} called by {indexed_name}")
            v[full_index] = 1.0
            Hinv_v = self.cg_solve_adaptive(lambda: self.model.loss(), v)
            return torch.tensor(Hinv_v[full_index].item(), device=v.device)

        else:
            # Full tensor case
            flat_indices = [
                (k, i) for k, i in self.pt.name_to_index_map.items()
                if k.startswith(base + "[") and i < dim
            ]
            flat_indices.sort(key=lambda x: x[1])  # ensure order

            values = []
            for _, full_index in flat_indices:
                v.zero_()
                v[full_index] = 1.0
                Hinv_v = self.cg_solve_adaptive(lambda: self.model.loss(), v)
                values.append(Hinv_v[full_index].item())

            # Restore correct shape
            param_tensor = dict(self.pt.named_params)[base]
            return torch.tensor(values, dtype=torch.float32, device=param_tensor.device).view(param_tensor.shape)

    ###############################################################################
    # EIGENANALYSIS
    ###############################################################################

    def extremal_eigenpair(self,
                        which: str = "smallest",  # "largest" or "smallest"
                        max_iters: int = 50,
                        tol: float = 1e-6) -> Tuple[float, torch.Tensor]:
        """
        Estimates an extremal (smallest or largest) eigenvalue and eigenvector of the Hessian.

        Args:
            which (str): Either "smallest" (uses inverse iteration) or "largest" (uses direct power iteration).
            max_iters (int): Maximum number of iterations.
            tol (float): Convergence threshold on the eigenvalue.

        Returns:
            Tuple[float, torch.Tensor]: (eigenvalue, eigenvector)
        """
        assert which in ("smallest", "largest"), "which must be 'smallest' or 'largest'"

        device = self.params[0].device
        dim = self.pt.flatten_parameters().numel()
        v = torch.randn(dim, device=device)
        v /= v.norm()

        lambda_old = 0.0

        for i in range(max_iters):
            if which == "smallest":
                # Solve Hx = v → x ≈ H⁻¹v using CG
                x = self.cg_solve_adaptive(lambda: self.model.loss(), v)
                v_new = x / x.norm()
            else:  # "largest"
                Hv = self.hvp(self.model.loss(), v)
                v_new = Hv / Hv.norm()

            # Rayleigh quotient λ = vᵀ H v
            Hv = self.hvp(self.model.loss(), v_new)
            lambda_new = v_new @ Hv

            if self.verbose:
                print(f"[{i}] λ ({which}) ≈ {lambda_new.item():.6e}")

            if abs(lambda_new - lambda_old) < tol:
                break

            lambda_old = lambda_new
            v = v_new

        return lambda_new.item(), v_new
