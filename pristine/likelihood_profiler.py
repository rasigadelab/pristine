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
The LikelihoodProfiler class estimates confidence intervals for model
parameters using a technique called likelihood profiling. Rather than
relying solely on local curvature (as in the Laplace approximation), it
perturbs a selected parameter, re-optimizes the model for each value, and
tracks how the likelihood changes. The confidence bounds are identified as
the parameter values where the log-likelihood falls below a predefined
threshold from its maximum.

This method provides a more reliable estimate of uncertainty than
second-order approximations, particularly when the likelihood surface is
non-quadratic or asymmetric. It is especially useful in maximum-likelihood
settings where parameters may be poorly constrained or interdependent.

The class includes robust fallbacks: if profiling fails (e.g., due to
optimization instability or bracketing issues), it returns an interval based
on the Laplace approximation. It also supports profiling multiple parameters
in sequence or in parallel, returning confidence intervals in a structured
format suitable for downstream use.
"""

import torch
import copy
import warnings
import scipy.stats as stats
import pandas as pd
from typing import Tuple, List
from .laplace_estimator import LaplaceEstimator
from .parameter_tools import ParameterTools, TensorAccessor
from .optimize import Optimizer
from scipy.optimize import brentq
from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing as mp
import os
from tqdm import tqdm
from scipy.stats import norm

class LikelihoodProfiler:
    """
    Refines confidence intervals using likelihood profiling.
    """
    def __init__(self, model, name_with_index: str = None):
        self.model = model
        self.pt = ParameterTools(model)
        self.grid_points = 21

        if name_with_index is not None:
            resolved_name = self.pt.normalize_name(name_with_index)
            self.name = resolved_name
            indexed_name = self.pt.normalize_name(resolved_name)
            self.name = indexed_name
            self.index = self.pt.get_named_index(indexed_name)

            self.param_tensor = None
            self.tensor_idx = None
            self._resolve_tensor()
            self.accessor = TensorAccessor(self.param_tensor, self.tensor_idx)
        else:
            self.name = None
            self.index = None
            self.param_tensor = None
            self.tensor_idx = None
            self.accessor = None

    def _resolve_tensor(self):
        """
        Store the actual tensor and the local index for the named parameter.
        """
        name = self.name.split("[")[0]
        self.param_tensor = dict(self.pt.named_params)[name]
        idx_match = torch.tensor(eval(self.name[self.name.index("["):]))  # parse [3] or [1,2]
        self.tensor_idx = tuple(idx_match.tolist()) if idx_match.ndim > 0 else int(idx_match)

    def _copy_model(self):
        """
        Return a deep copy of the model to avoid modifying the original.
        """
        return copy.deepcopy(self.model)

    def set_target_parameter(self, name_with_index: str):
        """
        Set the parameter to profile after construction.
        Allows switching parameters interactively.
        """
        resolved = self.pt.normalize_name(name_with_index)
        self.name = resolved
        self.index = self.pt.get_named_index(resolved)

        base = self.name.split("[")[0]
        self.param_tensor = dict(self.pt.named_params)[base]
        idx_match = torch.tensor(eval(self.name[self.name.index("["):]))
        self.tensor_idx = tuple(idx_match.tolist()) if idx_match.ndim > 0 else int(idx_match)
        self.accessor = TensorAccessor(self.param_tensor, self.tensor_idx)

    def profile__(self, range_factor: float = 2.0, tol: float = 1.92, maxiter: int = 20, max_expansions: int = 5) -> Tuple[float, float]:
        """
        Estimate lower and upper bound for a 95% CI by profiling log-likelihood using Brent's method.
        If the root is not found in the initial bracket, the interval is doubled up to max_expansions times.
        """
        lap = LaplaceEstimator(self.model)
        center = self.accessor.get()
        stddev = lap.estim_variance_by_name(self.name).sqrt().item()
        ll_center = self.model.loss().item()

        def make_objective():
            def objective(v):
                model_copy = self._copy_model()
                profiler = LikelihoodProfiler(model_copy, self.name)
                profiler.accessor.set(v)
                optimizer = Optimizer(model_copy)
                optimizer.optimize()
                profiler.accessor.zero_grad()
                ll = model_copy.loss().item()
                return ll - (ll_center + tol)
            return objective

        def search_bound(side: str) -> float:
            """
            Attempts to find the bound (lower or upper) using Brent's method.
            Expands the bracket adaptively if necessary.
            """
            direction = -1 if side == "lower" else 1
            span = stddev * range_factor
            objective = make_objective()

            for i in range(max_expansions):
                a = center + direction * span
                b = center
                bracket = (a, b) if side == "lower" else (b, a)

                try:
                    return brentq(objective, *bracket, maxiter=maxiter)
                except ValueError:
                    span *= 2  # Expand bracket and retry

            warnings.warn(f"{side.capitalize()} bound: bracketing failed after {max_expansions} expansions.")
            return center  # Fallback: return center if all attempts fail

        lower = search_bound("lower")
        upper = search_bound("upper")
        return lower, upper

    def profile(self, range_factor: float = 2.0, tol: float = 1.92) -> Tuple[List[float], List[float]]:
        """
        Estimate lower and upper bound for a 95% CI by profiling log-likelihood.
        """
        # Step 1: Estimate MLE and CI from Laplace
        lap = LaplaceEstimator(self.model)
        center = self.accessor.get()
        stddev = lap.estim_variance_by_name(self.name).sqrt().item()
        span = stddev * range_factor
        profile_points = torch.linspace(center - span, center + span, steps=self.grid_points)

        # Step 2: Evaluate profile likelihoods
        ll_center = self.model.loss().item()
        log_likelihoods = []

        for v in profile_points:
            model_copy = self._copy_model()
            profiler = LikelihoodProfiler(model_copy, self.name)
            profiler.accessor.set(v)
            optimizer = Optimizer(model_copy)
            optimizer.optimize()
            profiler.accessor.zero_grad()
            log_likelihoods.append(model_copy.loss().item())

        # Step 3: Determine bounds from likelihood threshold
        diffs = [ll - ll_center for ll in log_likelihoods]
        lower = upper = center

        for x, d in zip(profile_points, diffs):
            if d < tol and x < center:
                lower = x.item()
            if d < tol and x > center:
                upper = x.item()

        return lower, upper

    def estimate_confint(self, confidence_level=0.95, range_factor=3.0) -> Tuple[float, float]:
        """
        Estimate confidence interval using likelihood profiling.
        Falls back to Laplace approximation if profiling fails.

        Parameters:
        - confidence_level: confidence level for the interval (default 0.95)
        - num_points: number of profiling points to try
        - range_factor: span in stddev units to explore
        """
        if self.name is None:
            first_param, _ = self.pt.named_params[0]
            self.set_target_parameter(f"{first_param}[0]")
            warnings.warn(f"[Info] No parameter name provided. Defaulting to first parameter: {self.name}")
        
        # For profiling: use chi-square threshold for 1 DOF
        chi2_threshold = stats.chi2.ppf(confidence_level, df=1)
        delta_logL = 0.5 * chi2_threshold

        try:
            lower, upper = self.profile(
                # num_points=num_points,
                range_factor=range_factor,
                tol=delta_logL
            )

            if lower == upper:
                raise RuntimeError("Profile bounds did not diverge from center.")

            return lower, upper

        except Exception as e:
            warnings.warn(f"[Fallback] Likelihood profiling failed: {str(e)}\n"
                          f"Returning Laplace-based symmetric interval.")

            from .laplace_estimator import LaplaceEstimator
            lap = LaplaceEstimator(self.model)
            center = self.accessor.get()
            std = lap.estim_variance_by_name(self.name).sqrt().item()
            z_score = stats.norm.ppf(0.5 + confidence_level / 2)
            delta = z_score * std
            return center - delta, center + delta

    def estimate_confint_all(self, names: List[str] = None, confidence_level=0.95,
                              range_factor=3.0) -> pd.DataFrame:
        """
        Estimate confidence intervals via profiling for multiple parameters.

        Returns a DataFrame with columns:
            name | center | lower | upper | width | used_fallback
        """
        if names is None:
            names = []
            for pname, tensor in self.pt.named_params:
                if tensor.ndim == 0:
                    names.append(f"{pname}[0]")
                else:
                    for i in range(tensor.numel()):
                        names.append(f"{pname}[{i}]")

        records = []
        for name in names:
            print(f"\nProfiling: {name}")
            model_copy = copy.deepcopy(self.model)
            profiler = LikelihoodProfiler(model_copy, name)
            center = profiler.accessor.get()
            try:
                ci_low, ci_high = profiler.estimate_confint(
                    confidence_level=confidence_level,
                    range_factor=range_factor
                )
                used_fallback = False
            except Exception as e:
                warnings.warn(f"[Fallback] {name}: {e}")
                from .laplace_estimator import LaplaceEstimator
                lap = LaplaceEstimator(profiler.model)
                var = lap.estim_variance_by_name(name)
                if not torch.is_tensor(var):
                    var = torch.tensor(var)
                std = var.sqrt().item()
                from scipy.stats import norm
                z = norm.ppf(0.5 + confidence_level / 2)
                delta = z * std
                ci_low, ci_high = center - delta, center + delta
                used_fallback = True

            records.append({
                "name": name,
                "lower": ci_low,
                "center": center,
                "upper": ci_high,
                "used_fallback": used_fallback
            })

        return pd.DataFrame(records)

    def estimate_confint_all_parallel(self, names: List[str] = None, confidence_level=0.95,
                                    range_factor=3.0, max_workers: int = 16) -> pd.DataFrame:
        """
        Parallelized version of estimate_confint_all using ThreadPoolExecutor.

        Parameters:
            names (List[str]): List of parameter names (e.g. 'clock.log_rate[0]').
            confidence_level (float): Confidence level for intervals.
            range_factor (float): Range in stddev units for profiling.
            max_workers (int): Number of threads.

        Returns:
            pd.DataFrame with columns: name | lower | center | upper | laplace
        """
        if names is None:
            names = []
            for pname, tensor in self.pt.named_params:
                if tensor.ndim == 0:
                    names.append(f"{pname}[0]")
                else:
                    for i in range(tensor.numel()):
                        names.append(f"{pname}[{i}]")

        def profile_one(name: str) -> dict:
            model_copy = self._copy_model()
            profiler = LikelihoodProfiler(model_copy, name)
            center = profiler.accessor.get()
            try:
                ci_low, ci_high = profiler.estimate_confint(
                    confidence_level=confidence_level,
                    range_factor=range_factor
                )
                used_fallback = False
            except Exception as e:
                from .laplace_estimator import LaplaceEstimator
                lap = LaplaceEstimator(profiler.model)
                var = lap.estim_variance_by_name(name)
                std = torch.tensor(var).sqrt().item()
                from scipy.stats import norm
                z = norm.ppf(0.5 + confidence_level / 2)
                delta = z * std
                ci_low, ci_high = center - delta, center + delta
                used_fallback = True
            return {
                "name": name,
                "lower": ci_low,
                "center": center,
                "upper": ci_high,
                "laplace": used_fallback
            }

        records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(profile_one, name): name for name in names}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Profiling"):
                records.append(future.result())

        df = pd.DataFrame(records)
        df['order'] = df['name'].map({name: i for i, name in enumerate(names)})
        return df.sort_values("order").drop(columns="order").reset_index(drop=True)

    def estimate_confint_all_laplace(self, confidence_level=0.95, dense: bool = True, num_samples = 20) -> pd.DataFrame:
        """
        Compute confidence intervals using the Laplace approximation
        via the full dense inverse Hessian (high accuracy, complexity O(n^3)).
        
        Parameters:
            confidence_level: confidence level (default 0.95)
            dense:  if True (default), use full dense Hessian inverse (accurate, slower); 
                    if False, use Hutchinson estimator (faster, approximate)       
        Returns:
            pd.DataFrame with columns: name | lower | center | upper | laplace
        """

        lap = LaplaceEstimator(self.model)
        lap.dense = dense  # Force dense Hessian inversion
        lap.hutchinson_num_samples = num_samples
        variances = lap.estim_all_variances_dict()
        pt = ParameterTools(self.model)

        z_score = norm.ppf(0.5 + confidence_level / 2)

        records = []
        for name, tensor in pt.named_params:
            center_vals = tensor.detach().view(-1)
            stddev_vals = variances[name].sqrt().view(-1)
            for i in range(center_vals.numel()):
                param_name = f"{name}[{i}]"
                center = center_vals[i].item()
                std = stddev_vals[i].item()
                delta = z_score * std
                records.append({
                    "name": param_name,
                    "lower": center - delta,
                    "center": center,
                    "upper": center + delta,
                    "laplace": True  # Always fallback, no profiling
                })

        return pd.DataFrame(records)
