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
import math
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
    def __init__(self, model):
        self.model = model
        self.pt = ParameterTools(model)
        self.range_factor: float = 2
        self.reoptimize: bool = True
        self.max_expand: int = 8
        self.max_workers: int = 16

    def _copy_model(self):
        """
        Return a deep copy of the model to avoid modifying the original.
        """
        return copy.deepcopy(self.model)

    ############################################################################
    # BRACKETING LIKELIHOOD PROFILING
    ############################################################################

    @staticmethod
    def optimize_with_fixed_parameters(model: object,
                                    fixed_params: List[Tuple[str, float]],
                                    reoptimize: bool = True,
                                    optimizer_args: dict = None
                                    ) -> Tuple[object, float]:
        """
        Fix specified parameters, optionally optimize all others, and return loss.
        Model is modified, use a deep-copy if required.

        Args:
            model: A model object with a .loss() method and optimizable parameters.
            fixed_params: A list of (name_with_index, value) to fix.
                        E.g., [("substitution_model.rates_log[0]", 1.0)]
            reoptimize: optimize the remaining free parameters (likelihood profiling)
            optimizer_args: Optional dict passed to Optimizer constructor. Ignored if
                reoptimize is False

        Returns:
            final_loss_value
        """
        # model_copy = copy.deepcopy(model)
        pt = ParameterTools(model)

        # Fix the parameters by disabling gradient and setting value
        for name, value in fixed_params:
            base, idx = pt.parse_name(name)
            param = dict(pt.named_params)[base]
            accessor = TensorAccessor(param, idx)
            accessor.set(value)

            if param.requires_grad and reoptimize is True:
                if idx is None or param.ndim == 0: # Scalar or full tensor
                    param.requires_grad_(False)
                else:
                    def hook(grad):
                            grad = grad.clone()
                            grad[idx] = 0.0
                            return grad
                    param.register_hook(hook)

        if reoptimize:
            # Optimize the rest
            opt = Optimizer(model, **(optimizer_args or {}))
            opt.optimize()

        return model.loss().item()
    

    def profile_brent(self, name: str, tol: float = 1.92) -> Tuple[float, float]:
        """
        Estimate a confidence interval for a scalar parameter using likelihood profiling
        and Brent's root-finding method.

        This method identifies the values of a target parameter at which the negative
        log-likelihood increases by a specified threshold (`tol`) from its minimum.
        It reoptimizes all other model parameters while keeping the target parameter fixed,
        using Brent's algorithm to solve for the interval endpoints.

        If the initial search bracket does not contain a root (i.e., the profile likelihood
        does not cross the threshold), the bracket is expanded geometrically up to 
        `max_expand` times. If no root is found, the corresponding bound is set to infinity.

        Parameters:
            reoptimize (bool): Whether to reoptimize other parameters at each fixed value
                            of the profiled parameter. Should be True for valid profiling.
            range_factor (float): Number of Laplace-based standard deviations to use for
                                the initial search bracket around the MLE.
            tol (float): The log-likelihood drop (Δℓ) defining the confidence bound.
                        For a 95% CI, use 1.92 (≈0.5·χ²₀.₉₅,₁).
            max_expand (int): Maximum number of bracket-doubling steps if the root is not
                            bracketed initially.

        Returns:
            Tuple[float, float]: The lower and upper bounds of the confidence interval
                                for the profiled parameter. If a bound could not be
                                determined, it will be set to ±∞.

        Notes:
            - This method assumes the model defines a `.loss()` method returning
            the negative log-likelihood, and uses a Laplace approximation to estimate
            the standard deviation of the target parameter.
        """
        # Step 1: Get MLE point and stddev from Laplace
        lap = LaplaceEstimator(self.model)
        # center = self.accessor.get()
        center = self.pt.get_accessor(name).get()
        stddev = lap.estim_variance_by_name(name).sqrt().item()
        span = stddev * self.range_factor
        loss_mle = self.model.loss().item()

        # Define objective: zero when loss crosses the profiling threshold
        def loss_diff(value: float) -> float:
            fixed = [(name, value)]
            modcopy = copy.deepcopy(self.model)
            loss = LikelihoodProfiler.optimize_with_fixed_parameters(
                modcopy,
                fixed_params=fixed,
                reoptimize=self.reoptimize,
                optimizer_args={"initial_lr": 0.1, "max_iterations": 500}
            )
            return loss - loss_mle - tol

        def find_bound(direction: str) -> float:
            sign = -1.0 if direction == "lower" else 1.0
            a = center
            b = center + sign * span
            for i in range(self.max_expand):
                fa = loss_diff(a)
                fb = loss_diff(b)
                if fa * fb < 0:
                    # Proper bracket found
                    return brentq(loss_diff, a, b, xtol=1e-3)
                # Expand the bracket
                b = center + sign * (2 ** (i + 1)) * span
            return float("inf") * sign

        lower = find_bound("lower")
        upper = find_bound("upper")
        return lower, upper

    @staticmethod
    def get_delta_log_likelihood(confidence_level=0.95, dof=1)->float:
        # For profiling: use chi-square threshold
        chi2_threshold = stats.chi2.ppf(confidence_level, df=1)
        delta_logL = 0.5 * chi2_threshold
        return delta_logL

    def estimate_confint(self, name: str, confidence_level=0.95) -> Tuple[float, float]:
        """
        Estimate confidence interval using likelihood profiling.
        Falls back to Laplace approximation if profiling fails.

        Parameters:
        - confidence_level: confidence level for the interval (default 0.95)
        """
        
        # For profiling: use chi-square threshold for 1 DOF
        delta_logL = LikelihoodProfiler.get_delta_log_likelihood(
            confidence_level=confidence_level, dof=1)

        try:
            lower, upper = self.profile_brent(
                name=name,
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
            center = self.pt.get_accessor(name).get()
            std = lap.estim_variance_by_name(name).sqrt().item()
            z_score = stats.norm.ppf(0.5 + confidence_level / 2)
            delta = z_score * std
            return center - delta, center + delta

    def estimate_confint_all(self, names: List[str] = None, confidence_level=0.95) -> pd.DataFrame:
        """
        Estimate confidence intervals via profiling for multiple parameters.

        Returns a DataFrame with columns:
            name | center | lower | upper | width | used_fallback
        """
        if names is None:
            names = self.pt.get_indexed_names()

        records = []
        for name in names:
            print(f"\nProfiling: {name}")
            model_copy = copy.deepcopy(self.model)
            profiler = LikelihoodProfiler(model_copy)
            center = profiler.pt.get_accessor(name).get()
            try:
                ci_low, ci_high = profiler.estimate_confint(
                    name=name,
                    confidence_level=confidence_level
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

    def estimate_confint_all_parallel(self, names: List[str] = None, confidence_level=0.95) -> pd.DataFrame:
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
            names = self.pt.get_indexed_names()

        def profile_one(name: str) -> dict:
            model_copy = self._copy_model()
            profiler = LikelihoodProfiler(model_copy)
            center = profiler.pt.get_accessor(name).get()
            try:
                ci_low, ci_high = profiler.estimate_confint(
                    name=name,
                    confidence_level=confidence_level
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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
    
