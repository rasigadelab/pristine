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

import torch
import copy
import warnings
import math
import scipy.stats as stats
import pandas as pd
from typing import Tuple, List
from .hessian_tools import HessianTools
from .parameter_tools import ParameterTools, TensorAccessor
from .optimize import Optimizer
from scipy.optimize import brentq
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
from scipy.stats import norm

class LikelihoodProfiler:
    """
    Estimate parameter confidence intervals via likelihood profiling and Laplace approximation.

    This class supports two complementary strategies for quantifying parameter uncertainty
    in differentiable models:

    1. Likelihood profiling:
        Re-optimizes all parameters while perturbing one target parameter at a time.
        Confidence intervals are derived by identifying where the log-likelihood drops
        by a fixed threshold from its maximum. Profiling handles non-quadratic or
        asymmetric likelihood surfaces and provides robust intervals in poorly-identified
        models.

    2. Laplace approximation:
        Estimates confidence intervals using a local quadratic approximation to the
        log-likelihood. Inverts the (dense or implicit) Hessian to recover marginal
        variances. Faster but less accurate when the likelihood is not approximately Gaussian.

    The profiler also supports:
        - Parallelized interval estimation for large models
        - Fallback from profiling to Laplace when optimization fails
        - Eigenvalue-based curvature diagnostics to assess identifiability and sloppiness

    Parameters:
        model: A differentiable model instance with a `.loss()` method and parameter tensors.

    Attributes:
        pt: ParameterTools instance for the model
        lap: HessianTools instance for Laplace variance estimation
        range_factor: Span (in stddevs) used to initialize profiling brackets
        reoptimize: Whether to re-fit model when profiling each fixed parameter value
        max_expand: Max number of bracketing attempts for root-finding in profiling
        max_workers: Maximum number of threads for parallel profiling

    Methods:
        estimate_confint_scalar(name, confidence_level, method):
            Compute 1D confidence interval for a named parameter using either
            'profile' or 'laplace' method.

        estimate_confints(names, confidence_level, method):
            Compute intervals for a list of parameters (sequential version).

        estimate_confints_parallel(names, confidence_level, method):
            Same as above, but using parallel execution with threads.

        estimate_all_confints_laplace(confidence_level, dense, num_samples):
            Estimate Laplace-based intervals for all parameters.

        curvature_report(top_k):
            Print and return curvature diagnostics using Hessian eigenanalysis,
            including smallest/largest eigenvalue, flatness ratio, and
            most influential parameters.
    """
    def __init__(self, model):
        self.model = model
        self.pt = ParameterTools(model)
        self.lap = HessianTools(model)
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
    

    def profile_brent(self, name: str, loglik_drop: float = 1.92) -> Tuple[float, float]:
        """
        Estimate a confidence interval for a scalar parameter using likelihood profiling
        with Brent’s root-finding method.

        This method fixes the parameter identified by `name` at different values and reoptimizes
        all other free parameters. It identifies the points at which the negative log-likelihood
        increases by a threshold `loglik_drop` from its minimum. These threshold-crossing points define the
        lower and upper bounds of the confidence interval.

        Brent's algorithm is used to solve for the roots of the log-likelihood difference function
        on each side of the maximum-likelihood estimate. If no bracketing root is found initially,
        the method expands the search interval geometrically up to `max_expand` times.

        Args:
            name (str):
                The fully qualified parameter name, possibly with index (e.g., "clock.log_rate[0]").
            
            loglik_drop (float, default=1.92):
                The log-likelihood drop defining the confidence bound. Use
                $loglik_drop = 0.5 \cdot \chi^2_{1, 1 - \alpha}$ for a $(1-\alpha) \cdot 100\%$ confidence level.
                For example, 1.92 corresponds to a 95% interval when degrees of freedom = 1.

        Returns:
            Tuple[float, float]:
                The (lower_bound, upper_bound) of the profiled confidence interval.
                If a bound cannot be determined due to failed bracketing, ±∞ is returned.

        Raises:
            RuntimeError:
                If Brent’s method fails to converge within the expanded bracket range.

        Notes:
            - The method uses a Laplace approximation to initialize the search interval, based on
            the estimated standard deviation of the parameter.
            - The model must define `.loss()` and be differentiable with respect to its parameters.
            - Profiling is most reliable when `reoptimize = True` (default).
        """
        # Step 1: Get MLE point and stddev from Laplace
        lap = HessianTools(self.model)
        # center = self.accessor.get()
        center = self.pt.get_accessor(name).get()
        stddev = lap.estim_variance_by_name(name).sqrt().item()
        span = stddev * self.range_factor
        loss_mle = self.model.loss().item()

        def find_bound(direction: str) -> float:
            # Define objective: zero when loss crosses the profiling threshold
            sign = -1.0 if direction == "lower" else 1.0
            # modcopy = copy.deepcopy(self.model)
            
            def loss_diff(value: float) -> float:
                fixed = [(name, value)]
                modcopy = copy.deepcopy(self.model)          
                loss = LikelihoodProfiler.optimize_with_fixed_parameters(
                    modcopy,
                    fixed_params=fixed,
                    reoptimize=self.reoptimize,
                    optimizer_args={"initial_lr": 0.1, "max_iterations": 500}
                )
                return loss - loss_mle - loglik_drop            
            
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

    ############################################################################
    # SINGLE-PARAMETER CONFIDENCE BOUNDS
    ############################################################################

    def estimate_confint_scalar_profile(
        self,
        name: str,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute the confidence interval for a single parameter using likelihood profiling.

        Parameters:
            name: full parameter name (e.g., "clock.log_rate" or "treecal.node_dates[2]")
            confidence_level: confidence level (e.g., 0.95)

        Returns:
            (lower, upper) bounds
        """
        delta_logL = self.get_delta_log_likelihood(confidence_level=confidence_level)
        return self.profile_brent(name=name, loglik_drop=delta_logL)

    def estimate_confint_scalar_laplace(
        self,
        name: str,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Estimate confidence interval for a parameter using Laplace approximation
        with conjugate gradient-based inverse Hessian.

        Parameters:
            name: full parameter name (e.g., 'clock.log_rate')
            confidence_level: desired level (default: 0.95)

        Returns:
            (lower, upper) bounds as floats
        """
        self.lap.dense = False  # Enforce CG-based diagonal Hessian inversion
        center = self.pt.get_accessor(name).get()
        std = self.lap.estim_variance_by_name(name).sqrt().item()
        z_score = stats.norm.ppf(0.5 + confidence_level / 2)
        delta = z_score * std
        return center - delta, center + delta

    def estimate_confint_scalar(
        self,
        name: str,
        confidence_level: float = 0.95,
        method: str = "profile"
    ) -> Tuple[float, float]:
        """
        Wrapper to choose CI estimation method.

        method: "profile" or "laplace"
        """
        if method == "profile":
            return self.estimate_confint_scalar_profile(name, confidence_level)
        elif method == "laplace":
            return self.estimate_confint_scalar_laplace(
                name, confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")

    ############################################################################
    # MULTI-PARAMETER CONFIDENCE BOUNDS
    ############################################################################

    def estimate_confints(self, names: str | List[str] = None, confidence_level=0.95, method: str = "profile") -> pd.DataFrame:
        """
        Estimate confidence intervals via profiling for multiple parameters.

        Returns a DataFrame with columns:
            name | lower | center | upper | method
        """
        if names is None:
            names = self.pt.get_indexed_names()        
        elif isinstance(names, str):
            names: List[str] = [names]

        records = []
        for name in names:
            print(f"\nProfiling: {name}")
            model_copy = self._copy_model()
            profiler = LikelihoodProfiler(model_copy)
            center = profiler.pt.get_accessor(name).get()

            ci_low, ci_high = profiler.estimate_confint_scalar(
                name=name,
                confidence_level=confidence_level,
                method=method
            )

            records.append({
                "name": name,
                "lower": ci_low,
                "center": center,
                "upper": ci_high,
                "method": method
            })

        return pd.DataFrame(records)

    def estimate_confints_parallel(self, names: List[str] = None, confidence_level=0.95, method: str = "profile") -> pd.DataFrame:
        """
        Parallelized version of estimate_confint_all using ThreadPoolExecutor.

        Parameters:
            names (List[str]): List of parameter names (e.g. 'clock.log_rate[0]').
            confidence_level (float): Confidence level for intervals.
            range_factor (float): Range in stddev units for profiling.
            max_workers (int): Number of threads.

        Returns:
            pd.DataFrame with columns: name | lower | center | upper | method
        """
        if names is None:
            names = self.pt.get_indexed_names()        
        elif isinstance(names, str):
            names: List[str] = [names]

        def profile_one(name: str) -> dict:
            model_copy = self._copy_model()
            profiler = LikelihoodProfiler(model_copy)
            center = profiler.pt.get_accessor(name).get()

            ci_low, ci_high = profiler.estimate_confint_scalar(
                name=name,
                confidence_level=confidence_level,
                method=method
            )

            return {
                "name": name,
                "lower": ci_low,
                "center": center,
                "upper": ci_high,
                "method": method
            }

        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(profile_one, name): name for name in names}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Profiling"):
                records.append(future.result())

        df = pd.DataFrame(records)
        df['order'] = df['name'].map({name: i for i, name in enumerate(names)})
        return df.sort_values("order").drop(columns="order").reset_index(drop=True)

    def estimate_all_confints_laplace(self, confidence_level=0.95, dense: bool = True, num_samples = 20) -> pd.DataFrame:
        """
        Compute confidence intervals using the Laplace approximation
        via the full dense inverse Hessian (high accuracy, complexity O(n^3))
        or (for very large models) Hutchinson method
        
        Parameters:
            confidence_level: confidence level (default 0.95)
            dense:  if True (default), use full dense Hessian inverse (accurate, slower); 
                    if False, use Hutchinson estimator (faster, approximate)       
        Returns:
            pd.DataFrame with columns: name | lower | center | upper | laplace
        """
        self.lap.dense = dense  # Force dense Hessian inversion
        self.lap.hutchinson_num_samples = num_samples
        variances = self.lap.estim_all_variances_dict()

        z_score = norm.ppf(0.5 + confidence_level / 2)

        records = []
        for name, tensor in self.pt.named_params:
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
                    "method": "laplace"
                })

        return pd.DataFrame(records)
    
    ###############################################################################
    # CURVATURE ANALYSIS
    ###############################################################################
    """
    In optimization and statistical inference, curvature analysis helps us understand how
    sensitive the likelihood or loss function is to changes in the model parameters. This
    sensitivity is captured by the second derivative of the loss, called the Hessian matrix.

    The eigenvalues of the Hessian represent the curvature along specific directions in
    parameter space. A large eigenvalue means the loss increases rapidly in that direction,
    indicating a steep, well-constrained surface. A small eigenvalue means the loss changes
    very little, even when moving far along that direction. This reveals a flat, poorly
    identified, or redundant parameter combination.

    By computing the smallest and largest eigenvalues of the Hessian, we can assess the
    overall shape of the loss surface. The ratio of these two values is called the curvature
    ratio or condition number. If this ratio is close to one, the surface is round and
    well-behaved. If the ratio is very small—say, one millionth—it means there is a flat
    valley where many different parameter settings give nearly the same likelihood.

    Flat directions in the Hessian often signal non-identifiability. That is, the data do
    not contain enough information to estimate some parameter combinations independently.
    For example, if increasing one parameter has the same effect on the likelihood as
    decreasing another, they are not separately identifiable.

    To help interpret this, the curvature report prints both the smallest and largest
    eigenvalues, the flatness ratio, and a classification: well-conditioned, mildly sloppy,
    or severely non-identifiable. It also lists the most influential parameters in both the
    flattest and steepest directions. This tells you which parameters are uncertain or
    possibly redundant, and which ones are tightly constrained by the data.

    Curvature analysis is not just a numerical tool; it gives insight into the structure of
    your model. It reveals where the likelihood is informative and where it is indifferent.
    This can guide reparameterization, regularization, or data collection strategies to
    improve inference and interpretability.
    """    
    def curvature_report(self, top_k: int = 8) -> dict:
        """
        Print and return a full curvature diagnostic report:
        - Smallest and largest eigenvalues
        - Flatness ratio
        - Top contributing parameters for each direction

        Returns:
            dict with keys:
                'lambda_min', 'lambda_max', 'flatness_ratio',
                'interpretation', 'top_min', 'top_max'
        """
        λmin, vmin = self.lap.extremal_eigenpair("smallest")
        λmax, vmax = self.lap.extremal_eigenpair("largest")
        ratio = λmin / λmax if λmax != 0 else float("inf")

        if ratio > 1e-2:
            note = "Well-conditioned"
        elif ratio > 1e-6:
            note = "Mild sloppiness"
        else:
            note = "Severe non-identifiability"

        print("\n================ CURVATURE REPORT ================\n")
        print(f"  Smallest eigenvalue     : {λmin:.3e}")
        print(f"  Largest  eigenvalue     : {λmax:.3e}")
        print(f"  Flatness ratio (λ_min/λ_max): {ratio:.3e}")
        print(f"  Interpretation          : {note}")

        entries = list(self.pt.name_to_index_map.items())   
        top_k = min(top_k, len(entries))

        def top_components(direction: torch.Tensor, label: str) -> List[Tuple[str, float]]:
            flat = direction.detach().cpu()         
            index_to_name = {v: k for k, v in entries}
            top_entries = torch.topk(flat.abs(), top_k)

            result = []
            print(f"\n  >>> Top {top_k} parameters in {label} direction:")
            for idx, val in zip(top_entries.indices.tolist(), top_entries.values.tolist()):
                name = index_to_name.get(idx, f"<unknown-{idx}>")
                component = direction[idx].item()
                print(f"    {name:<30} {component:+8.4e}")
                result.append((name, component))
            return result

        top_min = top_components(vmin, "flattest (λ_min)")
        top_max = top_components(vmax, "steepest (λ_max)")

        print("\n==================================================\n")

        return {
            "lambda_min": λmin,
            "lambda_max": λmax,
            "flatness_ratio": ratio,
            "interpretation": note,
            "top_min": top_min,
            "top_max": top_max
        }