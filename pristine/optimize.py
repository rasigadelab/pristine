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
The Optimizer class performs numerical optimization of a model's parameters  
by minimizing its loss function using the Adam algorithm. Conceptually, it  
automates gradient-based fitting of statistical models or probabilistic  
phylogenetic models, where each parameter has a well-defined influence on a  
likelihood or objective function.

This optimizer is designed with robustness in mind: if a step leads to a  
non-finite (NaN or inf) or worse loss, it backtracks by reducing the learning  
rate until the update becomes acceptable. This safeguards the fitting  
procedure from diverging due to unstable gradients, which are common in  
complex likelihood surfaces. Once stability is restored, the optimizer can  
slowly increase the learning rate again.

The optimization stops when either the loss stabilizes (i.e., changes less  
than a small threshold), the learning rate becomes too small, or a maximum  
number of iterations is reached. This makes it well-suited for models with  
sharp or noisy likelihood landscapes, like those encountered in phylogenetic  
inference.
"""

import sys;import os;sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.optim as optim
from typing import List, Any
from .parameter_tools import ParameterTools

class Optimizer:
    """
    Configuration parameters for Adam optimization with backtracking.

    Attributes:
        initial_lr (float): The initial learning rate for Adam.
        backtrack_factor (float): The factor by which to reduce the learning rate when backtracking.
        lr_accel_decel_factor (float): The factor by which the learning rate re-accelerates after backtracking
        min_lr (float): The minimum allowable learning rate; if reached, optimization stops.
        max_iterations (int): The maximum number of iterations to perform.
        print_interval (int): Print progress every `print_interval` iterations.
        convergence_threshold (float): If the absolute difference between new loss and old loss
        is below this threshold, the optimization is considered converged.
    """
    def __init__(
        self,
        model: Any,
        initial_lr: float = 0.1,
        backtrack_enable: bool = True,
        backtrack_factor: float = 0.9,
        lr_accel_decel_factor: float = 0.5,
        min_lr: float = 1e-6,
        max_iterations: int = 1000,
        print_interval: int = 100,
        convergence_threshold: float = 0.001
    ):
        self.model = model
        self.parameters = ParameterTools(model).params
        self.initial_lr = initial_lr
        self.backtrack_enable = backtrack_enable
        self.backtrack_factor = backtrack_factor
        self.lr_accel_decel_factor = lr_accel_decel_factor
        self.min_lr = min_lr
        self.max_iterations = max_iterations
        self.print_interval = print_interval
        self.convergence_threshold = convergence_threshold
        self.num_iter = 0 # Iteration counter

    def optimize(self) -> List[torch.Tensor]:
        """
        Robust adaptive moment (ADAM) optimization with backtracking to handle NaN or infinite losses.
        If a problematic update is encountered, the learning rate is reduced by a specified
        factor (`backtrack_factor`) until the new loss is finite and not worse than the old loss.
        If the learning rate falls below `min_lr`, the optimization is stopped.
        """
        # Define optimizer
        # self.parameters = self.parameters()        
        optimizer = optim.Adam(self.parameters, lr=self.initial_lr)

        for self.num_iter in range(self.max_iterations):
            # Zero-out existing gradients
            optimizer.zero_grad()

            # Compute loss
            loss = self.model.loss()

            # Backpropagate
            loss.backward(retain_graph=True) 

            # Check variable and grad finiteness
            for param in self.parameters:
                if not torch.isfinite(param).all():
                    raise RuntimeError("Non-finite free parameter, exiting. Consider investigating \
                        using torch.autograd.set_detect_anomaly(True)")   
                if not torch.isfinite(param.grad).all():
                    raise RuntimeError("Non-finite gradient, exiting. Consider investigating \
                        using torch.autograd.set_detect_anomaly(True)")                

            # Print progress at given intervals
            if self.num_iter % self.print_interval == 0:
                print(".", end="", flush=True)

            # Store old values of parameters
            old_parameters = [param.clone().detach() for param in self.parameters]

            # Optimizer step (initial attempt)
            optimizer.step()

            # Compute new loss after the step
            new_loss = self.model.loss()

            # Backtracking if new loss is NaN/inf or worse than old loss
            while (not torch.isfinite(new_loss)) or (new_loss > loss + self.convergence_threshold):
                if self.backtrack_enable is not True:
                    raise RuntimeError("Non-finite objective or bad search direction. Consoder enabling backtracking.")

                # Reduce the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.backtrack_factor

                    # If learning rate is below minimum, stop
                    if param_group['lr'] < self.min_lr:
                        # print(f"\nStopping early at iteration {self.num_iter}: learning rate ({param_group['lr']:.2e}) too small.")
                        print("!", end="", flush=True)
                        return self.parameters

                # Revert parameters to old values
                with torch.no_grad():
                    for old_param, param in zip(old_parameters, self.parameters):
                        param.copy_(old_param)

                # Retry the step with reduced learning rate
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                new_loss = self.model.loss()

            # Check for convergence
            if self.num_iter > 10 and abs(new_loss - loss) < self.convergence_threshold:
                # print(f"\nConverged at iteration {self.num_iter}, Loss: {loss.item():.3e}")
                break

            # Restore the learning rate partially (per original code: dividing by backtrack_factor**0.5)
            for param_group in optimizer.param_groups:
                if param_group['lr'] < self.initial_lr:
                    param_group['lr'] /= (self.backtrack_factor ** self.lr_accel_decel_factor)

        return self.parameters

