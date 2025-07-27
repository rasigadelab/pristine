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
import torch.optim
from typing import List, Any
import matplotlib.pyplot as plt
from .parameter_tools import ParameterTools

class Optimizer:
    """
    Adaptive gradient-based optimizer using Adam with loss-aware backtracking.

    This class performs robust optimization of a differentiable model's parameters by
    minimizing its `.loss()` method using the Adam algorithm. It is specifically designed
    for statistical models with sharp or non-convex likelihood surfaces, such as those
    encountered in phylogenetics or probabilistic inference.

    The optimizer incorporates backtracking logic: if a step leads to a non-finite or
    worse loss, it reverts the update and reduces the learning rate until the step becomes
    acceptable. This prevents divergence due to unstable gradients and improves convergence
    in irregular objective landscapes.

    Optimization stops when:
        - The loss stabilizes (i.e., the change is below `convergence_threshold`),
        - The learning rate falls below `min_lr`,
        - Or the maximum number of iterations (`max_iterations`) is reached.

    Parameters:
        model (Any):
            The model instance to optimize. Must define a `.loss()` method and contain
            differentiable leaf parameters (typically via `requires_grad=True`).

        initial_lr (float, default=0.1):
            Initial learning rate for the Adam optimizer.

        backtrack_enable (bool, default=True):
            Whether to enable backtracking if the loss increases or becomes non-finite.

        backtrack_factor (float, default=0.9):
            Multiplicative factor by which to reduce learning rate during backtracking.

        lr_accel_decel_factor (float, default=0.5):
            Factor governing how quickly the learning rate recovers after backtracking.

        min_lr (float, default=1e-6):
            Minimum learning rate. Optimization halts if learning rate drops below this.

        max_iterations (int, default=1000):
            Maximum number of optimization iterations.

        print_interval (int, default=100):
            Interval at which to print progress (prints a dot).

        convergence_threshold (float, default=0.001):
            Threshold for loss change to consider convergence.

    Attributes:
        parameters (List[torch.Tensor]):
            List of model parameters to optimize (discovered via `ParameterTools`).
        num_iter (int):
            Counter tracking the number of completed iterations.

    Returns:
        optimize() -> List[torch.Tensor]:
            The list of optimized parameters (in-place modification of the model).
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
        self.pt = ParameterTools(model)

        # Parameters
        self.initial_lr = initial_lr
        self.backtrack_enable = backtrack_enable
        self.backtrack_factor = backtrack_factor
        self.lr_accel_decel_factor = lr_accel_decel_factor
        self.min_lr = min_lr
        self.max_iterations = max_iterations
        self.print_interval = print_interval
        self.convergence_threshold = convergence_threshold

        # Optimizer reset/restart control
        self.reset_interval = 500
        self.patience = 50
        self.patience_ratio = 10
        self.patience_loglik_threshold = convergence_threshold * self.patience_ratio
        self.num_post_convergence_restarts = 5
        self.max_lr_restarts = 1  # configurable
        self.lr_restart_count = 0

        # Run diagnostics
        self.num_iter = 0 # Iteration counter
        self.backtrack_counter = 0
        self.loss_history = []       # Store loss
        self.lr_history = []         # Store learning rate per iteration
        self.restart_count = -1
        self.restart_iters = []      # Iterations at which restarts occurred
        self.backtrack_iters = []    # Iterations where any backtracking was triggered
        self.convergence_restart_count = 0

        self.optimizer = torch.optim.Adam(self.pt.params, lr=self.initial_lr)
        # self.reset_optimizer()

    def reset_optimizer(self) -> None:
        """
        Reset the internal Adam optimizer state and learning rate.

        This clears all moment estimates (m, v) and resets the learning rate
        to `self.initial_lr`. Use during learning rate scheduling restarts.
        """
        self.optimizer = torch.optim.Adam(self.pt.params, lr=self.initial_lr)
        self.backtrack_counter = 0
        self.restart_count += 1
        self.restart_iters.append(self.num_iter)
        print("_", end="", flush=True)

    def optimize(self) -> List[torch.Tensor]:
        """
        Perform gradient-based optimization of the model's loss using Adam,
        with optional backtracking for robustness.

        This method iteratively updates model parameters by minimizing the
        `.loss()` function, applying adaptive learning rate reduction if
        the loss increases or becomes non-finite. It automatically reverts
        failed steps and retries with smaller learning rates until stability
        is restored or the minimum threshold is reached.

        Returns:
            List[torch.Tensor]:
                A list of optimized parameter tensors (in-place in the model).

        Raises:
            RuntimeError:
                If parameter values or gradients become non-finite, and
                backtracking is disabled.
        """
        for self.num_iter in range(self.max_iterations):
            # Zero-out existing gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss = self.model.loss()

            # Backpropagate
            loss.backward(retain_graph=True) 

            # Check variable and grad finiteness
            for param in self.pt.params:
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
            old_parameters = [param.clone().detach() for param in self.pt.params]

            # Optimizer step (initial attempt)
            self.optimizer.step()

            # Compute new loss after the step
            new_loss = self.model.loss()

            #############################################################################
            # BACKTRACKING LOGIC
            #############################################################################

            # Backtracking if new loss is NaN/inf or worse than old loss
            while (not torch.isfinite(new_loss)) or (new_loss > loss + self.convergence_threshold):
                if self.backtrack_enable is not True:
                    raise RuntimeError("Non-finite objective or bad search direction. Consoder enabling backtracking.")

                self.backtrack_counter += 1

                # Reduce the learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.backtrack_factor

                    # If learning rate is below minimum, restart
                    if param_group['lr'] < self.min_lr:
                        if self.lr_restart_count < self.max_lr_restarts:
                            self.lr_restart_count += 1
                            print("!", end="", flush=True)
                            # print(f"\n[Restart] LR fell below min_lr at iter {self.num_iter}, restarting...")
                            self.reset_optimizer()
                            continue  # restart optimization from scratch
                        else:
                            # We're really stuck...
                            print("/!\\", end="", flush=True)
                            return self.pt.params

                # Revert parameters to old values
                with torch.no_grad():
                    for old_param, param in zip(old_parameters, self.pt.params):
                        param.copy_(old_param)

                # Retry the step with reduced learning rate
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                new_loss = self.model.loss()

            #############################################################################
            # LOG DIAGNOSTICS
            #############################################################################

            # Log that this iteration backtracked at least once
            if self.backtrack_counter > 0:
                self.backtrack_iters.append(self.num_iter)
                self.backtrack_counter = 0

            # Track loss
            self.loss_history.append(new_loss.item())

            # Store base learning rate (same for all params in this setup)
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])

            #############################################################################
            # OPTIMIZER RESET LOGIC
            #############################################################################

            # 1. Stagnation-based restart
            if len(self.loss_history) > self.patience:
                window = self.loss_history[-self.patience:]
                if max(window) - min(window) < self.patience_loglik_threshold:
                    self.reset_optimizer()
                    continue

            # 2. Fixed interval restart            
            if self.num_iter > 0 and self.num_iter % self.reset_interval == 0:
                self.reset_optimizer()
                continue

            #############################################################################
            # CONVERGENCE CHECK AND REACCELERATION
            #############################################################################
            # Check for convergence
            if self.num_iter > 10 and abs(new_loss - loss) < self.convergence_threshold:
                if self.convergence_restart_count < self.num_post_convergence_restarts:
                    self.convergence_restart_count += 1
                    self.reset_optimizer()
                    # print(">", end="", flush=True)
                    continue  # Resume optimization from restart
                else:
                    print(">", end="", flush=True)
                    # print(f"\n[Info] Converged after {self.num_iter + 1} iterations "
                        # f"with {self.convergence_restart_count} convergence restarts.")
                    break


            # Restore the learning rate partially (per original code: dividing by backtrack_factor**0.5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= (self.backtrack_factor ** self.lr_accel_decel_factor)

                # if param_group['lr'] < self.initial_lr:
                #     param_group['lr'] /= (self.backtrack_factor ** self.lr_accel_decel_factor)

        return self.pt.params

    def plot_diagnostics(self):
        """
        Plot loss and learning rate trajectories over iterations,
        marking restarts (dashed lines) and backtracks (triangles).
        """
        if not self.loss_history or not self.lr_history:
            print("No optimization history available to plot.")
            return

        iterations = list(range(len(self.loss_history)))

        fig, ax1 = plt.subplots(figsize=(8, 4))

        # --- Loss curve (left y-axis)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(iterations, self.loss_history, color="tab:blue", label="Loss")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Mark backtracking steps
        if hasattr(self, "backtrack_iters"):
            bt_valid = [(i, self.loss_history[i]) for i in self.backtrack_iters if i < len(self.loss_history)]
            if bt_valid:
                x_bt, y_bt = zip(*bt_valid)
                ax1.plot(x_bt, y_bt, "v", color="red", label="Backtrack", markersize=5)

        # Mark restarts
        if hasattr(self, "restart_iters"):
            for i in self.restart_iters:
                ax1.axvline(x=i, color="gray", linestyle="--", alpha=0.5)
            if self.restart_iters:
                ax1.text(self.restart_iters[-1], max(self.loss_history), "Restarts", color="gray", ha="left", fontsize=8)

        # --- Learning rate curve (right y-axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Learning Rate", color="tab:red")
        ax2.plot(iterations, self.lr_history, color="tab:red", linestyle="--", label="Learning Rate")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # --- Final layout
        fig.tight_layout()
        plt.title("Loss, Learning Rate, and Restart/Backtrack Diagnostics")
        plt.show()
