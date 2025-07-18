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
import matplotlib.pyplot as plt

def plot_compare(x_ref, x_estim, title = "Calibration plot"):
    """
    Display a square scatterplot of reference and estimated parameters.
    """
    

    plt.figure(figsize=(3, 3))
    plt.scatter(x_ref, x_estim, color='blue', alpha=0.7, label="Parameters")

    # Add a diagonal dashed line (y = x)
    min_val = min(min(x_ref), min(x_estim))
    max_val = max(max(x_ref), max(x_estim))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Diagonal", alpha = 0.2)

    # Set square axes
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.axis("equal")  # Ensure same scale for x and y

    # Labels and title
    plt.xlabel("Reference value")
    plt.ylabel("Estimated value")
    plt.title(title)
    plt.legend()
    plt.draw()

def plot_compare_error_bars(X: torch.Tensor, Y: torch.Tensor, L: torch.Tensor, U: torch.Tensor, 
                            title = ""):
    # Ensure all tensors are at least 1D
    X = X.view(-1)
    Y = Y.view(-1)
    L = L.view(-1)
    U = U.view(-1)

    # Compute asymmetric errors
    lower_errors = (Y - L).view(1, -1)
    upper_errors = (U - Y).view(1, -1)
    yerr = torch.cat([lower_errors, upper_errors], dim=0)  # shape (2, N)

    # Plot
    plt.figure(figsize=(3, 3))
    plt.errorbar(
        X.tolist(), Y.tolist(),
        yerr=yerr.tolist(),
        fmt='o', capsize=0,
        label='Estimate with CI',
        alpha=0.6, markersize=4
    )
    plt.xlabel("Reference")
    plt.ylabel("Estimate")
    plt.title(title)
    plt.grid(False)

    # Add a diagonal dashed line (y = x)
    min_val = min(X.min(), L.min()).item()
    max_val = max(X.max(), U.max()).item()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Diagonal", alpha=0.2)
    plt.legend()
    plt.draw()
