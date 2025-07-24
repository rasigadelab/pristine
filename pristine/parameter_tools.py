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
TensorAccessor is a lightweight utility for safely accessing and modifying
individual values inside PyTorch tensors, including scalar and indexed entries.
It ensures gradient integrity by modifying values in-place without interfering
with autograd. It can also selectively zero out gradients at specific positions,
which is useful during likelihood profiling or parameter updates in manual
optimization workflows.

ParameterTools is a model introspection and manipulation utility designed to
manage structured parameters in custom PyTorch models. It recursively traverses
the model's attributes and substructures to identify all trainable leaf tensors,
records their names and shapes, and builds a flat index map for efficient access.
It enables easy flattening and unflattening of parameters, translating between
flat vectors (useful for optimizers, profilers, and Laplace estimators) and the
original model structure. Additionally, it supports name normalization,
name-to-index mapping, and parsing of parameter element references like
"clock.log_rate[2]", making it ideal for diagnostic and statistical tasks such
as confidence interval estimation and profiling.
"""

from typing import List, Tuple, Any, Dict
import torch
import re

class TensorAccessor:
    """
    Utility to safely access and modify scalar or indexed tensor values,
    including handling gradients.
    """
    def __init__(self, tensor: torch.Tensor, index: int | tuple | None = None):
        self.tensor = tensor
        self.index = index if tensor.ndim > 0 else None

    def get(self) -> float:
        if self.index is None:
            return self.tensor.item()
        return self.tensor[self.index].item()

    def set(self, value: float):
        with torch.no_grad():
            if self.index is None:
                if isinstance(value, torch.Tensor):
                    self.tensor.copy_(value.clone().detach())
                else:
                    self.tensor.copy_(torch.tensor(value, dtype=self.tensor.dtype, device=self.tensor.device))
            else:
                self.tensor[self.index] = value

    def zero_grad(self):
        if self.tensor.grad is not None:
            with torch.no_grad():
                if self.index is None:
                    self.tensor.grad.zero_()
                else:
                    self.tensor.grad[self.index] = 0.0


class ParameterTools:
    """
    A utility class for managing and manipulating learnable leaf parameters in a structured model.
    This instance-based version holds model state, flat parameter index maps, shapes, and names.
    """

    def __init__(self, model: Any):
        """
        Initializes the parameter tools for a given model.
        Automatically collects all trainable leaf parameters and their names.
        """
        self.model = model
        self.named_params: List[Tuple[str, torch.Tensor]] = self._collect_named_leaf_parameters(model)
        self.params: List[torch.Tensor] = [p for _, p in self.named_params]
        self.shapes: List[torch.Size] = [p.shape for p in self.params]
        self.name_to_index_map: Dict[str, int] = self._build_flat_index_map()

    @staticmethod
    def flatten_tensor_list(tensors: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Flattens any list of tensors into a 1D vector.
        Useful for non-param objects like gradient lists.
        """
        return torch.cat([t.view(-1) for t in tensors])

    def _collect_named_leaf_parameters(self, obj: Any, prefix="") -> List[Tuple[str, torch.Tensor]]:
        named_params = []

        if isinstance(obj, torch.Tensor):
            if obj.requires_grad and obj.is_leaf:
                named_params.append((prefix.rstrip("."), obj))

        elif isinstance(obj, dict):
            for k, v in obj.items():
                named_params.extend(self._collect_named_leaf_parameters(v, prefix + f"{k}."))

        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                named_params.extend(self._collect_named_leaf_parameters(item, prefix + f"{i}."))

        elif hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                named_params.extend(self._collect_named_leaf_parameters(v, prefix + f"{k}."))

        return named_params

    def _build_flat_index_map(self) -> Dict[str, int]:
        """
        Maps fully-qualified parameter element names to flat indices in the global flattened vector.
        E.g., "clock.weight[3]" → 3
        """
        name_map = {}
        offset = 0
        for name, param in self.named_params:
            for i in range(param.numel()):
                name_map[f"{name}[{i}]"] = offset # + i
                offset += 1
        return name_map

    def flatten_parameters(self) -> torch.Tensor:
        """
        Returns a single 1D tensor by flattening all collected parameters.
        """
        return torch.cat([p.view(-1) for p in self.params])

    def unflatten_vector(self, vec: torch.Tensor) -> List[torch.Tensor]:
        """
        Given a flat vector, reshapes and returns it into the original parameter structure.
        """
        outputs = []
        offset = 0
        for shape in self.shapes:
            numel = int(torch.prod(torch.tensor(shape)).item())
            outputs.append(vec[offset:offset + numel].view(shape))
            offset += numel
        return outputs

    def vector_to_parameter_list(self, vec: torch.Tensor) -> List[torch.Tensor]:
        """
        Alias for unflatten_vector for interface compatibility.
        """
        return self.unflatten_vector(vec)

    def get_named_index(self, name_with_index: str) -> int:
        """
        Returns the global flat index corresponding to a named parameter element (e.g., "clock.weight[2]").
        """
        return self.name_to_index_map[name_with_index]

    def get_named_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Returns a list of (name, parameter) tuples.
        """
        return self.named_params
 
    def get_indexed_names(self)->List[str]:
        names = []
        for pname, tensor in self.named_params:
            if tensor.ndim == 0:
                names.append(f"{pname}")
            else:
                for i in range(tensor.numel()):
                    names.append(f"{pname}[{i}]")
        return names

    def normalize_name(self, name: str) -> str:
        """
        Ensures scalar names like 'clock.log_rate' are mapped to 'clock.log_rate[0]' if needed.
        """
        if name in self.name_to_index_map:
            return name
        base = name.split("[")[0]
        candidate = f"{base}[0]"
        if candidate in self.name_to_index_map:
            return candidate
        raise KeyError(f"Name '{name}' could not be resolved to a parameter index.")

    def parse_name(self, name: str) -> Tuple[str, int | None]:
        """
        Parse a parameter name or indexed name.
        Returns (base_name, optional index) where:
        - "clock.weight[3]" → ("clock.weight", 3)
        - "treecal.lengths" → ("treecal.lengths", None)
        """
        match = re.match(r"^(.*)\[(\d+)\]$", name)
        if match:
            return match.group(1), int(match.group(2))
        return name, None
    
    def get_accessor(self, name: str)->TensorAccessor:
        base, idx = self.parse_name(name)
        param = dict(self.named_params)[base]
        return TensorAccessor(param, idx)