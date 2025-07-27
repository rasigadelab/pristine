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

from typing import List, Tuple, Any, Dict
import torch
import re

class TensorAccessor:
    """
    Utility to safely access and modify scalar or indexed tensor values,
    including handling gradients. Supports in-place modification and gradient
    zeroing without breaking autograd.
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
    Introspects a model to extract, flatten, and manipulate its differentiable parameters.

    - Deduplicates parameters based on tensor identity (avoids optimizer duplication).
    - Maintains canonical and alias names for referencing parameters.
    - Provides flat indexing, name normalization, and alias-aware lookups.
    """

    def __init__(self, model: Any):
        """
        Recursively collects all differentiable leaf tensors in the model.
        Deduplicates based on identity and builds alias-to-canonical maps.
        """
        self.model = model

        all_named_tensors = self._collect_all_named_tensors(model)

        id_to_names: Dict[int, List[str]] = {}
        id_to_tensor: Dict[int, torch.Tensor] = {}

        for name, tensor in all_named_tensors:
            tid = id(tensor)
            if tid not in id_to_names:
                id_to_names[tid] = [name]
                id_to_tensor[tid] = tensor
            else:
                id_to_names[tid].append(name)

        self.named_params: List[Tuple[str, torch.Tensor]] = [
            (id_to_names[tid][0], tensor) for tid, tensor in id_to_tensor.items()
        ]
        self.params: List[torch.Tensor] = [tensor for _, tensor in self.named_params]
        self.shapes: List[torch.Size] = [p.shape for p in self.params]

        self.alias_map: Dict[str, str] = {
            alias: names[0]
            for names in id_to_names.values()
            for alias in names
        }

        self.name_to_index_map: Dict[str, int] = self._build_flat_index_map()

    @staticmethod
    def flatten_tensor_list(tensors: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Flattens a tuple or list of tensors into a single 1D vector.
        """
        return torch.cat([t.view(-1) for t in tensors])

    def _collect_all_named_tensors(self, obj: Any, prefix="") -> List[Tuple[str, torch.Tensor]]:
        """
        Recursively traverses the object and returns a list of (name, tensor) pairs
        for all leaf tensors that require gradients. May contain duplicates.
        """
        named_params = []

        if isinstance(obj, torch.Tensor):
            if obj.requires_grad and obj.is_leaf:
                named_params.append((prefix.rstrip("."), obj))

        elif isinstance(obj, dict):
            for k, v in obj.items():
                named_params.extend(self._collect_all_named_tensors(v, prefix + f"{k}."))

        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                named_params.extend(self._collect_all_named_tensors(item, prefix + f"{i}."))

        elif hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                named_params.extend(self._collect_all_named_tensors(v, prefix + f"{k}."))

        return named_params

    def _build_flat_index_map(self) -> Dict[str, int]:
        """
        Maps fully qualified parameter names with index notation to flat indices.
        E.g., "treecal.node_dates[3]" → global offset in the flat parameter vector.
        """
        name_map = {}
        offset = 0
        for name, param in self.named_params:
            for i in range(param.numel()):
                name_map[f"{name}[{i}]"] = offset
                offset += 1
        return name_map

    def flatten_parameters(self) -> torch.Tensor:
        """
        Returns a flat 1D tensor of all unique parameters in order.
        """
        return torch.cat([p.view(-1) for p in self.params])

    def unflatten_vector(self, vec: torch.Tensor) -> List[torch.Tensor]:
        """
        Given a flat vector, reshapes it back into the original parameter tensor shapes.
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
        Alias for unflatten_vector for naming compatibility.
        """
        return self.unflatten_vector(vec)

    def get_named_index(self, name_with_index: str) -> int:
        """
        Return flat vector index of a given parameter element.
        Automatically resolves base name aliases.
        """
        base, idx = self.parse_name(name_with_index)
        canonical_base = self.resolve_alias(base)
        if idx is None:
            return self.name_to_index_map[canonical_base]
        indexed_name = f"{canonical_base}[{idx}]"
        return self.name_to_index_map[indexed_name]

    def get_named_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Returns a list of unique (canonical_name, tensor) tuples.
        """
        return self.named_params

    def get_indexed_names(self) -> List[str]:
        """
        Returns a list of fully qualified parameter names with index notation.
        """
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
        Ensures consistent naming by resolving base aliases and adding [0] index if missing.
        Useful for scalar tensors.
        """
        base, idx = self.parse_name(name)
        canonical_base = self.resolve_alias(base)
        if idx is None:
            return canonical_base
        indexed_name = f"{canonical_base}[{idx}]"
        if indexed_name in self.name_to_index_map:
            return indexed_name
        raise KeyError(f"Name '{name}' could not be resolved to a parameter index.")

    def parse_name(self, name: str) -> Tuple[str, int | None]:
        """
        Splits a name like 'foo.bar[3]' into ('foo.bar', 3).
        Returns index=None for scalars.
        """
        match = re.match(r"^(.*)\[(\d+)\]$", name)
        if match:
            return match.group(1), int(match.group(2))
        return name, None

    def get_accessor(self, name: str) -> TensorAccessor:
        """
        Returns a TensorAccessor for a given parameter name (with or without alias).
        """
        base, idx = self.parse_name(name)
        base = self.resolve_alias(base)
        param = dict(self.named_params)[base]
        return TensorAccessor(param, idx)

    def resolve_alias(self, name: str) -> str:
        """
        Maps an alias name to its canonical name.
        Works for base names and indexed names like 'foo.bar[2]'.
        """
        if name in self.alias_map:
            return self.alias_map[name]

        match = re.match(r"^(.*)\[(\d+)\]$", name)
        if match:
            base, idx = match.groups()
            canonical_base = self.alias_map.get(base, base)
            return f"{canonical_base}[{idx}]"

        return name
