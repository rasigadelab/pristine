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
from typing import Any, Dict


class IOTools:
    """
    Utility for serializing and restoring the internal state of an object to/from disk.

    This tool recursively traverses all attributes of a model object (including nested classes,
    dictionaries, lists, and tensors), extracts scalar values and tensors into a flat dictionary,
    and enables saving/loading of the state using PyTorch's `.pt` format.

    Use cases:
        - Checkpointing model state across training runs
        - Serializing fitted parameter values
        - Exporting models for reproducible inference or evaluation

    Attributes:
        obj (Any): The target object whose state is being saved or restored.
        filename (str): Path to the file used for saving/loading state.
        state (Dict[str, Any]): Flattened dictionary representing the object's state.

    Notes:
        - Tensor values are cloned and detached when saved.
        - Scalars (int, float, bool, str) and scalar tensors are stored as native Python types.
        - During loading, tensor shapes are preserved, and scalar tensors are restored in-place
          when possible.
        - Key format is dot-qualified for attributes and indexed for sequences (e.g., 'foo.bar[0].weight').
    """
    def __init__(self, obj: Any, filename: str):
        self.obj: Any = obj
        self.filename: str = filename
        self.state: Dict[str, Any] = {}

    def extract_state(self) -> Dict[str, Any]:
        """
        Traverse an object and collect all scalar values and tensors into a flat dictionary.
        The keys are dot-qualified names.
        """
        self.state = {}

        def _traverse(prefix, item):
            if isinstance(item, (int, float, bool, str)) or (isinstance(item, torch.Tensor) and item.numel() == 1):
                self.state[prefix] = item.item() if isinstance(item, torch.Tensor) else item
            elif isinstance(item, torch.Tensor):
                self.state[prefix] = item.detach().clone()
            elif isinstance(item, dict):
                for k, v in item.items():
                    _traverse(f"{prefix}.{k}" if prefix else k, v)
            elif isinstance(item, (list, tuple)):
                for i, v in enumerate(item):
                    _traverse(f"{prefix}[{i}]", v)
            elif hasattr(item, '__dict__'):
                for k, v in vars(item).items():
                    _traverse(f"{prefix}.{k}" if prefix else k, v)

        _traverse("", self.obj)
        return self.state


    def load_state(self) -> None:
        """
        Restore values into an object from a flat dictionary of dot-qualified names.
        Supports both tensors and Python scalar types.
        """
        def _assign(target, key_path, value):
            if not key_path:
                return
            key = key_path[0]
            rest = key_path[1:]
            if isinstance(target, dict):
                if rest:
                    _assign(target[key], rest, value)
                else:
                    target[key] = value
            elif isinstance(target, (list, tuple)):
                idx = int(key.strip("[]"))
                if rest:
                    _assign(target[idx], rest, value)
                else:
                    target[idx] = value
            elif hasattr(target, '__dict__'):
                if rest:
                    _assign(getattr(target, key), rest, value)
                else:
                    attr = getattr(target, key, None)
                    if isinstance(attr, torch.Tensor):
                        with torch.no_grad():
                            if isinstance(value, torch.Tensor):
                                attr.copy_(value)
                            elif attr.numel() == 1:
                                attr.fill_(value)
                            else:
                                attr.copy_(torch.tensor(value, dtype=attr.dtype))
                    else:
                        setattr(target, key, value)
            else:
                raise ValueError(f"Unsupported assignment path: {'.'.join(key_path)}")

        for full_key, value in self.state.items():
            parts = []
            for part in full_key.replace("]", "").split("."):
                if "[" in part:
                    prefix, idx = part.split("[")
                    parts.append(prefix)
                    parts.append(f"[{idx}]")
                else:
                    parts.append(part)
            _assign(self.obj, parts, value)


    def save(self) -> None:
        """
        Save the object's current state to disk.

        This method traverses all nested components of the target object (`self.obj`), 
        extracts scalar and tensor values into a flat dictionary, and saves the result 
        as a `.pt` file at `self.filename`.

        File format:
            The saved file is a PyTorch checkpoint containing all model parameters
            and scalars, suitable for later reloading using the `load()` method.

        Raises:
            IOError: If saving to the specified path fails.
        """
        self.state = self.extract_state()
        torch.save(self.state, self.filename)


    def load(self) -> None:
        """
        Load the object's state from a `.pt` file.

        This method reads the flattened dictionary stored at `self.filename`,
        and reassigns the values into the corresponding fields of `self.obj`.

        Behavior:
            - Tensors are restored using in-place `.copy_()` when possible.
            - Scalar tensors with one element are filled using `.fill_()`.
            - Non-tensor fields (ints, floats, strings, etc.) are directly reassigned.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the target object does not support a corresponding attribute path.
        """
        self.state = torch.load(self.filename, weights_only=True)
        self.load_state()