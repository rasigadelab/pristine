import torch
from typing import Any, Dict


class IOTools:
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
        Save model state (tensors + scalars) to a .pt file.
        """
        self.state = self.extract_state()
        torch.save(self.state, self.filename)


    def load(self) -> None:
        """
        Load model state from a .pt file into the provided object.
        """
        self.state = torch.load(self.filename, weights_only=True)
        self.load_state()