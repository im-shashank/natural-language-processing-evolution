import os
import torch
import torch.nn as nn

# Calculate base_dir to point to the GPT-2/main folder
# __file__ is in GPT-2/main/model/weight_manager.py
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(base_dir, "weights")

class WeightManager:
    def __init__(self, weights_dir=WEIGHTS_DIR):
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.weights_dir, filename)

    def save(self, model: nn.Module, filename: str = "gpt2_model.pt",
             optimizer=None, step: int | None = None,
             extra: dict | None = None):
        checkpoint = {"model_state_dict": model.state_dict()}
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if step is not None:
            checkpoint["step"] = step
        if extra:
            checkpoint.update(extra)

        path = self._path(filename)
        torch.save(checkpoint, path)
        print(f"weights saved → {path}")

    def load(self, model: nn.Module, filename: str = "gpt2_model.pt",
             optimizer=None, device=None):
        path = self._path(filename)
        if not os.path.isfile(path):
            print(f"no weights found at {path}, starting from scratch")
            return None

        # Added weights_only=True for safety as per modern PyTorch conventions
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"weights loaded ← {path}")
        return checkpoint

    def exists(self, filename: str = "gpt2_model.pt") -> bool:
        return os.path.isfile(self._path(filename))
