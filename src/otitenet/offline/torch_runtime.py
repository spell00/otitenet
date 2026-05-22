from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from otitenet.offline.deployment import OfflineDeployment


def _normalize_model_name(model_name: str) -> str:
    model_name = str(model_name or "resnet18")
    if model_name.endswith("_mobile"):
        model_name = model_name[: -len("_mobile")]
    return model_name


def _build_state_dict_model(model_name: str, n_cats: int, n_batches: int, n_subcenters: int):
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision.models import resnet18, resnet50

    class OfflineStateDictNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_name = model_name

            if model_name == "resnet18":
                base = resnet18(weights=None)
                self.model = nn.Sequential(*(list(base.children())[:-1]))
                feature_dim = 512
            elif model_name == "resnet18t":
                base = resnet18(weights=None)
                self.model = nn.Sequential(
                    *(list(base.children())[:-4]),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                feature_dim = 128
            elif model_name == "resnet50":
                base = resnet50(weights=None)
                self.model = nn.Sequential(*(list(base.children())[:-1]))
                feature_dim = 2048
            else:
                raise ValueError(
                    f"Offline state-dict deployment does not support model_name={model_name!r}. "
                    "Export the production model to ONNX for this architecture."
                )

            self.dann = nn.Linear(feature_dim, n_batches)
            self.linear = nn.Linear(feature_dim, n_cats)
            self.subcenters = nn.Parameter(torch.randn(n_cats, n_subcenters, feature_dim))

        def forward(self, inp):
            enc = self.model(inp)
            enc = enc.squeeze(-1).squeeze(-1)
            domout = self.dann(enc)
            return enc, F.log_softmax(domout, dim=1)

    return OfflineStateDictNet()


def load_state_dict_model(path: Path, deployment: OfflineDeployment, device: str = "cpu"):
    import torch

    state_dict = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state_dict).__name__}")

    labels = deployment.labels
    n_cats = len(labels)
    if n_cats == 0 and "linear.weight" in state_dict:
        n_cats = int(state_dict["linear.weight"].shape[0])

    n_batches = 1
    if "dann.weight" in state_dict:
        n_batches = int(state_dict["dann.weight"].shape[0])

    n_subcenters = 3
    if "subcenters" in state_dict:
        subcenters = state_dict["subcenters"]
        if getattr(subcenters, "ndim", 0) >= 2:
            n_subcenters = int(subcenters.shape[1])

    model = _build_state_dict_model(
        model_name=_normalize_model_name(deployment.model_name),
        n_cats=n_cats,
        n_batches=n_batches,
        n_subcenters=n_subcenters,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def torch_logits_from_output(output: Any, model: Any):
    import torch

    if isinstance(output, tuple):
        if output and torch.is_tensor(output[0]) and hasattr(model, "linear"):
            return model.linear(output[0])
        output = output[0]

    if isinstance(output, Sequence) and output and torch.is_tensor(output[0]):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(f"Model returned unsupported output type: {type(output).__name__}")

    return output
