from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from otitenet.api.mobile_deployment import sha256_file
from otitenet.offline.deployment import DEFAULT_DEPLOYMENT_DIR, OfflineDeployment
from otitenet.offline.torch_runtime import load_state_dict_model, torch_logits_from_output


class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch_logits_from_output(self.model(image), self.model)


class EmbeddingOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim > 2:
            output = output.reshape(output.shape[0], -1)
        return output


def load_manifest(deployment_dir: Path) -> dict:
    manifest_path = deployment_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing deployment manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_uses_web_head(manifest: dict) -> bool:
    params = manifest.get("production_params", {}) or {}
    prototypes = str(params.get("prototypes_to_use") or params.get("prototypes") or "").lower()
    head = str(params.get("head") or manifest.get("head") or manifest.get("head_type") or "").lower()
    return prototypes not in {"", "none", "no", "nan"} or "prototype" in head or "knn" in head


def _check_onnx_export(wrapper: torch.nn.Module, onnx_path: Path, dummy: torch.Tensor) -> float:
    import onnxruntime as ort

    with torch.no_grad():
        expected = wrapper(dummy).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    actual = session.run(None, {input_name: dummy.detach().cpu().numpy()})[0]
    return float(np.max(np.abs(expected - actual)))


def _quantize_dynamic_uint8(source_path: Path, output_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source_path), str(output_path), weight_type=QuantType.QUInt8)


def export_onnx(
    deployment_dir: Path,
    output_name: str,
    opset: int,
    check: bool,
    quantize: bool,
    keep_float: bool,
    allow_head_mismatch: bool,
    embedding_output: bool,
) -> tuple[Path, float | None, str]:
    manifest = load_manifest(deployment_dir)
    uses_web_head = _manifest_uses_web_head(manifest)
    if uses_web_head and not (allow_head_mismatch or embedding_output):
        raise RuntimeError(
            "The deployment manifest records a prototype/KNN-style production head, "
            "but this exporter writes logits from the model's linear classifier. "
            "That will not match localhost scores. Export a matching embedding/prototype "
            "deployment, or pass --allow-head-mismatch only if different scoring is intentional."
        )
    deployment = OfflineDeployment(root=deployment_dir, manifest=manifest)

    source_model = deployment.model_file
    if source_model.suffix.lower() == ".onnx" and source_model.name == output_name:
        return source_model, None, str(manifest.get("quantization", "none"))

    model = load_state_dict_model(source_model, deployment, device="cpu")
    wrapper = EmbeddingOnlyWrapper(model).eval() if embedding_output else LogitsOnlyWrapper(model).eval()

    height, width = deployment.input_size
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)
    output_name_for_graph = "embedding" if embedding_output else "logits"

    output_path = deployment_dir / output_name
    float_path = output_path.with_name(f"{output_path.stem}.float{output_path.suffix}")
    export_path = float_path if quantize else output_path
    torch.onnx.export(
        wrapper,
        dummy,
        str(export_path),
        input_names=["image"],
        output_names=[output_name_for_graph],
        dynamic_axes={
            "image": {0: "batch"},
            output_name_for_graph: {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )

    quantization = "none"
    if quantize:
        _quantize_dynamic_uint8(export_path, output_path)
        quantization = "dynamic_uint8"
        if not keep_float:
            export_path.unlink(missing_ok=True)

    max_abs_diff = _check_onnx_export(wrapper, output_path, dummy) if check else None
    return output_path, max_abs_diff, quantization


def update_manifest(
    deployment_dir: Path,
    onnx_path: Path,
    keep_pytorch: bool,
    quantization: str,
    embedding_output: bool,
) -> dict:
    manifest = load_manifest(deployment_dir)
    updated = copy.deepcopy(manifest)

    if embedding_output and _manifest_uses_web_head(manifest):
        updated["model_type"] = "onnx_embedding_prototype"
        updated["runtime"] = "onnxruntime"
        updated["head_type"] = "prototype"
    else:
        updated["model_type"] = "onnx_classifier"
        updated["runtime"] = "onnxruntime"
        updated["head_type"] = "linear_classifier"
    updated["quantization"] = quantization
    updated.setdefault("files", {})
    updated["files"]["model"] = onnx_path.name
    updated["files"]["manifest"] = "manifest.json"
    updated.setdefault("sha256", {})
    updated["sha256"]["model"] = sha256_file(onnx_path)

    if not keep_pytorch:
        old_model = OfflineDeployment(root=deployment_dir, manifest=manifest).model_file
        if old_model.exists() and old_model.resolve() != onnx_path.resolve():
            old_model.unlink()

    manifest_path = deployment_dir / "manifest.json"
    backup_path = deployment_dir / "manifest.pytorch.json"
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the current offline production deployment to ONNX."
    )
    parser.add_argument(
        "--deployment-dir",
        default=str(DEFAULT_DEPLOYMENT_DIR),
        help="Directory containing manifest.json and the deployed PyTorch model.",
    )
    parser.add_argument("--output-name", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Export full-precision ONNX instead of the default dynamic UINT8 model.",
    )
    parser.add_argument(
        "--keep-float",
        action="store_true",
        help="Keep the intermediate full-precision ONNX file when quantizing.",
    )
    parser.add_argument(
        "--keep-pytorch",
        action="store_true",
        help="Keep the original .pth/.pt model next to the ONNX file.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the ONNX Runtime output check after export.",
    )
    parser.add_argument(
        "--check-tolerance",
        type=float,
        default=0.25,
        help="Maximum accepted absolute difference for the export check.",
    )
    parser.add_argument(
        "--allow-head-mismatch",
        action="store_true",
        help="Allow logits-only ONNX export even when the manifest records a prototype/KNN production head.",
    )
    parser.add_argument(
        "--embedding-output",
        action="store_true",
        help="Export model embeddings instead of logits. Use this for prototype-head parity without bundling PyTorch.",
    )
    args = parser.parse_args()

    if args.embedding_output and args.output_name == "model.onnx":
        args.output_name = "embedding_model.onnx"

    deployment_dir = Path(args.deployment_dir)
    onnx_path, max_abs_diff, quantization = export_onnx(
        deployment_dir,
        args.output_name,
        args.opset,
        check=not args.skip_check,
        quantize=not args.no_quantize,
        keep_float=args.keep_float,
        allow_head_mismatch=args.allow_head_mismatch,
        embedding_output=args.embedding_output,
    )
    if max_abs_diff is not None and max_abs_diff > args.check_tolerance:
        raise RuntimeError(
            f"ONNX export check failed: max_abs_diff={max_abs_diff:.6g} "
            f"> tolerance={args.check_tolerance:.6g}"
        )

    manifest = update_manifest(
        deployment_dir,
        onnx_path,
        keep_pytorch=args.keep_pytorch,
        quantization=quantization,
        embedding_output=args.embedding_output,
    )

    print("Exported offline production model to ONNX:")
    print(f"  model:    {onnx_path}")
    print(f"  manifest: {deployment_dir / 'manifest.json'}")
    print(f"  format:   ONNX ({quantization})")
    print(f"  sha256:   {manifest['sha256']['model']}")
    if max_abs_diff is not None:
        print(f"  check:    max_abs_diff={max_abs_diff:.6g}")


if __name__ == "__main__":
    main()
