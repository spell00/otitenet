import torch
import torchvision.models as models
from pathlib import Path

def export_mobile_model():
    # 1. Use a lightweight MobileNetV3 as a placeholder
    # In a real scenario, you would load your trained OtiteNet model here
    model = models.mobilenet_v3_small(pretrained=True)
    model.eval()

    # 2. Trace the model with a dummy input (224x224 is standard)
    example_input = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example_input)

    # 3. Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_script_module)

    # 4. Save it to the deployment directory
    deploy_dir = Path("data/mobile_deployments/current")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    model_path = deploy_dir / "model.ptl"
    optimized_model._save_for_lite_interpreter(str(model_path))

    # 5. Create a basic manifest so the app knows what to do
    import json
    manifest = {
        "model_id": 1,
        "model_name": "OtiteNet-Alpha",
        "labels": ["Normal", "Abnormal"],
        "input": {"image_size": [224, 224]},
        "files": {"model": "model.ptl"}
    }
    with open(deploy_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    print(f"Success! Mobile model exported to: {model_path}")

if __name__ == "__main__":
    export_mobile_model()
