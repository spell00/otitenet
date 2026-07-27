# -*- mode: python ; coding: utf-8 -*-

import os
import json
from pathlib import Path

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


ROOT = Path(SPECPATH).parents[1]
SRC = ROOT / "src"


def existing_data(path, target=None):
    source = ROOT / path
    if source.exists():
        destination = target or (str(Path(path).parent) if source.is_file() else path)
        return [(str(source), destination)]
    return []


def include_streamlit_submodule(name):
    return not name.startswith("streamlit.external.langchain")


def deployment_datas():
    deployment_dir = ROOT / "data" / "mobile_deployments" / "current"
    manifest_path = deployment_dir / "manifest.json"
    if not manifest_path.exists():
        return []

    bundled = [(str(manifest_path), "data/mobile_deployments/current")]
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return bundled

    for filename in (manifest.get("files") or {}).values():
        source = deployment_dir / Path(str(filename)).name
        if source.exists() and source.is_file() and source != manifest_path:
            bundled.append((str(source), "data/mobile_deployments/current"))
    return bundled


hiddenimports = []
hiddenimports += [
    "otitenet.offline",
    "otitenet.offline.deployment",
    "otitenet.offline.history",
    "otitenet.offline.predictor",
    "otitenet.app.services.gradcam_service",
]
hiddenimports += [
    "joblib",
    "joblib.numpy_pickle",
    "sklearn",
    "sklearn.discriminant_analysis",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.naive_bayes",
    "sklearn.neural_network",
    "sklearn.svm",
    "sklearn.svm._base",
    "sklearn.svm._classes",
    "sklearn.svm._liblinear",
    "sklearn.svm._libsvm",
    "sklearn.svm._libsvm_sparse",
    "sklearn.tree",
]
hiddenimports += collect_submodules("streamlit", filter=include_streamlit_submodule)

excludes = [
    # Documentation/test/notebook packages that PyInstaller hooks may probe.
    "alabaster",
    "IPython",
    "jupyter",
    "jupyterlab",
    "matplotlib",
    "notebook",
    "nvidia",
    "nvidia.cublas",
    "nvidia.cuda_cupti",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.cufile",
    "nvidia.curand",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.cusparselt",
    "nvidia.nccl",
    "nvidia.nvjitlink",
    "nvidia.nvshmem",
    "nvidia.nvtx",
    "pytest",
    "seaborn",
    "sphinx",
    "sphinx.application",
    # Optional database drivers not used by the offline app.
    "MySQLdb",
    "pysqlite2",
    "sqlalchemy",
    # DVC and related - not used by offline desktop app
    "dvc",
    "dvclive",
    # Online/training/analysis dependencies not used by app_offline.py.
    "comet_ml",
    "cv2",
    "kaleido",
    "langchain",
    "mlflow",
    "optuna",
    "plotly",
    "shap",
    "skopt",
    "tensorboard",
    "tensorboardX",
    "tensorflow",
    "tf_keras",
    "torch.distributed",
    "torch.utils.tensorboard",
    "triton",
    "xgboost",
    # GUI backends. The offline app runs through Streamlit in a browser/webview.
    "_tkinter",
    "tkinter",
]

if os.environ.get("OTITENET_DESKTOP_VARIANT", "compact").lower() not in {"exact", "full", "torch"}:
    excludes += [
        "torch",
        "torchvision",
        "torchaudio",
        "torchgen",
        "functorch",
    ]

datas = []
datas += existing_data("app_offline.py")
datas += existing_data("styles.css")
datas += existing_data(".streamlit")
datas += copy_metadata("streamlit")
datas += collect_data_files("streamlit")
datas += deployment_datas()

if os.environ.get("OTITENET_BUNDLE_DATA") == "1":
    for directory in ("data",):
        source = ROOT / directory
        if source.exists():
            datas += [Tree(str(source), prefix=directory)]


a = Analysis(
    [str(ROOT / "packaging" / "pyinstaller" / "streamlit_entrypoint.py")],
    pathex=[str(SRC), str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "packaging" / "pyinstaller" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

if os.environ.get("OTITENET_PYINSTALLER_ONEFILE") == "1":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="otitenet-streamlit",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="otitenet-streamlit",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="otitenet-streamlit",
    )
