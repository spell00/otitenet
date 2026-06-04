"""DVCLive tracking helpers for training runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is part of DVC deps
    yaml = None


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_scalar(v) for k, v in value.items()}
    try:
        return value.item()
    except Exception:
        return str(value)


def _run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def code_version_payload() -> dict[str, Any]:
    """Return Git/code state useful for reproducing a run."""
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--short"])
    requirements_hash = ""
    req_path = Path("requirements.txt")
    if req_path.exists():
        requirements_hash = hashlib.sha256(req_path.read_bytes()).hexdigest()
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_status_short": status,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "requirements_sha256": requirements_hash,
    }


def _dvc_outs_from_file(dvc_file: Path) -> list[dict[str, Any]]:
    if yaml is None or not dvc_file.exists():
        return []
    try:
        payload = yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    rows = []
    for out in payload.get("outs", []) or []:
        if not isinstance(out, dict):
            continue
        out_path = str(out.get("path", "")).replace("\\", "/").strip("/")
        if not out_path:
            continue
        # DVC output paths are relative to the .dvc file directory.
        full_path = (dvc_file.parent / out_path).as_posix().strip("./")
        rows.append(
            {
                "dvc_file": dvc_file.as_posix(),
                "path": full_path,
                "hash": out.get("hash", "md5"),
                "md5": out.get("md5"),
                "size": out.get("size"),
                "nfiles": out.get("nfiles"),
            }
        )
    return rows


def dvc_pointer_versions(paths: list[str | Path] | None = None) -> dict[str, Any]:
    """Collect DVC pointer hashes for all .dvc files, plus matches for selected paths."""
    all_outs: list[dict[str, Any]] = []
    for dvc_file in sorted(Path(".").rglob("*.dvc")):
        if ".dvc/cache" in dvc_file.as_posix():
            continue
        all_outs.extend(_dvc_outs_from_file(dvc_file))

    selected: dict[str, Any] = {}
    for raw_path in paths or []:
        wanted = str(raw_path or "").replace("\\", "/").strip().strip("/")
        for prefix in ("./",):
            if wanted.startswith(prefix):
                wanted = wanted[len(prefix) :]
        best = None
        for out in all_outs:
            out_path = str(out.get("path", "")).strip("/")
            if wanted == out_path or wanted.startswith(out_path + "/"):
                if best is None or len(out_path) > len(str(best.get("path", ""))):
                    best = out
        if best is not None:
            selected[wanted] = best

    return {
        "tracked_outs": all_outs,
        "selected": selected,
    }


def values_latest_metrics(values: dict[str, Any]) -> dict[str, float]:
    """Flatten the latest value from the training values dict."""
    metrics: dict[str, float] = {}
    for key in ("rec_loss", "dom_loss", "dom_acc"):
        seq = values.get(key, []) if isinstance(values, dict) else []
        if seq:
            val = _safe_scalar(seq[-1])
            if isinstance(val, (int, float)):
                metrics[key] = float(val)
    for group, group_values in (values or {}).items():
        if not isinstance(group_values, dict):
            continue
        for metric_name, seq in group_values.items():
            if not isinstance(seq, (list, tuple)) or not seq:
                continue
            val = _safe_scalar(seq[-1])
            if isinstance(val, (int, float)):
                metrics[f"{group}/{metric_name}"] = float(val)
    return metrics


def _resolve_dvcyaml_setting(args: Any, base_dir: str) -> str | None:
    """Resolve CLI dvcyaml setting into a value accepted by dvclive.Live."""
    raw = str(getattr(args, "dvclive_dvcyaml", "none") or "none").strip()
    lowered = raw.lower()

    if lowered in {"", "0", "false", "off", "none", "null", "disable", "disabled"}:
        return None

    if lowered in {"1", "true", "on", "auto"}:
        return os.path.join(base_dir, "dvc.yaml")

    # Any non-empty custom path is accepted.
    return raw


def _safe_exp_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return text[:120] or "otitenet-run"


def _safe_path_component(value: str, fallback: str = "unknown") -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return text[:120] or fallback


def branch_dvclive_log_dir(branch: str, exp_name: str) -> str:
    """Return the branch/run-specific DVC log mirror directory."""
    return os.path.join(
        "logs",
        "dvc_exp",
        "branches",
        _safe_path_component(branch, "detached"),
        _safe_path_component(exp_name, "otitenet-run"),
    )


def dvc_experiment_branch_name(branch: str, exp_name: str) -> str:
    """Return the Git branch used when promoting a saved DVC experiment."""
    return "/".join(
        [
            "dvc-exp",
            _safe_path_component(branch, "detached"),
            _safe_path_component(exp_name, "otitenet-run"),
        ]
    )


def _artifact_id_from_path(artifact: str, index: int) -> str:
    """Build a DVCLive-safe artifact ID from a file path."""
    base = Path(artifact).stem or "artifact"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
    if not safe:
        safe = "artifact"
    if not safe[0].isalpha():
        safe = f"artifact_{safe}"
    return f"{safe}_{index}"


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    if yaml is None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _pick_keys(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in payload and payload.get(key) is not None:
            out[key] = _safe_scalar(payload.get(key))
    return out


def compact_dvclive_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a DVC Experiments friendly subset of DVCLive metrics."""
    groups = {}
    for group_name, keys in {
        "train": ["closs", "dloss", "dom_acc"],
        "valid": ["closs", "dloss", "acc", "mcc", "tpr", "tnr", "ppv", "npv"],
        "test": ["closs", "dloss", "acc", "mcc", "tpr", "tnr", "ppv", "npv"],
        "timing": ["deep_train_s", "classifier_fit_s", "eval_s", "prototypes_s", "loader_refresh_s", "logging_s"],
    }.items():
        value = metrics.get(group_name)
        if isinstance(value, dict):
            picked = _pick_keys(value, keys)
            if picked:
                groups[group_name] = picked

    final = metrics.get("final")
    if isinstance(final, dict):
        picked_final = _pick_keys(final, ["best_mcc", "best_acc", "best_closs", "duration_seconds"])
        batch = final.get("batch")
        if isinstance(batch, dict):
            picked_batch = _pick_keys(
                batch,
                [
                    "batch_entropy",
                    "batch_entropy_loss",
                    "batch_acc_loss",
                    "batch_mcc_loss",
                    "batch_nmi",
                    "batch_ari",
                    "batch_ami",
                    "batch_silhouette",
                ],
            )
            if picked_batch:
                picked_final["batch"] = picked_batch
        run = final.get("run")
        if isinstance(run, dict):
            picked_run = _pick_keys(run, ["pruned", "failed"])
            if picked_run:
                picked_final["run"] = picked_run
        if picked_final:
            groups["final"] = picked_final

    if "step" in metrics:
        groups["step"] = _safe_scalar(metrics.get("step"))
    return groups


def compact_dvclive_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a compact params file for DVC Experiments columns."""
    args = params.get("args") if isinstance(params.get("args"), dict) else {}
    out: dict[str, Any] = {
        "args": _pick_keys(
            args,
            [
                "exp_id",
                "uuid",
                "task",
                "model_name",
                "path",
                "path_original",
                "train_datasets",
                "valid_dataset",
                "test_dataset",
                "effective_train_datasets",
                "effective_valid_dataset",
                "effective_test_dataset",
                "new_size",
                "normalize",
                "n_epochs",
                "n_trials",
                "trial_index",
                "bs",
                "dloss",
                "classif_loss",
                "prototypes_to_use",
                "prototype_strategy",
                "prototype_components",
                "siamese_inference",
                "n_neighbors",
                "fgsm",
                "n_calibration",
                "n_positives",
                "n_negatives",
                "dist_fct",
                "n_aug",
                "seed",
                "groupkfold",
                "weighted_sampler",
                "device",
            ],
        )
    }
    optimized = params.get("optimized_params")
    if isinstance(optimized, dict):
        out["optimized_params"] = _pick_keys(optimized, sorted(optimized.keys()))

    run = params.get("run")
    if isinstance(run, dict):
        out["run"] = _pick_keys(run, ["foldername", "complete_log_path"])

    code = params.get("code")
    if isinstance(code, dict):
        out["code"] = _pick_keys(code, ["git_commit", "git_branch", "git_dirty", "requirements_sha256"])

    dvc_payload = params.get("dvc")
    if isinstance(dvc_payload, dict):
        selected = dvc_payload.get("selected")
        if isinstance(selected, dict):
            selected_paths = []
            selected_dvc_files = []
            selected_md5s = []
            for key, value in selected.items():
                if isinstance(value, dict):
                    picked = _pick_keys(value, ["dvc_file", "path", "md5"])
                    selected_paths.append(str(key))
                    if picked.get("dvc_file") is not None:
                        selected_dvc_files.append(str(picked["dvc_file"]))
                    if picked.get("md5") is not None:
                        selected_md5s.append(str(picked["md5"]))
            if selected_paths:
                out["dvc"] = {
                    "selected_paths": ";".join(selected_paths),
                    "selected_dvc_files": ";".join(sorted(set(selected_dvc_files))),
                    "selected_md5s": ";".join(selected_md5s),
                }

    return {key: value for key, value in out.items() if value}


class DVCLiveTracker:
    """Thin wrapper around DVCLive with defensive no-op behavior."""

    def __init__(
        self,
        args: Any,
        params: dict[str, Any],
        complete_log_path: str,
        foldername: str,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.live = None
        self.dir = os.path.join(complete_log_path, "dvclive")
        self.complete_log_path = complete_log_path
        self.foldername = foldername
        self.save_dvc_exp = bool(int(getattr(args, "dvclive_save_dvc_exp", 1)))
        self.branch_dvc_exp = bool(int(getattr(args, "dvclive_branch_exp", 1)))
        self.git_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "detached"
        self.exp_name = _safe_exp_name(
            f"{getattr(args, 'task', 'otitenet')}-{getattr(args, 'exp_id', foldername)}-{foldername}"
        )
        self.branch_log_dir = branch_dvclive_log_dir(self.git_branch, self.exp_name)
        if not self.enabled:
            return
        try:
            from dvclive import Live
        except Exception as exc:
            print(f"[DVCLive] disabled: could not import dvclive ({exc})")
            self.enabled = False
            return

        try:
            monitor_system = bool(int(getattr(args, "dvclive_monitor_system", 1)))
            self.live = Live(
                dir=self.dir,
                resume=False,
                report="md",
                # We save the DVC experiment ourselves after publishing stable
                # logs/dvc_exp files. This avoids sparse exp rows and duplicate
                # per-run dvc.yaml files.
                save_dvc_exp=False,
                dvcyaml=False,
                monitor_system=monitor_system,
            )
            self.log_start(args, params, foldername)
        except Exception as exc:
            print(f"[DVCLive] disabled: failed to initialize ({exc})")
            self.enabled = False
            self.live = None

    def log_start(self, args: Any, params: dict[str, Any], foldername: str) -> None:
        if self.live is None:
            return
        args_payload = {k: _safe_scalar(v) for k, v in vars(args).items()}
        params_payload = {k: _safe_scalar(v) for k, v in (params or {}).items()}
        dvc_payload = dvc_pointer_versions(
            [
                getattr(args, "path", ""),
                "data",
                "logs/best_models",
                "configs/datasets.csv",
                "configs/best_models.csv",
            ]
        )
        payload = {
            "args": args_payload,
            "optimized_params": params_payload,
            "run": {
                "foldername": foldername,
                "complete_log_path": self.complete_log_path,
            },
            "code": code_version_payload(),
            "dvc": dvc_payload,
        }
        self.live.log_params(payload)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "repro_context.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def log_epoch(self, values: dict[str, Any], epoch: int, extra: dict[str, Any] | None = None) -> None:
        if self.live is None:
            return
        try:
            for name, value in values_latest_metrics(values).items():
                self.live.log_metric(name, value)
            for name, value in (extra or {}).items():
                value = _safe_scalar(value)
                if isinstance(value, (int, float)):
                    self.live.log_metric(name, float(value))
            self.live.next_step()
            # Keep stable DVC-facing files fresh during training, not only at run end.
            self._publish_latest_for_dvc()
        except Exception as exc:
            print(f"[DVCLive] warning: failed to log epoch {epoch + 1}: {exc}")

    def log_final(self, metrics: dict[str, Any] | None = None, artifacts: list[str] | None = None) -> None:
        if self.live is None:
            return
        try:
            for name, value in (metrics or {}).items():
                value = _safe_scalar(value)
                if isinstance(value, (int, float)):
                    self.live.log_metric(f"final/{name}", float(value), plot=False)
            for index, artifact in enumerate(artifacts or [], start=1):
                if artifact and os.path.exists(artifact):
                    artifact_id = _artifact_id_from_path(artifact, index)
                    self.live.log_artifact(artifact, type="model", name=artifact_id, cache=False)
        except Exception as exc:
            print(f"[DVCLive] warning: failed to log final data: {exc}")

    def end(self) -> None:
        if self.live is None:
            return
        try:
            self.live.end()
            self._publish_latest_for_dvc()
            self._save_dvc_experiment()
        except Exception as exc:
            print(f"[DVCLive] warning: failed to close run: {exc}")

    def _publish_latest_for_dvc(self) -> None:
        """Mirror DVCLive files to stable workspace paths and branch/run history."""
        latest_dir = os.path.join("logs", "dvc_exp")
        os.makedirs(latest_dir, exist_ok=True)
        os.makedirs(self.branch_log_dir, exist_ok=True)

        metrics_src = os.path.join(self.dir, "metrics.json")
        metrics_dst = os.path.join(latest_dir, "latest_metrics.json")
        branch_metrics_dst = os.path.join(self.branch_log_dir, "metrics.json")
        params_src = os.path.join(self.dir, "params.yaml")
        params_dst = os.path.join(latest_dir, "latest_params.yaml")
        branch_params_dst = os.path.join(self.branch_log_dir, "params.yaml")
        if os.path.exists(metrics_src):
            compact_metrics = compact_dvclive_metrics(_load_json(metrics_src))
            with open(metrics_dst, "w", encoding="utf-8") as handle:
                json.dump(compact_metrics, handle, indent=2)
            with open(branch_metrics_dst, "w", encoding="utf-8") as handle:
                json.dump(compact_metrics, handle, indent=2)
        if os.path.exists(params_src):
            compact_params = compact_dvclive_params(_load_yaml(params_src))
            _write_yaml(params_dst, compact_params)
            _write_yaml(branch_params_dst, compact_params)

    def _save_dvc_experiment(self) -> None:
        if not self.save_dvc_exp:
            return
        cmd = [
            "dvc",
            "exp",
            "save",
            "--force",
            "--name",
            self.exp_name,
            "--include-untracked",
            "dvc.yaml",
            "--include-untracked",
            "dvc.lock",
            "--include-untracked",
            os.path.join("logs", "dvc_exp", "latest_metrics.json"),
            "--include-untracked",
            os.path.join("logs", "dvc_exp", "latest_params.yaml"),
            "--include-untracked",
            self.branch_log_dir,
        ]
        try:
            proc = subprocess.run(cmd, cwd=Path.cwd(), check=False, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                print(f"[DVCLive] warning: dvc exp save failed: {msg}")
                return
            self._branch_dvc_experiment()
        except Exception as exc:
            print(f"[DVCLive] warning: dvc exp save failed: {exc}")

    def _branch_dvc_experiment(self) -> None:
        if not self.branch_dvc_exp:
            return
        branch_name = dvc_experiment_branch_name(self.git_branch, self.exp_name)
        cmd = ["dvc", "exp", "branch", self.exp_name, branch_name]
        try:
            proc = subprocess.run(cmd, cwd=Path.cwd(), check=False, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                print(f"[DVCLive] warning: dvc exp branch failed: {msg}")
        except Exception as exc:
            print(f"[DVCLive] warning: dvc exp branch failed: {exc}")
