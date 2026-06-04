from pathlib import Path

import numpy as np

from otitenet.logging.dvclive_tracking import (
    branch_dvclive_log_dir,
    compact_dvclive_metrics,
    compact_dvclive_params,
    dvc_experiment_branch_name,
    dvc_pointer_versions,
    values_latest_metrics,
)


def test_dvc_pointer_versions_matches_nested_dataset_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data.dvc").write_text(
        "outs:\n"
        "- md5: abc123.dir\n"
        "  size: 10\n"
        "  nfiles: 2\n"
        "  hash: md5\n"
        "  path: data\n",
        encoding="utf-8",
    )

    payload = dvc_pointer_versions(["data/otite_ds_64/USA"])

    assert payload["selected"]["data/otite_ds_64/USA"]["md5"] == "abc123.dir"
    assert payload["selected"]["data/otite_ds_64/USA"]["dvc_file"] == "data.dvc"


def test_values_latest_metrics_flattens_training_values():
    metrics = values_latest_metrics(
        {
            "rec_loss": [0.5],
            "valid": {"mcc": [0.1, np.float64(0.8)], "acc": [0.7]},
            "test": {"mcc": [float("nan")], "acc": [0.6]},
        }
    )

    assert metrics["rec_loss"] == 0.5
    assert metrics["valid/mcc"] == 0.8
    assert metrics["valid/acc"] == 0.7
    assert "test/mcc" not in metrics
    assert metrics["test/acc"] == 0.6


def test_compact_dvclive_metrics_excludes_system_noise():
    compact = compact_dvclive_metrics(
        {
            "system": {"cpu": {"usage (%)": 99}},
            "valid": {"mcc": 0.5, "acc": 0.75, "unused": 123},
            "test": {"mcc": -0.1},
            "final": {
                "best_mcc": 0.6,
                "duration_seconds": 12.5,
                "batch": {"batch_nmi": 0.2, "other": 9},
                "run": {"pruned": 1, "unexpected": 5},
            },
            "step": 3,
        }
    )

    assert "system" not in compact
    assert compact["valid"] == {"acc": 0.75, "mcc": 0.5}
    assert compact["test"] == {"mcc": -0.1}
    assert compact["final"]["best_mcc"] == 0.6
    assert compact["final"]["batch"] == {"batch_nmi": 0.2}
    assert compact["final"]["run"] == {"pruned": 1}
    assert compact["step"] == 3


def test_compact_dvclive_params_keeps_reproducibility_pointers_without_status_dump():
    compact = compact_dvclive_params(
        {
            "args": {
                "exp_id": "TEST",
                "task": "notNormal",
                "model_name": "resnet18",
                "path": "data/otite_ds_64/USA",
                "random_recs": 1,
                "git_status_short": "ignored",
            },
            "optimized_params": {"lr": 0.001, "wd": 0.0},
            "code": {
                "git_commit": "abc",
                "git_branch": "master",
                "git_dirty": True,
                "git_status_short": "large noisy diff",
            },
            "dvc": {
                "tracked_outs": [{"path": "data"}],
                "selected": {
                    "data/otite_ds_64/USA": {
                        "dvc_file": "data.dvc",
                        "path": "data",
                        "md5": "abc.dir",
                        "size": 10,
                    }
                },
            },
        }
    )

    assert compact["args"]["exp_id"] == "TEST"
    assert "random_recs" not in compact["args"]
    assert compact["optimized_params"]["lr"] == 0.001
    assert compact["code"] == {"git_commit": "abc", "git_branch": "master", "git_dirty": True}
    assert compact["dvc"] == {
        "selected_paths": "data/otite_ds_64/USA",
        "selected_dvc_files": "data.dvc",
        "selected_md5s": "abc.dir",
    }
    assert "tracked_outs" not in compact["dvc"]
    assert "git_status_short" not in compact["code"]


def test_branch_dvclive_log_dir_uses_safe_stable_components():
    assert branch_dvclive_log_dir("feature/exp logs", "run:01") == (
        "logs/dvc_exp/branches/feature-exp-logs/run-01"
    )


def test_dvc_experiment_branch_name_uses_current_branch_namespace():
    assert dvc_experiment_branch_name("master", "notNormal-run:01") == (
        "dvc-exp/master/notNormal-run-01"
    )


def test_trainae_defaults_to_dvclive_and_not_comet():
    source = Path("src/otitenet/train/train_triplet_new.py").read_text(encoding="utf-8")

    assert "log_comet=False" in source
    assert "log_dvclive=True" in source
    assert "parser.add_argument('--log_comet', type=int, default=0" in source
    assert "parser.add_argument('--log_dvclive', type=int, default=1" in source
    assert "parser.add_argument('--dvclive_branch_exp', type=int, default=1" in source
