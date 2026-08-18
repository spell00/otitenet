from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.paper.run_inference_fraction_experiments import build_cmd


def test_fraction_launcher_records_preassigned_inference_count_as_n_calibration():
    cfg = pd.Series({
        "model_name": "resnet18", "fgsm": 0, "prototypes": "no",
        "n_positives": 1, "n_negatives": 1, "dloss": "no",
        "dist_fct": "cosine", "classif_loss": "ce", "normalize": "yes", "knn": 1,
    })
    scenario = pd.Series({
        "scenario_label": "0p5", "dataset_path": "data/scenario", "inference_train": 131,
    })
    args = Namespace(
        experiment_label="inference_fraction_hist_v2", n_trials=20, n_epochs=1000,
        early_stop=20, num_workers=4,
    )

    cmd = build_cmd(cfg, scenario, rank=1, seed=42, args=args)

    assert cmd[cmd.index("--n_calibration") + 1] == "131"
    assert cmd[cmd.index("--calibration_preassigned_train") + 1] == "1"
    assert cmd[cmd.index("--dvclive_monitor_system") + 1] == "0"
