# OtiteAI paper reproduction commands

Branch: `paper-reproduction-experiments`

This is the command index for the manuscript experiments. It keeps the paper runs narrow: only the current top four-class model family is rerun, rather than the full historical grid.

## Environment

```bash
cd /home/simon/otitenet
git switch paper-reproduction-experiments
/home/simon/otitenet/.conda/bin/dvc pull
```

Check GPU and historical timing:

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/estimate_gpu_time.py
```

Current VM GPU: NVIDIA A100-SXM4-40GB. From 434 existing four-class run summaries, the median run time is about 8.7 minutes and the interquartile range is about 6.0-12.5 minutes. Budget 15-25 minutes per fresh run to allow for setup, logging, early-stop variation, and system load. The timing script writes the actual observed distribution to `paper_outputs/tables/gpu_time_estimates.csv`.

## Print the numbered commands

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/run_paper_experiments.py
```

## E01. Target-source zero-shot top model

Purpose: train/evaluate the current top model family with the local/Quebec target source held out and no target calibration images.

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/run_paper_experiments.py --run E01
```

Approximate GPU time: usually 6-13 minutes on the current A100; budget 15-25 minutes.

After it finishes, note the new `logs/otitis_four_class/<run_id>` folder. That folder contains `splits/test.csv`, `test_predictions.csv`, `valid_predictions.csv`, `run_summary.json`, and the files needed for the paper tables.

## E02. Target-source four-shot calibration

Purpose: repeat E01, but move exactly four proportional images from the held-out validation split and four proportional images from the held-out target test split into the calibration/train support set. Those images must be removed from their original evaluation splits.

First create the calibration manifest from the E01 valid/test splits:

```bash
ZERO_SHOT_RUN=logs/otitis_four_class/<E01_run_id>
/home/simon/otitenet/.conda/bin/python scripts/paper/prepare_test_calibration_manifest.py \
  --split-csv "$ZERO_SHOT_RUN/splits/test.csv" \
  --valid-split-csv "$ZERO_SHOT_RUN/splits/valid.csv" \
  --n 4 \
  --seed 42 \
  --output paper_outputs/manifests/GMFUNL_jan2023_valid_test_n4_seed42.csv
```

Then run the matched four-shot experiment:

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/run_paper_experiments.py --run E02
```

Approximate GPU time: usually 6-13 minutes on the current A100; budget 15-25 minutes.

The manuscript should report the change from E01 to E02 directly: delta test MCC, delta accuracy, false positives, false negatives, and the final post-calibration test-set size.

## E03. Optional target support curve

Purpose: estimate whether more target support images produce a useful curve rather than only a 0-vs-4 comparison.

Create one manifest per support size and seed from the same E01 valid/test splits:

```bash
ZERO_SHOT_RUN=logs/otitis_four_class/<E01_run_id>
for N in 4 8 12 16 20; do
  /home/simon/otitenet/.conda/bin/python scripts/paper/prepare_test_calibration_manifest.py \
    --split-csv "$ZERO_SHOT_RUN/splits/test.csv" \
    --valid-split-csv "$ZERO_SHOT_RUN/splits/valid.csv" \
    --n "$N" \
    --seed 42 \
    --output "paper_outputs/manifests/GMFUNL_jan2023_valid_test_n${N}_seed42.csv"
done
```

Run the curve:

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/run_paper_experiments.py \
  --run-support-curve \
  --support-sizes 0,4,8,12,16,20 \
  --seeds 42 \
  --n-trials 1
```

Approximate GPU time: about one E01/E02 runtime per support-size/seed combination. With six support sizes and one seed, expect roughly 40-90 minutes and budget 1-3 hours. With three seeds, expect roughly 2-5 hours and budget 3-9 hours.

## E04. Paper tables and figures

Purpose: generate the reproducible files for the manuscript from finished `PAPER_*` runs.

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/run_paper_experiments.py --run E04
```

Outputs:

- `paper_outputs/tables/paper_run_summary.csv`
- `paper_outputs/tables/paper_run_summary.md`
- `paper_outputs/tables/missed_cases_for_physician_review.csv`
- `paper_outputs/tables/missed_cases_for_physician_review.md`
- `paper_outputs/figures/support_curve_test_mcc.png`
- `paper_outputs/figures/confusion_<run_tag>_test.png`
- `paper_outputs/figures/confusion_<run_tag>_valid.png`

Use `missed_cases_for_physician_review.csv` as the placeholder table for the physician-agreement section. It lists exactly which images were missed, the reference label, the model prediction, and whether the error is a binary false positive or false negative.

## E05. Legacy broad analysis

Purpose: refresh the older broad analysis from all historical logs. This is useful for appendix/exploration, but it should not be the main claim because the manuscript is now centered on source-aware target calibration.

```bash
/home/simon/otitenet/.conda/bin/python scripts/paper/run_paper_experiments.py --run E05
```

## Manuscript reporting notes

- Fill in the real Quebec/local target image count from the locked test split.
- If `n_calibration > 0`, report both the calibration count and the remaining held-out test count.
- Keep the 0-shot and 4-shot comparison paired by seed, split, and top-model configuration.
- Report false negatives separately from false positives. For triage, fewer false negatives may be preferable because missing abnormal cases is usually worse than sending extra images for review, but this is an intended-use decision and should be stated as such.
- Mark the support-curve experiment as unfinished until all predefined support sizes/seeds are run and summarized.
