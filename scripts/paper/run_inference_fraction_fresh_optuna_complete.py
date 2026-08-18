#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess

import pandas as pd

from run_inference_fraction_optuna_fresh import cnn_command, siamese_command

ROOT = Path(__file__).resolve().parents[2]
TASK = 'four_classes_220726'
FRACTION_CONFIG = {0.50: 131, 0.25: 66, 0.10: 26, 0.05: 13, 0.02: 5, 0.0: 0}
SCENARIO_LABELS = {0.50: '0p5', 0.25: '0p25', 0.10: '0p1', 0.05: '0p05', 0.02: '0p02', 0.0: '0'}
PYTHON = ROOT / '.conda' / 'bin' / 'python'

def _quote(cmd):
    parts = []
    for x in cmd:
        s = str(x)
        if ' ' in s:
            parts.append(f'{s}')
        else:
            parts.append(s)
    return ' '.join(parts)

def _build_optuna_cmd(
    fraction,
    n_calib,
    scenario_label,
    phase,
    seed,
    n_trials,
    n_epochs,
    early_stop,
    num_workers,
    siamese_batch_size,
    cnn_batch_size,
):
    dataset_dir = f'data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference_fraction_hist_v2_train{scenario_label}_seed{seed}'
    scenario = pd.Series({
        'dataset_path': dataset_dir,
        'scenario_label': scenario_label,
        'inference_train': n_calib,
    })
    builder_args = argparse.Namespace(
        seed=seed,
        n_trials=n_trials,
        n_epochs=n_epochs,
        early_stop=early_stop,
        num_workers=num_workers,
        siamese_batch_size=siamese_batch_size,
        cnn_batch_size=cnn_batch_size,
    )
    builders = {'siamese': siamese_command, 'cnn_mlp': cnn_command}
    if phase not in builders:
        raise ValueError(f'Unknown phase {phase!r}; expected one of {sorted(builders)}')
    _, command = builders[phase](scenario, builder_args)
    return command

def main():
    p = argparse.ArgumentParser(description='Fresh Optuna HPO for all inference fractions')
    p.add_argument('--fractions', default='0.5,0.25,0.1,0.05,0.02,0')
    p.add_argument('--phases', default='siamese,cnn_mlp')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-trials', type=int, default=20)
    p.add_argument('--n-epochs', type=int, default=1000)
    p.add_argument('--early-stop', type=int, default=20)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--siamese-batch-size', type=int, default=64)
    p.add_argument('--cnn-batch-size', type=int, default=128)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    
    fractions = [float(x.strip()) for x in args.fractions.split(',')]
    phases = [x.strip() for x in args.phases.split(',')]
    
    print(f'\n{"="*80}')
    print(f'Fresh Optuna HPO - Inference Fractions')
    print(f'{"="*80}')
    print(f'Fractions: {fractions}')
    print(f'Phases: {phases}')
    print(f'Trials per fraction: {args.n_trials}')
    print(f'{"="*80}\n')
    
    total_runs = 0
    failed_runs = 0
    
    for frac in fractions:
        n_calib = FRACTION_CONFIG.get(frac)
        if n_calib is None:
            print(f'❌ Invalid fraction {frac}')
            continue
        
        scenario_label = SCENARIO_LABELS[frac]
        
        for phase in phases:
            print(f'\n{"="*80}')
            print(f'Fraction {frac} ({scenario_label}) | Phase {phase} | n_cal={n_calib}')
            print(f'{"="*80}\n')
            
            cmd = _build_optuna_cmd(
                frac,
                n_calib,
                scenario_label,
                phase,
                args.seed,
                args.n_trials,
                args.n_epochs,
                args.early_stop,
                args.num_workers,
                args.siamese_batch_size,
                args.cnn_batch_size,
            )
            
            if args.dry_run:
                print('[DRY RUN]', _quote(cmd))
            else:
                rc = subprocess.call(cmd, cwd=ROOT)
                total_runs += 1
                if rc != 0:
                    print(f'❌ Failed with code {rc}')
                    failed_runs += 1
                else:
                    print(f'✅ Success')
    
    print(f'\n{"="*80}')
    print(f'Summary: {total_runs} runs, {failed_runs} failures')
    print(f'{"="*80}\n')
    
    return 0 if failed_runs == 0 else 1

if __name__ == '__main__':
    raise SystemExit(main())
