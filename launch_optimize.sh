#!/usr/bin/env bash

# Launch optimization script to find best hyperparameters for each model
# This script iterates over models and runs parameter optimization with ~100+ trials

n_epochs=1000
task=notNormal
new_size=224
early_stop=10
n_trials=150  # Number of optimization trials - can be increased for more thorough search
log_file="error_log_optimize.txt"
num_jobs=2  # Max number of concurrent jobs
job_count=0

# Clear previous log file
echo "Error Log - $(date)" > $log_file

run_optimization() {
    local model=$1

    echo "Launching optimization job for model: $model"

    PYTHONPATH="/home/simon/otitenet" /home/simon/otitenet/.conda/bin/python -m otitenet.train.train_triplet_new \
        --model_name="$model" --n_calibration="None" --is_stn=0 --dloss="None" --device=cuda:0 \
        --classif_loss="None" --n_epochs=$n_epochs --task=$task --weighted_sampler=0 \
        --groupkfold=1 --n_negatives="None" --n_positives="None" --prototypes_to_use="None" --fgsm="None" \
        --new_size=$new_size --early_stop=$early_stop --n_trials=$n_trials --normalize="None" --auto_select_k=1 > "logs/optimization_${model}.log" 2>> "logs/optimization_${model}_error.log"

    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Error encountered with the following parameters:" >> $log_file
        echo "Timestamp: $(date)" >> $log_file
        echo "model=$model" >> $log_file
        echo "Exit Code: $exit_code" >> $log_file
        echo "-----------------------------------------------------" >> $log_file
    fi
}

mkdir -p logs  # Ensure logs directory exists

# Iterate over models only, letting the training script optimize all other parameters
for model in resnet18 vgg16 efficientnet_b0 vit; do
    while [ $job_count -ge $num_jobs ]; do
        wait -n  # Wait for at least one job to finish before launching new one
        job_count=$(( job_count - 1 ))
    done
    echo "Starting optimization for model: $model"
    run_optimization "$model" &
    job_count=$(( job_count + 1 ))
done

wait
echo "All optimization jobs completed"
