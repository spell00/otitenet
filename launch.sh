#!/usr/bin/env bash

n_epochs=1000
task=notNormal
i=0
num_jobs=2  # Max number of concurrent jobs
job_count=0
new_size=224
early_stop=5
n_trials=10
log_file="error_log.txt"
auto_select_k=1


# Clear previous log file
echo "Error Log - $(date)" > $log_file

run_experiment() {
    local model=$1
    local fgsm=$2
    local n_calibration=$3
    local loss=$4
    local dloss=$5
    local prototype=$6
    local n_positives=$7
    local n_negatives=$8
    local normalize=$9
    local auto_select_k=${10}

    echo "Launching job: model=$model, fgsm=$fgsm, n_calibration=$n_calibration, loss=$loss, dloss=$dloss, prototype=$prototype, n_positives=$n_positives, n_negatives=$n_negatives"

    PYTHONPATH="/home/simon/otitenet" /home/simon/otitenet/.conda/bin/python -m otitenet.train.train_triplet_new \
        --n_calibration=$n_calibration --is_stn=0 --dloss=$dloss --device=cuda:0 \
        --classif_loss=$loss --n_epochs=$n_epochs --task=$task --weighted_sampler=0 \
        --groupkfold=1 --n_negatives=$n_negatives --n_positives=$n_positives --prototypes_to_use=$prototype --fgsm=$fgsm \
        --new_size=$new_size --early_stop=$early_stop --n_trials=$n_trials --normalize=$normalize --auto_select_k=$auto_select_k > "logs/job_${i}.log" 2>> "logs/job_${i}_error.log"

    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Error encountered with the following parameters:" >> $log_file
        echo "Timestamp: $(date)" >> $log_file
        echo "model=$model, fgsm=$fgsm, n_calibration=$n_calibration, loss=$loss, dloss=$dloss, prototype=$prototype, n_positives=$n_positives, n_negatives=$n_negatives" >> $log_file
        echo "Exit Code: $exit_code" >> $log_file
        echo "-----------------------------------------------------" >> $log_file
    fi

    job_count=$(( job_count - 1 ))
}

mkdir -p logs  # Ensure logs directory exists

for model in resnet18 vgg16 efficientnet_b0 vit; do
    for fgsm in 0 1; do
        for n_calibration in 0 4; do
            for normalize in yes no; do
                for loss in arcface triplet; do
                    for dloss in no inverseTriplet; do
                        for prototype in batch no class; do
                            if ! [[ "$prototype" == "batch" && "$dloss" == "no" ]]; then
                                while [ $job_count -ge $num_jobs ]; do
                                    wait -n  # Wait for at least one job to finish before launching new one
                                    job_count=$(( job_count - 1 ))
                                done
                                run_experiment "$model" "$fgsm" "$n_calibration" "$loss" "$dloss" "$prototype" 1 1 "$normalize" "$auto_select_k" &
                                job_count=$(( job_count + 1 ))
                            fi
                        done
                    done
                done

                for dloss in no inverseTriplet; do
                    for classif_loss in ce hinge; do
                        while [ $job_count -ge $num_jobs ]; do
                            wait -n
                            job_count=$(( job_count - 1 ))
                        done
                        run_experiment "$model" "$fgsm" "$n_calibration" "$classif_loss" "$dloss" "no" 1 1 "$normalize" "$auto_select_k" &
                        job_count=$(( job_count + 1 ))
                    done
                done

                for dloss in no inverse_softmax_contrastive inverseTriplet; do
                    while [ $job_count -ge $num_jobs ]; do
                        wait -n
                        job_count=$(( job_count - 1 ))
                    done
                    run_experiment "$model" "$fgsm" "$n_calibration" "softmax_contrastive" "$dloss" "no" 1 1 "$normalize" "$auto_select_k" &
                    job_count=$(( job_count + 1 ))
                done

                for loss in arcface; do
                    for dloss in no inverseTriplet; do
                        while [ $job_count -ge $num_jobs ]; do
                            wait -n
                            job_count=$(( job_count - 1 ))
                        done
                        run_experiment "$model" "$fgsm" "$n_calibration" "$loss" "$dloss" "no" 1 1 "$normalize" "$auto_select_k" &
                        job_count=$(( job_count + 1 ))
                    done
                done

                for n_positives in 1; do
                    for n_negatives in 1 5; do
                        for dloss in no inverseTriplet; do
                            while [ $job_count -ge $num_jobs ]; do
                                wait -n
                                job_count=$(( job_count - 1 ))
                            done
                            run_experiment "$model" "$fgsm" "$n_calibration" "triplet" "$dloss" "no" "$n_positives" "$n_negatives" "$normalize" "$auto_select_k" &
                            job_count=$(( job_count + 1 ))
                        done
                    done
                done
            done
        done
    done
done

wait
echo "All jobs completed"
```