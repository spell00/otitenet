#!/usr/bin/env bash

n_epochs=1000
task=notNormal
num_jobs=12
new_size=224
early_stop=10
n_trials=10
log_file="logs/error_log.txt"
auto_select_k=1
run_cnn_mlp=1
cnn_compare_all=0
verbose=0
test_mode=0
test_tag="PROD"
triplet_log_comet=1
triplet_log_mlflow=0
triplet_log_tracking=0
triplet_save_repro_artifacts=1
bs=32
user_set_num_jobs=0
user_set_bs=0
min_free_mb=2048
mem_per_job_mb=6000
max_oom_retries=3
poll_interval=5

# DANN/FGSM params shared with CNN/MLP.
gamma=1.0
epsilon=0.01

force_run=0
# CLI flags:
#   --test  : run smoke mode (1 epoch/1 trial/single job)
for arg in "$@"; do
    case "$arg" in
        --test)
            test_mode=1
            test_tag="TEST_SMOKE"
            ;;
        --force)
            force_run=1
            ;;
        --jobs=*)
            num_jobs="${arg#*=}"
            user_set_num_jobs=1
            ;;
        --bs=*)
            bs="${arg#*=}"
            user_set_bs=1
            ;;
        --min-free-mb=*)
            min_free_mb="${arg#*=}"
            ;;
        --mem-per-job-mb=*)
            mem_per_job_mb="${arg#*=}"
            ;;
        --max-oom-retries=*)
            max_oom_retries="${arg#*=}"
            ;;
        --poll-interval=*)
            poll_interval="${arg#*=}"
            ;;
    esac
done

# Clear previous log file
echo "Error Log - $(date)" > "$log_file"

if [ "$test_mode" -eq 1 ]; then
    n_epochs=1
    n_trials=1
    early_stop=1
    if [ "$user_set_num_jobs" -eq 0 ]; then
        # Better default for 40GB-class GPUs; override with --jobs=N.
        num_jobs=4
    fi
    # Extra speedups for smoke runs across the full grid.
    auto_select_k=0
    new_size=64
    if [ "$user_set_bs" -eq 0 ]; then
        bs=128
    fi
    triplet_log_comet=0
    triplet_save_repro_artifacts=0
fi

mkdir -p logs

total_models=0
finished_jobs=0

declare -A JOB_KIND
declare -A JOB_MODEL
declare -A JOB_FGSM
declare -A JOB_N_CAL
declare -A JOB_LOSS
declare -A JOB_DLOSS
declare -A JOB_PROTOTYPE
declare -A JOB_N_POS
declare -A JOB_N_NEG
declare -A JOB_NORMALIZE
declare -A JOB_AUTO_K
declare -A JOB_VARIANT
declare -A JOB_CLASSIF
declare -A JOB_EXP_ID
declare -A JOB_RETRY_COUNT
declare -A JOB_ERR_FILE
declare -A JOB_STDOUT_FILE
declare -A JOB_FAILED_FINAL

declare -A PID_TO_JOB_ID

pending_queue=()
retry_queue=()
running_pids=()
next_job_id=1
memory_pressure_active=0
dequeued_job_id=""

progress_tick() {
    echo "[$finished_jobs/$total_models] $1"
}

enqueue_pending() {
    pending_queue+=("$1")
}

enqueue_retry() {
    retry_queue+=("$1")
}

dequeue_retry() {
    if [ ${#retry_queue[@]} -eq 0 ]; then
        return 1
    fi
    dequeued_job_id="${retry_queue[0]}"
    retry_queue=("${retry_queue[@]:1}")
    return 0
}

dequeue_pending() {
    if [ ${#pending_queue[@]} -eq 0 ]; then
        return 1
    fi
    dequeued_job_id="${pending_queue[0]}"
    pending_queue=("${pending_queue[@]:1}")
    return 0
}

dequeue_next_job() {
    if dequeue_retry; then
        return 0
    fi
    if dequeue_pending; then
        return 0
    fi
    return 1
}

gpu_free_mb() {
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')
    if [[ -z "$free" || ! "$free" =~ ^[0-9]+$ ]]; then
        echo 999999
    else
        echo "$free"
    fi
}

dynamic_slot_cap() {
    local free="$1"
    local cap
    if [ "$free" -lt "$min_free_mb" ]; then
        echo 0
        return
    fi
    if [ "$mem_per_job_mb" -le 0 ]; then
        echo "$num_jobs"
        return
    fi
    cap=$(( free / mem_per_job_mb ))
    if [ "$cap" -lt 1 ]; then
        cap=1
    fi
    if [ "$cap" -gt "$num_jobs" ]; then
        cap="$num_jobs"
    fi
    echo "$cap"
}

is_oom_error() {
    local err_file="$1"
    grep -qiE "torch\.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|out of memory" "$err_file"
}

record_failure() {
    local job_id="$1"
    local exit_code="$2"
    local kind="${JOB_KIND[$job_id]}"
    local err_file="${JOB_ERR_FILE[$job_id]}"

    if [ "$kind" = "siamese" ]; then
        {
            echo "Error encountered with siamese parameters:"
            echo "Timestamp: $(date)"
            echo "model=${JOB_MODEL[$job_id]}, fgsm=${JOB_FGSM[$job_id]}, n_calibration=${JOB_N_CAL[$job_id]}, loss=${JOB_LOSS[$job_id]}, dloss=${JOB_DLOSS[$job_id]}, prototype=${JOB_PROTOTYPE[$job_id]}, n_positives=${JOB_N_POS[$job_id]}, n_negatives=${JOB_N_NEG[$job_id]}, normalize=${JOB_NORMALIZE[$job_id]}"
            echo "Exit Code: $exit_code"
            echo "Error file: $err_file"
            echo "-----------------------------------------------------"
        } >> "$log_file"
    else
        {
            echo "Error encountered with CNN/MLP parameters:"
            echo "Timestamp: $(date)"
            echo "model=${JOB_MODEL[$job_id]}, variant=${JOB_VARIANT[$job_id]}, classif_loss=${JOB_CLASSIF[$job_id]}, normalize=${JOB_NORMALIZE[$job_id]}, n_calibration=${JOB_N_CAL[$job_id]}, fgsm=${JOB_FGSM[$job_id]}, dloss=${JOB_DLOSS[$job_id]}"
            echo "Exit Code: $exit_code"
            echo "Error file: $err_file"
            echo "-----------------------------------------------------"
        } >> "$log_file"
    fi
}

register_siamese_job() {
    local model="$1"
    local fgsm="$2"
    local n_calibration="$3"
    local loss="$4"
    local dloss="$5"
    local prototype="$6"
    local n_positives="$7"
    local n_negatives="$8"
    local normalize="$9"
    local exp_id="${10}"

    local job_id="$next_job_id"
    next_job_id=$((next_job_id + 1))
    JOB_EXP_ID[$job_id]="$exp_id"

    if [ "$force_run" -eq 0 ] && [ -f "logs/${exp_id}.done" ]; then
        return
    fi

    JOB_KIND[$job_id]="siamese"
    JOB_MODEL[$job_id]="$model"
    JOB_FGSM[$job_id]="$fgsm"
    JOB_N_CAL[$job_id]="$n_calibration"
    JOB_LOSS[$job_id]="$loss"
    JOB_DLOSS[$job_id]="$dloss"
    JOB_PROTOTYPE[$job_id]="$prototype"
    JOB_N_POS[$job_id]="$n_positives"
    JOB_N_NEG[$job_id]="$n_negatives"
    JOB_NORMALIZE[$job_id]="$normalize"
    JOB_AUTO_K[$job_id]="$auto_select_k"
    JOB_RETRY_COUNT[$job_id]=0
    JOB_ERR_FILE[$job_id]="logs/${exp_id}_error.log"
    JOB_STDOUT_FILE[$job_id]="logs/${exp_id}.log"
    JOB_FAILED_FINAL[$job_id]=0

    enqueue_pending "$job_id"
}

register_cnn_mlp_job() {
    local model="$1"
    local variant="$2"
    local normalize="$3"
    local n_calibration="$4"
    local fgsm="$5"
    local dloss="$6"
    local classif_loss="$7"
    local exp_id="$8"

    local job_id="$next_job_id"
    next_job_id=$((next_job_id + 1))
    JOB_EXP_ID[$job_id]="$exp_id"

    if [ "$force_run" -eq 0 ] && [ -f "logs/${exp_id}.done" ]; then
        return
    fi

    JOB_KIND[$job_id]="cnn_mlp"
    JOB_MODEL[$job_id]="$model"
    JOB_VARIANT[$job_id]="$variant"
    JOB_NORMALIZE[$job_id]="$normalize"
    JOB_N_CAL[$job_id]="$n_calibration"
    JOB_FGSM[$job_id]="$fgsm"
    JOB_DLOSS[$job_id]="$dloss"
    JOB_CLASSIF[$job_id]="$classif_loss"
    JOB_RETRY_COUNT[$job_id]=0
    JOB_ERR_FILE[$job_id]="logs/${exp_id}_error.log"
    JOB_STDOUT_FILE[$job_id]="logs/${exp_id}.log"
    JOB_FAILED_FINAL[$job_id]=0

    enqueue_pending "$job_id"
}

run_siamese_job() {
    local job_id="$1"
    local model="${JOB_MODEL[$job_id]}"
    local fgsm="${JOB_FGSM[$job_id]}"
    local n_calibration="${JOB_N_CAL[$job_id]}"
    local loss="${JOB_LOSS[$job_id]}"
    local dloss="${JOB_DLOSS[$job_id]}"
    local prototype="${JOB_PROTOTYPE[$job_id]}"
    local n_positives="${JOB_N_POS[$job_id]}"
    local n_negatives="${JOB_N_NEG[$job_id]}"
    local normalize="${JOB_NORMALIZE[$job_id]}"
    local auto_k="${JOB_AUTO_K[$job_id]}"
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    local err_file="${JOB_ERR_FILE[$job_id]}"

    if [ "$verbose" -eq 1 ]; then
        echo "Launching siamese: model=$model fgsm=$fgsm n_calibration=$n_calibration loss=$loss dloss=$dloss prototype=$prototype n_pos=$n_positives n_neg=$n_negatives normalize=$normalize"
    fi

    PYTHONPATH="/home/simon/otitenet" /home/simon/otitenet/.conda/bin/python -m otitenet.train.train_triplet_new \
        --n_calibration="$n_calibration" --is_stn=0 --dloss="$dloss" --device=cuda:0 \
        --classif_loss="$loss" --n_epochs="$n_epochs" --task="$task" --weighted_sampler=0 \
        --groupkfold=1 --n_negatives="$n_negatives" --n_positives="$n_positives" --prototypes_to_use="$prototype" --fgsm="$fgsm" \
        --new_size="$new_size" --early_stop="$early_stop" --n_trials="$n_trials" --normalize="$normalize" --auto_select_k="$auto_k" \
        --bs="$bs" --log_comet="$triplet_log_comet" --log_mlflow="$triplet_log_mlflow" --log_tracking="$triplet_log_tracking" \
        --save_repro_artifacts="$triplet_save_repro_artifacts" \
        --exp_id="$exp_id" --run_tag="$test_tag" \
        > "$out_file" 2>> "$err_file"
}

run_cnn_mlp_job() {
    local job_id="$1"
    local model="${JOB_MODEL[$job_id]}"
    local variant="${JOB_VARIANT[$job_id]}"
    local normalize="${JOB_NORMALIZE[$job_id]}"
    local n_calibration="${JOB_N_CAL[$job_id]}"
    local fgsm="${JOB_FGSM[$job_id]}"
    local dloss="${JOB_DLOSS[$job_id]}"
    local classif_loss="${JOB_CLASSIF[$job_id]}"
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    local err_file="${JOB_ERR_FILE[$job_id]}"

    if [ "$verbose" -eq 1 ]; then
        echo "Launching $variant: model=$model normalize=$normalize n_calibration=$n_calibration fgsm=$fgsm dloss=$dloss classif_loss=$classif_loss"
    fi

    PYTHONPATH="/home/simon/otitenet" /home/simon/otitenet/.conda/bin/python -m otitenet.train.train_cnn_mlp_compare \
        --path=./data/otite_ds_64 --device=cuda:0 --task="$task" --groupkfold=1 --weighted_sampler=0 \
        --model_name="$model" --variant="$variant" --compare_all="$cnn_compare_all" --normalize="$normalize" \
        --new_size="$new_size" --n_epochs="$n_epochs" --early_stop="$early_stop" --bs="$bs" \
        --n_calibration="$n_calibration" --fgsm="$fgsm" --dloss="$dloss" --classif_loss="$classif_loss" \
        --gamma="$gamma" --epsilon="$epsilon" --run_tag="$test_tag" --exp_id="$exp_id" --verbose="$verbose" --log_mlflow=0 \
        > "$out_file" 2>> "$err_file"
}

run_job() {
    local job_id="$1"
    local kind="${JOB_KIND[$job_id]}"
    if [ "$kind" = "siamese" ]; then
        run_siamese_job "$job_id"
    else
        run_cnn_mlp_job "$job_id"
    fi
}

launch_job() {
    local job_id="$1"
    local retries="${JOB_RETRY_COUNT[$job_id]}"
    local kind="${JOB_KIND[$job_id]}"
    local model="${JOB_MODEL[$job_id]}"
    local free_mb
    free_mb="$(gpu_free_mb)"

    if [ "$verbose" -eq 0 ]; then
        if [ "$retries" -gt 0 ]; then
            echo "[retry=$retries] launching $kind job_id=$job_id model=$model free_mb=$free_mb"
        else
            echo "launching $kind job_id=$job_id model=$model free_mb=$free_mb"
        fi
    fi

    run_job "$job_id" &
    local pid=$!
    running_pids+=("$pid")
    PID_TO_JOB_ID[$pid]="$job_id"
}

finalize_job_result() {
    local job_id="$1"
    local exit_code="$2"
    local err_file="${JOB_ERR_FILE[$job_id]}"

    if [ "$exit_code" -eq 0 ]; then
        touch "logs/${JOB_EXP_ID[$job_id]}.done"
        finished_jobs=$((finished_jobs + 1))
        progress_tick "completed job_id=$job_id"
        return
    fi

    if is_oom_error "$err_file" && [ "${JOB_RETRY_COUNT[$job_id]}" -lt "$max_oom_retries" ]; then
        JOB_RETRY_COUNT[$job_id]=$((JOB_RETRY_COUNT[$job_id] + 1))
        enqueue_retry "$job_id"
        echo "OOM detected, re-queued job_id=$job_id (retry ${JOB_RETRY_COUNT[$job_id]}/$max_oom_retries)"
        return
    fi

    JOB_FAILED_FINAL[$job_id]=1
    record_failure "$job_id" "$exit_code"
    finished_jobs=$((finished_jobs + 1))
    progress_tick "failed job_id=$job_id exit=$exit_code"
}

poll_running_jobs() {
    local new_running=()
    local pid
    for pid in "${running_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_running+=("$pid")
            continue
        fi
        wait "$pid"
        local exit_code=$?
        local job_id="${PID_TO_JOB_ID[$pid]}"
        unset PID_TO_JOB_ID[$pid]
        finalize_job_result "$job_id" "$exit_code"
    done
    running_pids=("${new_running[@]}")
}

generate_jobs() {
    local n_calibration normalize model fgsm dloss loss prototype classif_loss n_negatives exp_id
    for n_calibration in 0 4; do
        for normalize in yes no; do
            for model in resnet18 vgg16 efficientnet_b0 vit; do
                for fgsm in 0 1; do
                    for dloss in no inverseTriplet; do
                        # 1) Siamese jobs first for this shared parameter set
                        for loss in arcface triplet; do
                            for prototype in batch no class; do
                                if ! [[ "$prototype" == "batch" && "$dloss" == "no" ]]; then
                                    exp_id="${test_tag}_${task}_siamese_${model}_${loss}_${dloss}_proto${prototype}_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                                    register_siamese_job "$model" "$fgsm" "$n_calibration" "$loss" "$dloss" "$prototype" 1 1 "$normalize" "$exp_id"
                                fi
                            done
                        done

                        for classif_loss in ce hinge; do
                            exp_id="${test_tag}_${task}_siamese_${model}_${classif_loss}_${dloss}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                            register_siamese_job "$model" "$fgsm" "$n_calibration" "$classif_loss" "$dloss" "no" 1 1 "$normalize" "$exp_id"
                        done

                        exp_id="${test_tag}_${task}_siamese_${model}_softmax_contrastive_${dloss}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                        register_siamese_job "$model" "$fgsm" "$n_calibration" "softmax_contrastive" "$dloss" "no" 1 1 "$normalize" "$exp_id"

                        for n_negatives in 1 5; do
                            exp_id="${test_tag}_${task}_siamese_${model}_triplet_${dloss}_nneg${n_negatives}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                            register_siamese_job "$model" "$fgsm" "$n_calibration" "triplet" "$dloss" "no" 1 "$n_negatives" "$normalize" "$exp_id"
                        done

                        # 2) CNN then MLP for the same shared parameters
                        if [ "$run_cnn_mlp" -eq 1 ]; then
                            for classif_loss in ce hinge; do
                                exp_id="${test_tag}_${task}_cnn_${model}_${classif_loss}_${dloss}_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                                register_cnn_mlp_job "$model" "cnn_transfer" "$normalize" "$n_calibration" "$fgsm" "$dloss" "$classif_loss" "$exp_id"

                                exp_id="${test_tag}_${task}_mlp_${model}_${classif_loss}_${dloss}_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                                register_cnn_mlp_job "$model" "mlp" "$normalize" "$n_calibration" "$fgsm" "$dloss" "$classif_loss" "$exp_id"
                            done
                        fi
                    done
                done
            done
        done
    done
    total_models=$((next_job_id - 1))
}

count_already_done() {
    local done_count=0
    local exp_id
    for exp_id in "${JOB_EXP_ID[@]}"; do
        if [ -f "logs/${exp_id}.done" ]; then
            done_count=$((done_count + 1))
        fi
    done
    echo "$done_count"
}

scheduler_loop() {
    local free_mb slot_cap running_count loop_counter
    loop_counter=0

    while :; do
        loop_counter=$((loop_counter + 1))
        poll_running_jobs

        running_count=${#running_pids[@]}
        free_mb="$(gpu_free_mb)"
        slot_cap="$(dynamic_slot_cap "$free_mb")"

        while [ "$running_count" -lt "$num_jobs" ] && [ "$running_count" -lt "$slot_cap" ]; do
            if ! dequeue_next_job; then
                break
            fi
            launch_job "$dequeued_job_id"
            running_count=${#running_pids[@]}
            free_mb="$(gpu_free_mb)"
            slot_cap="$(dynamic_slot_cap "$free_mb")"
        done

        if [ "$verbose" -eq 0 ] && [ $((loop_counter % 12)) -eq 0 ]; then
            echo "Scheduler heartbeat: done=$finished_jobs/$total_models running=${#running_pids[@]} pending=${#pending_queue[@]} retry=${#retry_queue[@]} free_mb=${free_mb} slot_cap=${slot_cap}/${num_jobs}"
        fi

        if [ ${#pending_queue[@]} -eq 0 ] && [ ${#retry_queue[@]} -eq 0 ] && [ ${#running_pids[@]} -eq 0 ]; then
            break
        fi

        if [ "$slot_cap" -lt "$num_jobs" ]; then
            if [ "$memory_pressure_active" -eq 0 ] && [ "$verbose" -eq 0 ]; then
                echo "MAX GPU MEMORY REACHED: launch capacity reduced to $slot_cap/$num_jobs (free=${free_mb}MB, min_free=${min_free_mb}MB, est_per_job=${mem_per_job_mb}MB)"
            fi
            memory_pressure_active=1
        else
            if [ "$memory_pressure_active" -eq 1 ] && [ "$verbose" -eq 0 ]; then
                echo "GPU memory pressure cleared: launch capacity restored to $slot_cap/$num_jobs (free=${free_mb}MB)"
            fi
            memory_pressure_active=0
        fi

        if [ "$slot_cap" -eq 0 ] && [ "$verbose" -eq 0 ]; then
            echo "Pausing launches: GPU free memory ${free_mb}MB < min ${min_free_mb}MB"
        fi

        sleep "$poll_interval"
    done
}

count_jobs() {
    local count=0
    for n_calibration in 0 4; do
        for normalize in yes no; do
            for model in resnet18 vgg16 efficientnet_b0 vit; do
                for fgsm in 0 1; do
                    for dloss in no inverseTriplet; do
                        # Siamese jobs (arcface/triplet + prototypes, siamese-only)
                        for loss in arcface triplet; do
                            for prototype in batch no class; do
                                if ! [[ "$prototype" == "batch" && "$dloss" == "no" ]]; then
                                    count=$((count + 1))
                                fi
                            done
                        done

                        # Siamese-only additional losses
                        for classif_loss in ce hinge; do
                            count=$((count + 1))
                        done
                        count=$((count + 1)) # softmax_contrastive

                        # Siamese triplet n_neg sweep
                        for n_negatives in 1 5; do
                            count=$((count + 1))
                        done

                        if [ "$run_cnn_mlp" -eq 1 ]; then
                            # CNN then MLP for shared params.
                            for classif_loss in ce hinge; do
                                count=$((count + 1)) # cnn_transfer
                                count=$((count + 1)) # mlp
                            done
                        fi
                    done
                done
            done
        done
    done
    total_models=$count
}

generate_jobs

already_done=$(count_already_done)
finished_jobs=$already_done
echo "Scheduler config: num_jobs=$num_jobs min_free_mb=$min_free_mb mem_per_job_mb=$mem_per_job_mb max_oom_retries=$max_oom_retries"
echo "Planned jobs: $total_models (already completed: $already_done, to be run: $((total_models - already_done)))"

scheduler_loop

failed_count=0
for job_id in $(seq 1 "$total_models"); do
    if [ "${JOB_FAILED_FINAL[$job_id]}" -eq 1 ]; then
        failed_count=$((failed_count + 1))
    fi
done

{
    echo "Run completed at: $(date)"
    echo "Total jobs: $total_models"
    echo "Failed jobs: $failed_count"
    echo "Primary log file: $log_file"
    echo "-----------------------------------------------------"
} >> "$log_file"

echo "Run mode: $( [ "$test_mode" -eq 1 ] && echo TEST || echo FULL )"
echo "Primary run log: $log_file"
echo "Per-job error logs: logs/*_error.log"
echo "All jobs completed. Failed jobs: $failed_count/$total_models"
