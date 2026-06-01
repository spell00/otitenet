#!/usr/bin/env bash

n_epochs=1000
task=otitis_four_class
num_jobs=1
new_size=224
early_stop=10
n_trials=3
log_file=""
auto_select_k=0
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
num_workers=8
dataset_name="otite_ds_64"
valid_dataset="Banque_Viscaino_Chili_2020"
test_dataset="inference"
train_datasets="Banque_Comert_Turquie_2020_jpg,Banque_Calaman_USA_2020_trie_CM,GMFUNL_jan2023"
user_set_num_jobs=0
user_set_bs=0
min_free_mb=2048
mem_per_job_mb=6000
max_oom_retries=3
poll_interval=5
show_progress_bar=1
siamese_inference="linearsvc" # knn, linearsvc, logisticregression

# DANN/FGSM params shared with CNN/MLP.
gamma=1.0
epsilon=0.01

force_run=0
# CLI flags:
#   --test  : run smoke mode (1 epoch/1 trial/single job)
registered_job_ids=()
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
        --num-workers=*)
            num_workers="${arg#*=}"
            ;;
        --dataset=*)
            dataset_name="${arg#*=}"
            ;;
        --valid-dataset=*)
            valid_dataset="${arg#*=}"
            ;;
        --test-dataset=*)
            test_dataset="${arg#*=}"
            ;;
        --train-datasets=*)
            train_datasets="${arg#*=}"
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
        --no-progress-bar)
            show_progress_bar=0
            ;;
        --task=*)
            task="${arg#*=}"
            ;;
        --progress-bar)
            show_progress_bar=1
            ;;
    esac
done

if [ "$siamese_inference" != "knn" ]; then
    auto_select_k=0
fi

infer_dataset_name_from_split_config() {
    local current_dataset_name="$1"
    local train_value="$2"
    local valid_value="$3"
    local test_value="$4"
    python - "$current_dataset_name" "$train_value" "$valid_value" "$test_value" <<'PY'
import sys
from otitenet.data.dataset_paths import infer_output_subdir_from_split_datasets

dataset_name, train_datasets, valid_dataset, test_dataset = sys.argv[1:5]
if "/" in dataset_name.strip("/"):
    print(dataset_name)
    raise SystemExit(0)

subdir = infer_output_subdir_from_split_datasets(
    train_datasets=train_datasets,
    valid_dataset=valid_dataset,
    test_dataset=test_dataset,
)
print(f"{dataset_name.rstrip('/')}/{subdir}" if subdir else dataset_name)
PY
}

resolve_dataset_path() {
    local requested_path="$1"
    python - "$requested_path" <<'PY'
import sys
from otitenet.data.dataset_paths import resolve_processed_dataset_path

print(resolve_processed_dataset_path(sys.argv[1]))
PY
}

dataset_name="$(PYTHONPATH="/home/simon/otitenet" infer_dataset_name_from_split_config "$dataset_name" "$train_datasets" "$valid_dataset" "$test_dataset")"
dataset_path="./data/${dataset_name}"
resolved_dataset_path="$(PYTHONPATH="/home/simon/otitenet" resolve_dataset_path "$dataset_path")"
if [ "$resolved_dataset_path" != "$dataset_path" ]; then
    dataset_path="$resolved_dataset_path"
    dataset_name="${dataset_path#./data/}"
    dataset_name="${dataset_name#data/}"
fi

validate_requested_split_against_infos() {
    local infos_path="${dataset_path}/infos.csv"
    if [ ! -f "$infos_path" ] || [ -z "$train_datasets" ]; then
        return
    fi

    python - "$infos_path" "$train_datasets" "$valid_dataset" "$test_dataset" <<'PY'
import csv
import sys

infos_path, requested_train, requested_valid, requested_test = sys.argv[1:5]
requested_train_set = {x.strip() for x in requested_train.split(",") if x.strip()}
train_set = set()
valid_set = set()
test_set = set()

try:
    with open(infos_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "group" not in (reader.fieldnames or []) or "dataset" not in (reader.fieldnames or []):
            raise SystemExit(0)
        for row in reader:
            group = str(row.get("group", "")).strip().lower()
            dataset = str(row.get("dataset", "")).strip()
            if not dataset:
                continue
            if group == "train":
                train_set.add(dataset)
            elif group == "valid":
                valid_set.add(dataset)
            elif group == "test":
                test_set.add(dataset)
except Exception as exc:
    print(f"[SplitWarning] Could not inspect {infos_path}: {exc}")
    raise SystemExit(0)

if not (train_set or valid_set or test_set):
    raise SystemExit(0)

missing_train = sorted(requested_train_set - train_set)
if missing_train:
    print(
        "[SplitWarning] infos.csv defines explicit split groups, so it overrides "
        "--train_datasets. Requested train dataset(s) missing from group=train: "
        + ",".join(missing_train)
    )
    print("[SplitWarning] Effective train_datasets from infos.csv: " + ",".join(sorted(train_set)))

if requested_valid and requested_valid not in valid_set:
    print(
        f"[SplitWarning] Requested valid_dataset={requested_valid} is not present in infos.csv group=valid. "
        f"Effective valid datasets: {','.join(sorted(valid_set))}"
    )

if requested_test and requested_test not in test_set:
    print(
        f"[SplitWarning] Requested test_dataset={requested_test} is not present in infos.csv group=test. "
        f"Effective test datasets: {','.join(sorted(test_set))}"
    )
PY
}

validate_requested_split_against_infos

# Use a filesystem-safe key for logs/markers/experiment ids.
# Keep dataset_name unchanged for the actual data path and CLI metadata.
dataset_key="$(printf '%s' "$dataset_name" | tr -c '[:alnum:]_.-' '_')"

progress_root="logs/progresses/${task}/${dataset_key}"

# Per-state folders — each has the same 6 subdirs: cache/ done/ error/ log/ warning/ db/
tmp_root="${progress_root}/tmp"
tmp_cache_dir="${tmp_root}/cache"
tmp_done_dir="${tmp_root}/done"
tmp_error_dir="${tmp_root}/error"
tmp_log_dir="${tmp_root}/log"
tmp_warning_dir="${tmp_root}/warning"
tmp_db_dir="${tmp_root}/db"

success_root="${progress_root}/success"
success_cache_dir="${success_root}/cache"
success_done_dir="${success_root}/done"
success_error_dir="${success_root}/error"
success_log_dir="${success_root}/log"
success_warning_dir="${success_root}/warning"
success_db_dir="${success_root}/db"

failed_root="${progress_root}/failed"
failed_cache_dir="${failed_root}/cache"
failed_done_dir="${failed_root}/done"
failed_error_dir="${failed_root}/error"
failed_log_dir="${failed_root}/log"
failed_warning_dir="${failed_root}/warning"
failed_db_dir="${failed_root}/db"

# All CSV / manifest files live in csv/
csv_dir="${progress_root}/csv"
runtime_log_file="${csv_dir}/${test_tag}_${task}_job_runtime.csv"
manifest_file="${csv_dir}/${test_tag}_${task}_job_manifest.csv"
status_file="${csv_dir}/${test_tag}_${task}_job_status.csv"
settings_file="${csv_dir}/${test_tag}_${task}_run_settings.csv"
completed_runs_file="${csv_dir}/${test_tag}_${task}_completed_runs.csv"
cumulative_error_log="${csv_dir}/error_cumulative.log"

# Launcher log lives in tmp/error for the active run
log_file="${tmp_error_dir}/${test_tag}_${task}_launcher.log"

# Clear tmp dirs at each launch (keep success/failed/csv across runs)
rm -rf "$tmp_root"
mkdir -p \
    "$tmp_cache_dir" "$tmp_done_dir" "$tmp_error_dir" "$tmp_log_dir" "$tmp_warning_dir" "$tmp_db_dir" \
    "$success_cache_dir" "$success_done_dir" "$success_error_dir" "$success_log_dir" "$success_warning_dir" "$success_db_dir" \
    "$failed_cache_dir" "$failed_done_dir" "$failed_error_dir" "$failed_log_dir" "$failed_warning_dir" "$failed_db_dir" \
    "$csv_dir" \
    logs
echo "Error Log - $(date)" > "$log_file"

done_marker_path() {
    local exp_id="$1"
    echo "${tmp_done_dir}/${exp_id}.done"
}

done_marker_exists() {
    local exp_id="$1"
    local marker
    marker="$(done_marker_path "$exp_id")"
    # Check tmp, success, and legacy locations
    [ -f "$marker" ] || [ -f "${success_done_dir}/${exp_id}.done" ] || [ -f "logs/${task}/${dataset_key}/${exp_id}.done" ] || [ -f "logs/${task}/${exp_id}.done" ] || [ -f "logs/${exp_id}.done" ]
}

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

# If data loaders use additional workers, run one model at a time to avoid CPU oversubscription.
if [ "$num_workers" != "0" ]; then
    num_jobs=1
fi

crush_old_models_for_dataset() {
    local dataset="$1"
    local base_dir="logs/best_models/${task}"
    local removed_count=0

    if [ ! -d "$base_dir" ]; then
        echo "[force] no previous model directory found at $base_dir"
        return
    fi

    while IFS= read -r -d '' ds_dir; do
        rm -rf "$ds_dir"
        removed_count=$((removed_count + 1))
    done < <(find "$base_dir" -type d -name "$dataset" -print0)

    echo "[force] crushed $removed_count model dataset folder(s) for $dataset under $base_dir"
}

clear_done_markers() {
    local removed_count=0
    local file

    shopt -s nullglob
    for file in "${tmp_done_dir}/${test_tag}_${task}_${dataset_key}_"*.done "${success_done_dir}/${test_tag}_${task}_${dataset_key}_"*.done; do
        rm -f "$file"
        removed_count=$((removed_count + 1))
    done
    shopt -u nullglob

    echo "[force] removed $removed_count done marker(s) matching ${test_tag}_${task}_${dataset_key}_*.done"
}

if [ "$force_run" -eq 1 ]; then
    crush_old_models_for_dataset "$dataset_name"
    clear_done_markers
fi

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
declare -A JOB_UUID
declare -A JOB_RETRY_COUNT
declare -A JOB_BS
declare -A JOB_ERR_FILE
declare -A JOB_STDOUT_FILE
declare -A JOB_FAILED_FINAL
declare -A JOB_THEORETICAL_GPU_MB
declare -A JOB_RESERVED_GPU_MB
declare -A JOB_ACTUAL_PEAK_GPU_MB
declare -A JOB_OOM_MISSING_GPU_MB
declare -A JOB_OOM_GPU_FREE_AT_FAILURE_MB
declare -A JOB_OOM_GPU_USED_AT_FAILURE_MB
declare -A JOB_RESERVATION_SOURCE_TAG
declare -A JOB_RESERVATION_MATCH_QUALITY
declare -A JOB_TELEMETRY_STATUS
declare -A JOB_BLOCKED_PRIOR_STATE

declare -A PREV_JOB_STATE_BY_EXP
declare -A SMOKE_PEAK_MB_BY_KEY
declare -A SMOKE_THEORETICAL_MB_BY_KEY
declare -A SMOKE_FALLBACK_PEAK_MB_BY_KEY
declare -A SMOKE_FALLBACK_THEORETICAL_MB_BY_KEY
declare -A SMOKE_FALLBACK_STATE_BY_KEY

declare -A PID_TO_JOB_ID
declare -A EXP_UUID_CACHE

uuid_cache_ready=0

pending_queue=()
retry_queue=()
running_pids=()
next_job_id=1
memory_pressure_active=0
dequeued_job_id=""
dequeued_from_queue=""

strip_outer_quotes() {
    local value="$1"
    value="${value#\"}"
    value="${value%\"}"
    value="${value//\"\"/\"}"
    printf '%s' "$value"
}

sanitize_key_part() {
    local value="$1"
    value="$(strip_outer_quotes "$value")"
    value="${value//|/_}"
    printf '%s' "$value"
}

build_siamese_match_key() {
    local model="$1"
    local loss="$2"
    local dloss="$3"
    local prototype="$4"
    local fgsm="$5"
    local n_cal="$6"
    local n_pos="$7"
    local n_neg="$8"
    local normalize="$9"
    printf 'siamese|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' \
        "$(sanitize_key_part "$model")" \
        "$(sanitize_key_part "$loss")" \
        "$(sanitize_key_part "$dloss")" \
        "$(sanitize_key_part "$prototype")" \
        "$(sanitize_key_part "$fgsm")" \
        "$(sanitize_key_part "$n_cal")" \
        "$(sanitize_key_part "$n_pos")" \
        "$(sanitize_key_part "$n_neg")" \
        "$(sanitize_key_part "$normalize")" \
        "$(sanitize_key_part "$dataset_name")" \
        "$(sanitize_key_part "$task")"
}

build_cnn_mlp_match_key() {
    local model="$1"
    local variant="$2"
    local classif_loss="$3"
    local dloss="$4"
    local fgsm="$5"
    local n_cal="$6"
    local normalize="$7"
    printf 'cnn_mlp|%s|%s|%s|%s|%s|%s|%s|%s|%s' \
        "$(sanitize_key_part "$model")" \
        "$(sanitize_key_part "$variant")" \
        "$(sanitize_key_part "$classif_loss")" \
        "$(sanitize_key_part "$dloss")" \
        "$(sanitize_key_part "$fgsm")" \
        "$(sanitize_key_part "$n_cal")" \
        "$(sanitize_key_part "$normalize")" \
        "$(sanitize_key_part "$dataset_name")" \
        "$(sanitize_key_part "$task")"
}

smoke_manifest_path() {
    if [ "$test_tag" = "TEST_SMOKE" ]; then
        printf '%s' "$manifest_file"
    else
        printf '%s' "${progress_root}/csv/TEST_SMOKE_${task}_job_manifest.csv"
    fi
}

load_previous_manifest_state_cache() {
    local path="$manifest_file"
    local exp_id state

    if [ ! -f "$path" ]; then
        return
    fi

    while IFS=$'\t' read -r exp_id state; do
        if [ -n "$exp_id" ]; then
            PREV_JOB_STATE_BY_EXP["$exp_id"]="$state"
        fi
    done < <(
        awk -F, '
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    gsub(/^"|"$/, "", $i)
                    if ($i == "exp_id") exp_col = i
                    if ($i == "job_state") state_col = i
                }
                next
            }
            exp_col > 0 && state_col > 0 {
                expid = $exp_col
                state = $state_col
                gsub(/^"|"$/, "", expid)
                gsub(/^"|"$/, "", state)
                gsub(/""/, "\"", expid)
                gsub(/""/, "\"", state)
                print expid "\t" state
            }
        ' "$path"
    )
}

load_smoke_reservation_cache() {
    local path
    path="$(smoke_manifest_path)"

    if [ ! -f "$path" ]; then
        return
    fi

    while IFS=$'\t' read -r match_key state peak theoretical; do
        if [ -z "$match_key" ]; then
            continue
        fi
        if [ "$state" = "completed" ]; then
            if [ -z "${SMOKE_PEAK_MB_BY_KEY[$match_key]:-}" ] && [ -n "$peak" ]; then
                SMOKE_PEAK_MB_BY_KEY["$match_key"]="$peak"
            fi
            if [ -z "${SMOKE_THEORETICAL_MB_BY_KEY[$match_key]:-}" ] && [ -n "$theoretical" ]; then
                SMOKE_THEORETICAL_MB_BY_KEY["$match_key"]="$theoretical"
            fi
        fi
        if [ -z "${SMOKE_FALLBACK_PEAK_MB_BY_KEY[$match_key]:-}" ] && [ -n "$peak" ]; then
            SMOKE_FALLBACK_PEAK_MB_BY_KEY["$match_key"]="$peak"
            SMOKE_FALLBACK_STATE_BY_KEY["$match_key"]="$state"
        fi
        if [ -z "${SMOKE_FALLBACK_THEORETICAL_MB_BY_KEY[$match_key]:-}" ] && [ -n "$theoretical" ]; then
            SMOKE_FALLBACK_THEORETICAL_MB_BY_KEY["$match_key"]="$theoretical"
            if [ -z "${SMOKE_FALLBACK_STATE_BY_KEY[$match_key]:-}" ]; then
                SMOKE_FALLBACK_STATE_BY_KEY["$match_key"]="$state"
            fi
        fi
    done < <(
        awk -F, -v dataset_name="$dataset_name" -v task="$task" '
            function unquote(v) {
                gsub(/^"|"$/, "", v)
                gsub(/""/, "\"", v)
                return v
            }
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    h = unquote($i)
                    if (h == "kind") c_kind = i
                    else if (h == "model") c_model = i
                    else if (h == "variant") c_variant = i
                    else if (h == "classif_loss") c_classif = i
                    else if (h == "loss") c_loss = i
                    else if (h == "dloss") c_dloss = i
                    else if (h == "prototype") c_proto = i
                    else if (h == "fgsm") c_fgsm = i
                    else if (h == "n_calibration") c_ncal = i
                    else if (h == "n_positives") c_npos = i
                    else if (h == "n_negatives") c_nneg = i
                    else if (h == "normalize") c_norm = i
                    else if (h == "dataset_name") c_dataset = i
                    else if (h == "task") c_task = i
                    else if (h == "job_state") c_state = i
                    else if (h == "actual_peak_gpu_mb") c_peak = i
                    else if (h == "theoretical_gpu_required_mb") c_theo = i
                }
                next
            }
            {
                kind = c_kind ? unquote($c_kind) : ""
                if (kind == "") next
                ds = c_dataset ? unquote($c_dataset) : ""
                t = c_task ? unquote($c_task) : ""
                if (dataset_name != "" && ds != dataset_name) next
                if (task != "" && t != task) next

                state = c_state ? unquote($c_state) : ""
                peak = c_peak ? unquote($c_peak) : ""
                theo = c_theo ? unquote($c_theo) : ""

                if (kind == "cnn_mlp") {
                    key = "cnn_mlp|" (c_model ? unquote($c_model) : "") "|" (c_variant ? unquote($c_variant) : "") "|" (c_classif ? unquote($c_classif) : "") "|" (c_dloss ? unquote($c_dloss) : "") "|" (c_fgsm ? unquote($c_fgsm) : "") "|" (c_ncal ? unquote($c_ncal) : "") "|" (c_norm ? unquote($c_norm) : "") "|" ds "|" t
                } else {
                    key = "siamese|" (c_model ? unquote($c_model) : "") "|" (c_loss ? unquote($c_loss) : "") "|" (c_dloss ? unquote($c_dloss) : "") "|" (c_proto ? unquote($c_proto) : "") "|" (c_fgsm ? unquote($c_fgsm) : "") "|" (c_ncal ? unquote($c_ncal) : "") "|" (c_npos ? unquote($c_npos) : "") "|" (c_nneg ? unquote($c_nneg) : "") "|" (c_norm ? unquote($c_norm) : "") "|" ds "|" t
                }
                print key "\t" state "\t" peak "\t" theo
            }
        ' "$path"
    )
}

strict_state_gate_blocks_job() {
    local exp_id="$1"
    local state="${PREV_JOB_STATE_BY_EXP[$exp_id]:-}"
    if [ -z "$state" ]; then
        return 1
    fi
    if [ "$state" = "done" ] || [ "$state" = "completed" ]; then
        return 0
    fi
    return 1
}

apply_reservation_for_job() {
    local job_id="$1"
    # Flat 4000MB reservation for every job regardless of smoke data.
    # actual_peak_gpu_mb is still measured and written to the manifest for reference.
    JOB_RESERVED_GPU_MB[$job_id]=4000
    JOB_RESERVATION_SOURCE_TAG[$job_id]="fixed_4000mb_flat"
    JOB_RESERVATION_MATCH_QUALITY[$job_id]="fixed"
    JOB_TELEMETRY_STATUS[$job_id]="pending"
}

required_gpu_mb_for_job() {
    local job_id="$1"
    local reserved="${JOB_RESERVED_GPU_MB[$job_id]:-}"
    if [[ -n "$reserved" && "$reserved" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s' "${reserved%.*}"
    else
        printf '%s' "$mem_per_job_mb"
    fi
}

extract_telemetry_field() {
    local line="$1"
    local key="$2"
    local segment
    IFS='|' read -r -a __segments <<< "$line"
    for segment in "${__segments[@]}"; do
        if [[ "$segment" == "$key="* ]]; then
            printf '%s' "${segment#*=}"
            return
        fi
    done
}

hydrate_job_gpu_telemetry() {
    local job_id="$1"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    local err_file="${JOB_ERR_FILE[$job_id]}"
    local last_line oom_line

    last_line="$(grep -h 'OTITENET_GPU_TELEMETRY|' "$out_file" "$err_file" 2>/dev/null | tail -n 1)"
    oom_line="$(grep -h 'OTITENET_GPU_TELEMETRY|event=oom|' "$out_file" "$err_file" 2>/dev/null | tail -n 1)"

    if [ -z "$last_line" ] && [ -z "$oom_line" ]; then
        if [ -z "${JOB_TELEMETRY_STATUS[$job_id]:-}" ] || [ "${JOB_TELEMETRY_STATUS[$job_id]}" = "pending" ]; then
            JOB_TELEMETRY_STATUS[$job_id]="unavailable"
        fi
        return
    fi

    if [ -n "$last_line" ]; then
        local theoretical peak
        theoretical="$(extract_telemetry_field "$last_line" "theoretical_gpu_required_mb")"
        peak="$(extract_telemetry_field "$last_line" "actual_peak_gpu_mb")"
        if [ -n "$theoretical" ]; then
            JOB_THEORETICAL_GPU_MB[$job_id]="$theoretical"
        fi
        if [ -n "$peak" ]; then
            JOB_ACTUAL_PEAK_GPU_MB[$job_id]="$peak"
        fi
    fi

    if [ -n "$oom_line" ]; then
        local missing free_mb used_mb
        missing="$(extract_telemetry_field "$oom_line" "oom_missing_gpu_mb")"
        free_mb="$(extract_telemetry_field "$oom_line" "oom_gpu_free_at_failure_mb")"
        used_mb="$(extract_telemetry_field "$oom_line" "oom_gpu_used_at_failure_mb")"
        if [ -n "$missing" ]; then
            JOB_OOM_MISSING_GPU_MB[$job_id]="$missing"
        fi
        if [ -n "$free_mb" ]; then
            JOB_OOM_GPU_FREE_AT_FAILURE_MB[$job_id]="$free_mb"
        fi
        if [ -n "$used_mb" ]; then
            JOB_OOM_GPU_USED_AT_FAILURE_MB[$job_id]="$used_mb"
        fi
        JOB_TELEMETRY_STATUS[$job_id]="oom"
        return
    fi

    JOB_TELEMETRY_STATUS[$job_id]="ok"
}

progress_tick() {
    echo "[$finished_jobs/$total_models] $1"
}

render_progress_bar() {
    if [ "$show_progress_bar" -ne 1 ]; then
        return
    fi

    local total="$total_models"
    local done="$finished_jobs"
    local running="${#running_pids[@]}"
    local pending="${#pending_queue[@]}"
    local retry="${#retry_queue[@]}"
    local width=30
    local percent=0
    local filled=0
    local empty=0
    local bar_filled
    local bar_empty

    if [ "$total" -gt 0 ]; then
        percent=$(( done * 100 / total ))
        filled=$(( done * width / total ))
    fi
    empty=$(( width - filled ))

    bar_filled=$(printf '%*s' "$filled" '' | tr ' ' '#')
    bar_empty=$(printf '%*s' "$empty" '')

    # In-place single-line status (no scrolling spam)
    printf '\r[%s%s] %3d%% (%d/%d) running=%d pending=%d retry=%d' \
        "$bar_filled" "$bar_empty" "$percent" "$done" "$total" "$running" "$pending" "$retry"
}

print_job_logs() {
    local job_id="$1"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    local err_file="${JOB_ERR_FILE[$job_id]}"

    if [ -n "$out_file" ] || [ -n "$err_file" ]; then
        echo "job_id $job_id logs: stdout ${out_file:-<none>} stderr ${err_file:-<none>}"
    fi
    # Append error log to cumulative error log if it exists and is non-empty
    if [ -n "$err_file" ] && [ -s "$err_file" ]; then
        echo "==== job_id $job_id error log at $(date) ====" >> "$cumulative_error_log"
        cat "$err_file" >> "$cumulative_error_log"
        echo >> "$cumulative_error_log"
    fi
}

cleanup_job_processes() {
    local job_id="$1"
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local pids

    if [ -z "$exp_id" ]; then
        return
    fi

    # Kill any lingering training processes for this specific experiment id.
    # This prevents orphaned GPU workers from surviving after OOM/failure.
    pids=$(pgrep -f "otitenet\.train\.(train_triplet_new|train_cnn_mlp_compare).+--exp_id=${exp_id}" || true)
    if [ -z "$pids" ]; then
        return
    fi

    echo "Cleaning up lingering process(es) for job_id=$job_id exp_id=$exp_id: $pids"
    kill $pids 2>/dev/null || true
    sleep 1

    pids=$(pgrep -f "otitenet\.train\.(train_triplet_new|train_cnn_mlp_compare).+--exp_id=${exp_id}" || true)
    if [ -n "$pids" ]; then
        echo "Force-killing lingering process(es) for job_id=$job_id exp_id=$exp_id: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
}

csv_escape() {
    local value="$1"
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

build_uuid_cache() {
    local force_refresh="${1:-0}"

    if [ "$force_refresh" -eq 0 ] && [ "$uuid_cache_ready" -eq 1 ]; then
        return
    fi

    local cached_key
    for cached_key in "${!EXP_UUID_CACHE[@]}"; do
        unset "EXP_UUID_CACHE[$cached_key]"
    done

    while IFS=$'\t' read -r exp_key uuid_value; do
        if [ -n "$exp_key" ] && [ -n "$uuid_value" ] && [ -z "${EXP_UUID_CACHE[$exp_key]:-}" ]; then
            EXP_UUID_CACHE[$exp_key]="$uuid_value"
        fi
    done < <(
        python - "$task" "$dataset_key" <<'PY'
import json
import os
import sys

task = sys.argv[1]
dataset_key = sys.argv[2]
roots = [
    os.path.join('logs', task, dataset_key),
    os.path.join('logs', task),
]
seen = set()

for root in roots:
    if not os.path.isdir(root):
        continue
    for dirpath, _, filenames in os.walk(root):
        if 'run_metadata.json' not in filenames:
            continue
        metadata_path = os.path.join(dirpath, 'run_metadata.json')
        try:
            with open(metadata_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            continue

        args = payload.get('args', {}) if isinstance(payload, dict) else {}
        exp_id = str(args.get('exp_id', '')).strip()
        run_tag = str(args.get('run_tag', '')).strip()
        foldername = str(payload.get('foldername', '') or payload.get('run_id', '')).strip()
        run_root = str(payload.get('complete_log_path', '') or payload.get('run_root', '')).strip()

        if not foldername and run_root:
            foldername = os.path.basename(run_root.rstrip('/'))
        if not exp_id or not foldername:
            continue

        aliases = {exp_id}
        suffix = f"_{task}_{run_tag}" if run_tag else f"_{task}"
        if suffix and exp_id.endswith(suffix):
            aliases.add(exp_id[:-len(suffix)])

        for alias in aliases:
            key = (alias, foldername)
            if key in seen:
                continue
            seen.add(key)
            print(f"{alias}\t{foldername}")
PY
    )

    uuid_cache_ready=1
}

hydrate_job_uuid() {
    local job_id="$1"
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local key

    if [ -n "${JOB_UUID[$job_id]:-}" ] || [ -z "$exp_id" ]; then
        return
    fi

    build_uuid_cache

    if [ -n "${EXP_UUID_CACHE[$exp_id]:-}" ]; then
        JOB_UUID[$job_id]="${EXP_UUID_CACHE[$exp_id]}"
        return
    fi

    for key in "${!EXP_UUID_CACHE[@]}"; do
        if [[ "$key" == "$exp_id"_* || "$exp_id" == "$key"_* ]]; then
            JOB_UUID[$job_id]="${EXP_UUID_CACHE[$key]}"
            return
        fi
    done

    # Retry once with a forced cache refresh in case metadata appeared after
    # the initial cache build (common while jobs are still running).
    build_uuid_cache 1

    if [ -n "${EXP_UUID_CACHE[$exp_id]:-}" ]; then
        JOB_UUID[$job_id]="${EXP_UUID_CACHE[$exp_id]}"
        return
    fi

    for key in "${!EXP_UUID_CACHE[@]}"; do
        if [[ "$key" == "$exp_id"_* || "$exp_id" == "$key"_* ]]; then
            JOB_UUID[$job_id]="${EXP_UUID_CACHE[$key]}"
            return
        fi
    done
}

resolve_job_run_dir() {
    local job_id="$1"
    local uuid="${JOB_UUID[$job_id]:-}"

    if [ -z "$uuid" ]; then
        hydrate_job_uuid "$job_id"
        uuid="${JOB_UUID[$job_id]:-}"
    fi

    if [ -z "$uuid" ]; then
        return 1
    fi

    local run_dir
    for run_dir in "logs/${task}/${dataset_key}/${uuid}" "logs/${task}/${uuid}"; do
        if [ -d "$run_dir" ]; then
            printf '%s' "$run_dir"
            return 0
        fi
    done

    return 1
}

validate_job_artifacts() {
    local job_id="$1"
    local run_dir
    run_dir="$(resolve_job_run_dir "$job_id")" || {
        echo "missing_run_dir"
        return 1
    }

    local model_path="${run_dir}/model.pth"
    local metadata_path="${run_dir}/run_metadata.json"
    local require_proto=0
    local prototype="${JOB_PROTOTYPE[$job_id]:-no}"

    if [ "${JOB_KIND[$job_id]}" = "siamese" ] && [ "$prototype" != "no" ] && [ -n "$prototype" ]; then
        require_proto=1
    fi

    if [ ! -f "$model_path" ]; then
        echo "missing_model_pth:${model_path}"
        return 1
    fi
    if [ ! -f "$metadata_path" ]; then
        echo "missing_run_metadata:${metadata_path}"
        return 1
    fi

    if [ "$require_proto" -eq 1 ]; then
        local proto_path="${run_dir}/prototypes.pkl"
        if [ ! -f "$proto_path" ]; then
            echo "missing_prototypes_pkl:${proto_path}"
            return 1
        fi
    fi

    echo "ok"
    return 0
}

array_contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

job_is_running() {
    local job_id="$1"
    local pid
    for pid in "${running_pids[@]}"; do
        if [ "${PID_TO_JOB_ID[$pid]:-}" = "$job_id" ]; then
            return 0
        fi
    done
    return 1
}

job_is_in_retry_queue() {
    local job_id="$1"
    array_contains "$job_id" "${retry_queue[@]}"
}

job_is_in_pending_queue() {
    local job_id="$1"
    array_contains "$job_id" "${pending_queue[@]}"
}

append_runtime_event() {
    local event="$1"
    local job_id="$2"
    local pid="$3"
    local exit_code="${4:-}"
    local retries="${JOB_RETRY_COUNT[$job_id]:-0}"
    local runtime_log="$runtime_log_file"

    if [ ! -f "$runtime_log" ]; then
        printf 'timestamp,event,job_id,pid,exp_id,kind,model,variant,retry_count,exit_code\n' > "$runtime_log"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "$(date '+%Y-%m-%d %H:%M:%S')")" \
        "$(csv_escape "$event")" \
        "$(csv_escape "$job_id")" \
        "$(csv_escape "$pid")" \
        "$(csv_escape "${JOB_EXP_ID[$job_id]}")" \
        "$(csv_escape "${JOB_KIND[$job_id]}")" \
        "$(csv_escape "${JOB_MODEL[$job_id]}")" \
        "$(csv_escape "${JOB_VARIANT[$job_id]}")" \
        "$(csv_escape "$retries")" \
        "$(csv_escape "$exit_code")" \
        >> "$runtime_log"
}

append_completed_run_event() {
    local job_id="$1"
    local status="$2"
    local exit_code="$3"
    local completed_log="$completed_runs_file"

    hydrate_job_uuid "$job_id"

    if [ ! -f "$completed_log" ]; then
        printf 'timestamp,job_id,status,exit_code,exp_id,uuid,kind,model,variant,fgsm,n_calibration,loss,dloss,prototype,n_positives,n_negatives,normalize,retry_count,error_file,stdout_file\n' > "$completed_log"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "$(date '+%Y-%m-%d %H:%M:%S')")" \
        "$(csv_escape "$job_id")" \
        "$(csv_escape "$status")" \
        "$(csv_escape "$exit_code")" \
        "$(csv_escape "${JOB_EXP_ID[$job_id]}")" \
        "$(csv_escape "${JOB_UUID[$job_id]}")" \
        "$(csv_escape "${JOB_KIND[$job_id]}")" \
        "$(csv_escape "${JOB_MODEL[$job_id]}")" \
        "$(csv_escape "${JOB_VARIANT[$job_id]}")" \
        "$(csv_escape "${JOB_FGSM[$job_id]}")" \
        "$(csv_escape "${JOB_N_CAL[$job_id]}")" \
        "$(csv_escape "${JOB_LOSS[$job_id]}")" \
        "$(csv_escape "${JOB_DLOSS[$job_id]}")" \
        "$(csv_escape "${JOB_PROTOTYPE[$job_id]}")" \
        "$(csv_escape "${JOB_N_POS[$job_id]}")" \
        "$(csv_escape "${JOB_N_NEG[$job_id]}")" \
        "$(csv_escape "${JOB_NORMALIZE[$job_id]}")" \
        "$(csv_escape "${JOB_RETRY_COUNT[$job_id]:-0}")" \
        "$(csv_escape "${JOB_ERR_FILE[$job_id]}")" \
        "$(csv_escape "${JOB_STDOUT_FILE[$job_id]}")" \
        >> "$completed_log"
}

resolve_job_state() {
    local job_id="$1"
    local kind="${JOB_KIND[$job_id]}"
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local failed_final="${JOB_FAILED_FINAL[$job_id]:-0}"

    RESOLVED_JOB_STATE="queued"
    RESOLVED_JOB_FINISHED=0

    if [ -n "$kind" ]; then
        if [ "$failed_final" -eq 1 ]; then
            RESOLVED_JOB_STATE="failed"
            RESOLVED_JOB_FINISHED=1
        elif done_marker_exists "$exp_id"; then
            RESOLVED_JOB_STATE="done"
            RESOLVED_JOB_FINISHED=1
        elif job_is_running "$job_id"; then
            RESOLVED_JOB_STATE="running"
        elif job_is_in_retry_queue "$job_id" || job_is_in_pending_queue "$job_id"; then
            RESOLVED_JOB_STATE="queued"
        fi
    fi
}

write_job_status_snapshot() {
    local status_path="$status_file"
    local job_id

    {
        printf 'job_id,job_state,finished,retry_count,exp_id,uuid,kind,model,variant,error_file,stdout_file\n'
        for job_id in $(seq 1 "$total_models"); do
            resolve_job_state "$job_id"
            hydrate_job_uuid "$job_id"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$job_id" \
                "$(csv_escape "$RESOLVED_JOB_STATE")" \
                "$(csv_escape "$RESOLVED_JOB_FINISHED")" \
                "$(csv_escape "${JOB_RETRY_COUNT[$job_id]:-0}")" \
                "$(csv_escape "${JOB_EXP_ID[$job_id]}")" \
                "$(csv_escape "${JOB_UUID[$job_id]}")" \
                "$(csv_escape "${JOB_KIND[$job_id]}")" \
                "$(csv_escape "${JOB_MODEL[$job_id]}")" \
                "$(csv_escape "${JOB_VARIANT[$job_id]}")" \
                "$(csv_escape "${JOB_ERR_FILE[$job_id]}")" \
                "$(csv_escape "${JOB_STDOUT_FILE[$job_id]}")"
        done
    } > "$status_path"
}

update_job_status_entry() {
    local job_id="$1"
    local status_path="$status_file"
    local tmp_path

    if [ ! -f "$status_path" ]; then
        write_job_status_snapshot
        return
    fi

    resolve_job_state "$job_id"
    hydrate_job_uuid "$job_id"
    tmp_path="${status_path}.tmp"
    awk -F, -v job_id="$job_id" \
        -v job_state="$(csv_escape "$RESOLVED_JOB_STATE")" \
        -v finished="$(csv_escape "$RESOLVED_JOB_FINISHED")" \
        -v retry_count="$(csv_escape "${JOB_RETRY_COUNT[$job_id]:-0}")" \
        -v exp_id="$(csv_escape "${JOB_EXP_ID[$job_id]}")" \
        -v uuid="$(csv_escape "${JOB_UUID[$job_id]}")" \
        -v kind="$(csv_escape "${JOB_KIND[$job_id]}")" \
        -v model="$(csv_escape "${JOB_MODEL[$job_id]}")" \
        -v variant="$(csv_escape "${JOB_VARIANT[$job_id]}")" \
        -v error_file="$(csv_escape "${JOB_ERR_FILE[$job_id]}")" \
        -v stdout_file="$(csv_escape "${JOB_STDOUT_FILE[$job_id]}")" '
            BEGIN { OFS="," }
            NR == 1 { print; next }
            $1 == job_id {
                print job_id, job_state, finished, retry_count, exp_id, uuid, kind, model, variant, error_file, stdout_file
                next
            }
            { print }
        ' "$status_path" > "$tmp_path" && mv "$tmp_path" "$status_path"
}

write_job_manifest() {
    local manifest_path="$manifest_file"
    local job_id

    {
        print_job_manifest_header
        for job_id in "${registered_job_ids[@]}"; do
            emit_job_manifest_row "$job_id"
        done
    } > "$manifest_path"

    echo "Job manifest written to: $manifest_path"
}

print_job_manifest_header() {
    printf 'job_id,job_state,kind,model,fgsm,n_calibration,loss,dloss,prototype,n_positives,n_negatives,normalize,auto_select_k,variant,classif_loss,exp_id,uuid,retry_count,error_file,stdout_file,failed_final,n_epochs,task,num_jobs,new_size,early_stop,n_trials,n_trials_done,log_file,run_cnn_mlp,cnn_compare_all,verbose,test_mode,test_tag,triplet_log_comet,triplet_log_mlflow,triplet_log_tracking,triplet_save_repro_artifacts,bs,dataset_name,dataset_key,valid_dataset,test_dataset,train_datasets,user_set_num_jobs,user_set_bs,min_free_mb,mem_per_job_mb,max_oom_retries,poll_interval,gamma,epsilon,force_run,theoretical_gpu_required_mb,reserved_gpu_mb,actual_peak_gpu_mb,oom_missing_gpu_mb,oom_gpu_free_at_failure_mb,oom_gpu_used_at_failure_mb,reservation_source_tag,reservation_match_quality,telemetry_status,finished\n'
}

emit_job_manifest_row() {
    local job_id="$1"
    local kind="${JOB_KIND[$job_id]}"
    local model="${JOB_MODEL[$job_id]}"
    local fgsm="${JOB_FGSM[$job_id]}"
    local n_cal="${JOB_N_CAL[$job_id]}"
    local loss="${JOB_LOSS[$job_id]}"
    local dloss="${JOB_DLOSS[$job_id]}"
    local prototype="${JOB_PROTOTYPE[$job_id]}"
    local n_pos="${JOB_N_POS[$job_id]}"
    local n_neg="${JOB_N_NEG[$job_id]}"
    local normalize="${JOB_NORMALIZE[$job_id]}"
    local auto_k="${JOB_AUTO_K[$job_id]}"
    local variant="${JOB_VARIANT[$job_id]}"
    local classif="${JOB_CLASSIF[$job_id]}"
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local retry_count="${JOB_RETRY_COUNT[$job_id]:-0}"
    local err_file="${JOB_ERR_FILE[$job_id]}"
    local stdout_file="${JOB_STDOUT_FILE[$job_id]}"
    local failed_final="${JOB_FAILED_FINAL[$job_id]:-0}"
    local theoretical_gpu_required_mb="${JOB_THEORETICAL_GPU_MB[$job_id]:-}"
    local n_trials_done="${JOB_N_TRIALS_DONE[$job_id]:-0}"
    local reserved_gpu_mb="${JOB_RESERVED_GPU_MB[$job_id]:-}"
    local actual_peak_gpu_mb="${JOB_ACTUAL_PEAK_GPU_MB[$job_id]:-}"
    local oom_missing_gpu_mb="${JOB_OOM_MISSING_GPU_MB[$job_id]:-}"
    local oom_gpu_free_at_failure_mb="${JOB_OOM_GPU_FREE_AT_FAILURE_MB[$job_id]:-}"
    local oom_gpu_used_at_failure_mb="${JOB_OOM_GPU_USED_AT_FAILURE_MB[$job_id]:-}"
    local reservation_source_tag="${JOB_RESERVATION_SOURCE_TAG[$job_id]:-}"
    local reservation_match_quality="${JOB_RESERVATION_MATCH_QUALITY[$job_id]:-}"
    local telemetry_status="${JOB_TELEMETRY_STATUS[$job_id]:-}"
    local job_state finished

    resolve_job_state "$job_id"
    hydrate_job_uuid "$job_id"
    job_state="$RESOLVED_JOB_STATE"
    finished="$RESOLVED_JOB_FINISHED"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$job_id" \
        "$(csv_escape "$job_state")" \
        "$(csv_escape "$kind")" \
        "$(csv_escape "$model")" \
        "$(csv_escape "$fgsm")" \
        "$(csv_escape "$n_cal")" \
        "$(csv_escape "$loss")" \
        "$(csv_escape "$dloss")" \
        "$(csv_escape "$prototype")" \
        "$(csv_escape "$n_pos")" \
        "$(csv_escape "$n_neg")" \
        "$(csv_escape "$normalize")" \
        "$(csv_escape "$auto_k")" \
        "$(csv_escape "$variant")" \
        "$(csv_escape "$classif")" \
        "$(csv_escape "$exp_id")" \
        "$(csv_escape "${JOB_UUID[$job_id]}")" \
        "$(csv_escape "$retry_count")" \
        "$(csv_escape "$err_file")" \
        "$(csv_escape "$stdout_file")" \
        "$(csv_escape "$failed_final")" \
        "$(csv_escape "$n_epochs")" \
        "$(csv_escape "$task")" \
        "$(csv_escape "$num_jobs")" \
        "$(csv_escape "$new_size")" \
        "$(csv_escape "$early_stop")" \
        "$(csv_escape "$n_trials")" \
        "$(csv_escape "$n_trials_done")" \
        "$(csv_escape "$log_file")" \
        "$(csv_escape "$run_cnn_mlp")" \
        "$(csv_escape "$cnn_compare_all")" \
        "$(csv_escape "$verbose")" \
        "$(csv_escape "$test_mode")" \
        "$(csv_escape "$test_tag")" \
        "$(csv_escape "$triplet_log_comet")" \
        "$(csv_escape "$triplet_log_mlflow")" \
        "$(csv_escape "$triplet_log_tracking")" \
        "$(csv_escape "$triplet_save_repro_artifacts")" \
        "$(csv_escape "$bs")" \
        "$(csv_escape "$dataset_name")" \
        "$(csv_escape "$dataset_key")" \
        "$(csv_escape "$valid_dataset")" \
        "$(csv_escape "$test_dataset")" \
        "$(csv_escape "$train_datasets")" \
        "$(csv_escape "$user_set_num_jobs")" \
        "$(csv_escape "$user_set_bs")" \
        "$(csv_escape "$min_free_mb")" \
        "$(csv_escape "$mem_per_job_mb")" \
        "$(csv_escape "$max_oom_retries")" \
        "$(csv_escape "$poll_interval")" \
        "$(csv_escape "$gamma")" \
        "$(csv_escape "$epsilon")" \
        "$(csv_escape "$force_run")" \
        "$(csv_escape "$theoretical_gpu_required_mb")" \
        "$(csv_escape "$reserved_gpu_mb")" \
        "$(csv_escape "$actual_peak_gpu_mb")" \
        "$(csv_escape "$oom_missing_gpu_mb")" \
        "$(csv_escape "$oom_gpu_free_at_failure_mb")" \
        "$(csv_escape "$oom_gpu_used_at_failure_mb")" \
        "$(csv_escape "$reservation_source_tag")" \
        "$(csv_escape "$reservation_match_quality")" \
        "$(csv_escape "$telemetry_status")" \
        "$(csv_escape "$finished")"
}

update_job_manifest_entry() {
    local job_id="$1"
    local manifest_path="$manifest_file"
    local tmp_path row

    if [ ! -f "$manifest_path" ]; then
        write_job_manifest
        return
    fi

    row="$(emit_job_manifest_row "$job_id")"
    tmp_path="${manifest_path}.tmp"
    awk -F, -v job_id="$job_id" -v row="$row" '
        BEGIN { replaced = 0 }
        NR == 1 { print; next }
        $1 == job_id {
            print row
            replaced = 1
            next
        }
        { print }
        END {
            if (!replaced) {
                print row
            }
        }
    ' "$manifest_path" > "$tmp_path" && mv "$tmp_path" "$manifest_path"
}

write_run_settings() {
    local settings_path="$settings_file"

    {
        printf 'key,value\n'
        printf 'n_epochs,%s\n' "$(csv_escape "$n_epochs")"
        printf 'task,%s\n' "$(csv_escape "$task")"
        printf 'num_jobs,%s\n' "$(csv_escape "$num_jobs")"
        printf 'new_size,%s\n' "$(csv_escape "$new_size")"
        printf 'early_stop,%s\n' "$(csv_escape "$early_stop")"
        printf 'n_trials,%s\n' "$(csv_escape "$n_trials")"
        printf 'log_file,%s\n' "$(csv_escape "$log_file")"
        printf 'auto_select_k,%s\n' "$(csv_escape "$auto_select_k")"
        printf 'run_cnn_mlp,%s\n' "$(csv_escape "$run_cnn_mlp")"
        printf 'cnn_compare_all,%s\n' "$(csv_escape "$cnn_compare_all")"
        printf 'verbose,%s\n' "$(csv_escape "$verbose")"
        printf 'test_mode,%s\n' "$(csv_escape "$test_mode")"
        printf 'test_tag,%s\n' "$(csv_escape "$test_tag")"
        printf 'triplet_log_comet,%s\n' "$(csv_escape "$triplet_log_comet")"
        printf 'triplet_log_mlflow,%s\n' "$(csv_escape "$triplet_log_mlflow")"
        printf 'triplet_log_tracking,%s\n' "$(csv_escape "$triplet_log_tracking")"
        printf 'triplet_save_repro_artifacts,%s\n' "$(csv_escape "$triplet_save_repro_artifacts")"
        printf 'bs,%s\n' "$(csv_escape "$bs")"
        printf 'num_workers,%s\n' "$(csv_escape "$num_workers")"
        printf 'dataset_name,%s\n' "$(csv_escape "$dataset_name")"
        printf 'dataset_key,%s\n' "$(csv_escape "$dataset_key")"
        printf 'valid_dataset,%s\n' "$(csv_escape "$valid_dataset")"
        printf 'test_dataset,%s\n' "$(csv_escape "$test_dataset")"
        printf 'train_datasets,%s\n' "$(csv_escape "$train_datasets")"
        printf 'user_set_num_jobs,%s\n' "$(csv_escape "$user_set_num_jobs")"
        printf 'user_set_bs,%s\n' "$(csv_escape "$user_set_bs")"
        printf 'min_free_mb,%s\n' "$(csv_escape "$min_free_mb")"
        printf 'mem_per_job_mb,%s\n' "$(csv_escape "$mem_per_job_mb")"
        printf 'max_oom_retries,%s\n' "$(csv_escape "$max_oom_retries")"
        printf 'poll_interval,%s\n' "$(csv_escape "$poll_interval")"
        printf 'gamma,%s\n' "$(csv_escape "$gamma")"
        printf 'epsilon,%s\n' "$(csv_escape "$epsilon")"
        printf 'force_run,%s\n' "$(csv_escape "$force_run")"
    } > "$settings_path"

    echo "Run settings written to: $settings_path"
}

persist_launcher_metadata() {
    local job_id
    # Force-refresh UUID cache once, then hydrate all jobs so status/manifest
    # snapshots consistently include UUID values when metadata exists.
    build_uuid_cache 1
    for job_id in $(seq 1 "$total_models"); do
        hydrate_job_uuid "$job_id"
    done

    write_run_settings
    write_job_status_snapshot
    write_job_manifest
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
    dequeued_from_queue="retry"
    return 0
}

dequeue_pending() {
    if [ ${#pending_queue[@]} -eq 0 ]; then
        return 1
    fi
    dequeued_job_id="${pending_queue[0]}"
    pending_queue=("${pending_queue[@]:1}")
    dequeued_from_queue="pending"
    return 0
}

requeue_front() {
    local queue_name="$1"
    local job_id="$2"
    if [ "$queue_name" = "retry" ]; then
        retry_queue=("$job_id" "${retry_queue[@]}")
    else
        pending_queue=("$job_id" "${pending_queue[@]}")
    fi
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
    if [ "$free" -lt "$min_free_mb" ]; then
        echo 0
        return
    fi
    # Slot cap is now governed by per-job reservation checks in scheduler_loop.
    echo "$num_jobs"
}

is_oom_error() {
    local err_file="$1"
    grep -qiE "torch\.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|out of memory" "$err_file"
}

is_oom_for_job() {
    local job_id="$1"
    local err_file="${JOB_ERR_FILE[$job_id]}"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    local pattern="torch\.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|out of memory"

    if [ -f "$err_file" ] && grep -qiE "$pattern" "$err_file"; then
        return 0
    fi
    if [ -f "$out_file" ] && grep -qiE "$pattern" "$out_file"; then
        return 0
    fi
    return 1
}

has_fatal_error_marker_for_job() {
    local job_id="$1"
    local err_file="${JOB_ERR_FILE[$job_id]}"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    # Keep this conservative to avoid false positives.
    local pattern="Error occurred:|Traceback \(most recent call last\)|RuntimeError:"

    if [ -f "$err_file" ] && grep -qiE "$pattern" "$err_file"; then
        return 0
    fi
    if [ -f "$out_file" ] && grep -qiE "$pattern" "$out_file"; then
        return 0
    fi
    return 1
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
    registered_job_ids+=("$job_id")
    JOB_N_TRIALS_DONE[$job_id]=0
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
    JOB_BS[$job_id]="$bs"
    JOB_RETRY_COUNT[$job_id]=0
    JOB_ERR_FILE[$job_id]="${tmp_error_dir}/${exp_id}_error.log"
    JOB_STDOUT_FILE[$job_id]="${tmp_log_dir}/${exp_id}.log"
    JOB_FAILED_FINAL[$job_id]=0
    JOB_VARIANT[$job_id]=""
    JOB_CLASSIF[$job_id]=""
    apply_reservation_for_job "$job_id"

    if [ "$force_run" -eq 0 ]; then
        if strict_state_gate_blocks_job "$exp_id"; then
            JOB_BLOCKED_PRIOR_STATE[$job_id]="${PREV_JOB_STATE_BY_EXP[$exp_id]}"
            return
        fi
        if done_marker_exists "$exp_id"; then
            return
        fi
    fi

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
    registered_job_ids+=("$job_id")
    JOB_N_TRIALS_DONE[$job_id]=0
    JOB_KIND[$job_id]="cnn_mlp"
    JOB_MODEL[$job_id]="$model"
    JOB_VARIANT[$job_id]="$variant"
    JOB_NORMALIZE[$job_id]="$normalize"
    JOB_N_CAL[$job_id]="$n_calibration"
    JOB_FGSM[$job_id]="$fgsm"
    JOB_DLOSS[$job_id]="$dloss"
    JOB_CLASSIF[$job_id]="$classif_loss"
    JOB_BS[$job_id]="$bs"
    JOB_RETRY_COUNT[$job_id]=0
    JOB_ERR_FILE[$job_id]="${tmp_error_dir}/${exp_id}_error.log"
    JOB_STDOUT_FILE[$job_id]="${tmp_log_dir}/${exp_id}.log"
    JOB_FAILED_FINAL[$job_id]=0
    JOB_LOSS[$job_id]=""
    JOB_PROTOTYPE[$job_id]=""
    JOB_N_POS[$job_id]=""
    JOB_N_NEG[$job_id]=""
    JOB_AUTO_K[$job_id]=""
    apply_reservation_for_job "$job_id"

    if [ "$force_run" -eq 0 ]; then
        if strict_state_gate_blocks_job "$exp_id"; then
            JOB_BLOCKED_PRIOR_STATE[$job_id]="${PREV_JOB_STATE_BY_EXP[$exp_id]}"
            return
        fi
        if done_marker_exists "$exp_id"; then
            return
        fi
    fi

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
    local split_args=(--valid_dataset="$valid_dataset")

    if [ -n "$test_dataset" ]; then
        split_args+=(--test_dataset="$test_dataset")
    fi

    if [ -n "$train_datasets" ]; then
        split_args+=(--train_datasets="$train_datasets")
    fi

    if [ "$verbose" -eq 1 ]; then
        echo "Launching siamese: model=$model fgsm=$fgsm n_calibration=$n_calibration loss=$loss dloss=$dloss prototype=$prototype n_pos=$n_positives n_neg=$n_negatives normalize=$normalize"
    fi

    PYTHONPATH="/home/simon/otitenet" /home/simon/otitenet/.conda/bin/python -m otitenet.train.train_triplet_new \
        --path="$dataset_path" \
        --siamese_inference="$siamese_inference" \
        --model_name="$model" \
        --n_calibration="$n_calibration" --is_stn=0 --dloss="$dloss" --device=cuda:0 \
        --classif_loss="$loss" --n_epochs="$n_epochs" --task="$task" --weighted_sampler=0 \
        --groupkfold=1 --n_negatives="$n_negatives" --n_positives="$n_positives" --prototypes_to_use="$prototype" --fgsm="$fgsm" \
        "${split_args[@]}" \
        --new_size="$new_size" --early_stop="$early_stop" --n_trials="$n_trials" --normalize="$normalize" --auto_select_k="$auto_k" \
        --bs="${JOB_BS[$job_id]:-$bs}" --log_comet="$triplet_log_comet" --log_mlflow="$triplet_log_mlflow" --log_tracking="$triplet_log_tracking" \
        --num_workers="$num_workers" \
        --save_repro_artifacts="$triplet_save_repro_artifacts" --run_explainability=0 \
        --exp_id="$exp_id" --run_tag="$test_tag" \
        > "$out_file" 2> "$err_file"
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
    local split_args=(--valid_dataset="$valid_dataset")

    if [ -n "$test_dataset" ]; then
        split_args+=(--test_dataset="$test_dataset")
    fi

    if [ -n "$train_datasets" ]; then
        split_args+=(--train_datasets="$train_datasets")
    fi

    if [ "$verbose" -eq 1 ]; then
        echo "Launching $variant: model=$model normalize=$normalize n_calibration=$n_calibration fgsm=$fgsm dloss=$dloss classif_loss=$classif_loss"
    fi

    PYTHONPATH="/home/simon/otitenet" /home/simon/otitenet/.conda/bin/python -m otitenet.train.train_cnn_mlp_compare \
        --path="$dataset_path" --device=cuda:0 --task="$task" --groupkfold=1 --weighted_sampler=0 \
        "${split_args[@]}" \
        --model_name="$model" --variant="$variant" --compare_all="$cnn_compare_all" --normalize="$normalize" \
        --new_size="$new_size" --n_epochs="$n_epochs" --early_stop="$early_stop" --bs="${JOB_BS[$job_id]:-$bs}" \
        --n_calibration="$n_calibration" --fgsm="$fgsm" --dloss="$dloss" --classif_loss="$classif_loss" \
        --gamma="$gamma" --epsilon="$epsilon" --num_workers="$num_workers" --run_tag="$test_tag" --exp_id="$exp_id" --verbose="$verbose" --log_mlflow=0 \
        > "$out_file" 2> "$err_file"
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
            echo "[retry=$retries] launching $kind job_id=$job_id model=$model free_mb=$free_mb reserve_mb=${JOB_RESERVED_GPU_MB[$job_id]:-$mem_per_job_mb}"
        else
            echo "launching $kind job_id=$job_id model=$model free_mb=$free_mb reserve_mb=${JOB_RESERVED_GPU_MB[$job_id]:-$mem_per_job_mb}"
        fi
    fi

    run_job "$job_id" &
    local pid=$!
    running_pids+=("$pid")
    PID_TO_JOB_ID[$pid]="$job_id"
    append_runtime_event "launched" "$job_id" "$pid"
    update_job_status_entry "$job_id"
    update_job_manifest_entry "$job_id"
}

finalize_job_result() {
    local job_id="$1"
    local exit_code="$2"
    local err_file="${JOB_ERR_FILE[$job_id]}"
    local done_file
    done_file="$(done_marker_path "${JOB_EXP_ID[$job_id]}")"

    # Some training paths catch exceptions and exit 0 while printing errors.
    # Promote those cases to failures based on log content.
    if [ "$exit_code" -eq 0 ]; then
        if is_oom_for_job "$job_id"; then
            exit_code=137
        elif has_fatal_error_marker_for_job "$job_id"; then
            exit_code=1
        fi
    fi

    if [ "$exit_code" -eq 0 ]; then
        local artifact_check
        artifact_check="$(validate_job_artifacts "$job_id")"
        if [ "$artifact_check" != "ok" ]; then
            exit_code=2
            {
                echo "Artifact validation failed for job_id=${job_id} exp_id=${JOB_EXP_ID[$job_id]}"
                echo "Reason: ${artifact_check}"
                echo "Required: model.pth, run_metadata.json, and prototypes.pkl when prototypes are used"
                echo "Timestamp: $(date)"
                echo "-----------------------------------------------------"
            } >> "$log_file"
        fi
    fi

    if [ "$exit_code" -eq 0 ]; then
        hydrate_job_gpu_telemetry "$job_id"
        if [ "$test_tag" = "TEST_SMOKE" ] && [ -n "${JOB_ACTUAL_PEAK_GPU_MB[$job_id]:-}" ]; then
            JOB_RESERVED_GPU_MB[$job_id]="${JOB_ACTUAL_PEAK_GPU_MB[$job_id]}"
            JOB_RESERVATION_SOURCE_TAG[$job_id]="test_smoke_self_measured_peak"
            JOB_RESERVATION_MATCH_QUALITY[$job_id]="measured"
        fi
        cleanup_job_processes "$job_id"
        touch "$done_file"
        append_runtime_event "completed" "$job_id" "" "$exit_code"
        append_completed_run_event "$job_id" "completed" "$exit_code"
        update_job_status_entry "$job_id"
        update_job_manifest_entry "$job_id"
        finished_jobs=$((finished_jobs + 1))
        progress_tick "completed job_id=$job_id"
        print_job_logs "$job_id"

        # Move completed job logs from tmp/ -> success/
        local exp_id="${JOB_EXP_ID[$job_id]}"
        local out_file="${JOB_STDOUT_FILE[$job_id]}"
        [ -f "$out_file" ] && mv "$out_file" "${success_log_dir}/${exp_id}.log"
        if [ -f "$err_file" ] && [ -s "$err_file" ]; then
            mv "$err_file" "${success_warning_dir}/${exp_id}.warn"
        fi
        [ -f "$done_file" ] && mv "$done_file" "${success_done_dir}/"
        return
    fi

    rm -f "$done_file"
    hydrate_job_gpu_telemetry "$job_id"
    cleanup_job_processes "$job_id"

    if is_oom_for_job "$job_id" && [ "${JOB_RETRY_COUNT[$job_id]}" -lt "$max_oom_retries" ]; then
        JOB_RETRY_COUNT[$job_id]=$((JOB_RETRY_COUNT[$job_id] + 1))
        
        # Adaptive OOM handling: reduce job-specific batch size
        local old_bs="${JOB_BS[$job_id]:-$bs}"
        local new_bs=$((old_bs > 2 ? old_bs / 2 : 1))
        JOB_BS[$job_id]="$new_bs"
        
        # Increase specific job memory reservation for better scheduling
        local current_res
        current_res="$(required_gpu_mb_for_job "$job_id")"
        JOB_RESERVED_GPU_MB[$job_id]=$((current_res * 2))

        append_runtime_event "oom_retry" "$job_id" "" "$exit_code"
        enqueue_retry "$job_id"
        update_job_status_entry "$job_id"
        update_job_manifest_entry "$job_id"
        echo "OOM detected, re-queued job_id=$job_id (retry ${JOB_RETRY_COUNT[$job_id]}/$max_oom_retries, new_bs=$new_bs, reserved=${JOB_RESERVED_GPU_MB[$job_id]}MB)"
        print_job_logs "$job_id"
        return
    fi

    JOB_FAILED_FINAL[$job_id]=1
    append_runtime_event "failed" "$job_id" "" "$exit_code"
    append_completed_run_event "$job_id" "failed" "$exit_code"
    update_job_status_entry "$job_id"
    update_job_manifest_entry "$job_id"
    record_failure "$job_id" "$exit_code"
    finished_jobs=$((finished_jobs + 1))
    progress_tick "failed job_id=$job_id exit=$exit_code"

    # Move failed job logs from tmp/ -> failed/
    local exp_id="${JOB_EXP_ID[$job_id]}"
    local out_file="${JOB_STDOUT_FILE[$job_id]}"
    local err_file="${JOB_ERR_FILE[$job_id]}"
    [ -f "$out_file" ] && mv "$out_file" "${failed_log_dir}/${exp_id}.log"
    [ -f "$err_file" ] && mv "$err_file" "${failed_error_dir}/${exp_id}.error.log"

    print_job_logs "$job_id"
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
            for model in resnet18 resnet50 vit_b_16 vit_b_16_384 densenet161 vgg16 efficientnet_b0 densenet121; do
                for fgsm in 0 1; do
                    for dloss in no inverseTriplet; do
                        # 1) Siamese jobs first for this shared parameter set
                        for loss in arcface triplet; do
                            for prototype in batch no class; do
                                if ! [[ "$prototype" == "batch" && "$dloss" == "no" ]]; then
                                    exp_id="${test_tag}_${task}_${dataset_key}_siamese_${model}_${loss}_${dloss}_proto${prototype}_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                                    register_siamese_job "$model" "$fgsm" "$n_calibration" "$loss" "$dloss" "$prototype" 1 1 "$normalize" "$exp_id"
                                fi
                            done
                        done

                        for classif_loss in ce hinge; do
                            exp_id="${test_tag}_${task}_${dataset_key}_siamese_${model}_${classif_loss}_${dloss}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                            register_siamese_job "$model" "$fgsm" "$n_calibration" "$classif_loss" "$dloss" "no" 1 1 "$normalize" "$exp_id"
                        done

                        exp_id="${test_tag}_${task}_${dataset_key}_siamese_${model}_softmax_contrastive_${dloss}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                        register_siamese_job "$model" "$fgsm" "$n_calibration" "softmax_contrastive" "$dloss" "no" 1 1 "$normalize" "$exp_id"

                        for n_negatives in 1 5; do
                            exp_id="${test_tag}_${task}_${dataset_key}_siamese_${model}_triplet_${dloss}_nneg${n_negatives}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                            register_siamese_job "$model" "$fgsm" "$n_calibration" "triplet" "$dloss" "no" 1 "$n_negatives" "$normalize" "$exp_id"
                        done

                    done

                    # 2) CNN/MLP with full BER sweep (no, DANN, inverseTriplet) — separate from siamese
                    if [ "$run_cnn_mlp" -eq 1 ]; then
                        for dloss in no DANN inverseTriplet; do
                            for classif_loss in ce hinge; do
                                exp_id="${test_tag}_${task}_${dataset_key}_cnn_${model}_${classif_loss}_${dloss}_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                                register_cnn_mlp_job "$model" "cnn_transfer" "$normalize" "$n_calibration" "$fgsm" "$dloss" "$classif_loss" "$exp_id"

                                exp_id="${test_tag}_${task}_${dataset_key}_mlp_${model}_${classif_loss}_${dloss}_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
                                register_cnn_mlp_job "$model" "mlp" "$normalize" "$n_calibration" "$fgsm" "$dloss" "$classif_loss" "$exp_id"
                            done
                        done
                    fi
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
        if done_marker_exists "$exp_id"; then
            done_count=$((done_count + 1))
        fi
    done
    echo "$done_count"
}

scheduler_loop() {
    local free_mb slot_cap running_count loop_counter required_mb
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
            required_mb="$(required_gpu_mb_for_job "$dequeued_job_id")"
            if [ "$free_mb" -lt "$required_mb" ]; then
                requeue_front "$dequeued_from_queue" "$dequeued_job_id"
                if [ "$verbose" -eq 0 ]; then
                    echo "Pausing launches: job_id=$dequeued_job_id requires ${required_mb}MB but free=${free_mb}MB"
                fi
                break
            fi
            launch_job "$dequeued_job_id"
            running_count=${#running_pids[@]}
            free_mb="$(gpu_free_mb)"
            slot_cap="$(dynamic_slot_cap "$free_mb")"
        done

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

        render_progress_bar

        sleep "$poll_interval"
    done

    if [ "$show_progress_bar" -eq 1 ]; then
        echo
    fi
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

                    done

                    # CNN/MLP with extended BER sweep (no, DANN, inverseTriplet)
                    if [ "$run_cnn_mlp" -eq 1 ]; then
                        for dloss in no DANN inverseTriplet; do
                            for classif_loss in ce hinge; do
                                count=$((count + 1)) # cnn_transfer
                                count=$((count + 1)) # mlp
                            done
                        done
                    fi
                done
            done
        done
    done
    total_models=$count
}

load_previous_manifest_state_cache
load_smoke_reservation_cache
generate_jobs
trap persist_launcher_metadata EXIT
persist_launcher_metadata
write_job_status_snapshot

already_done=$(count_already_done)
finished_jobs=$already_done
to_be_run=$((total_models - already_done))
echo "Scheduler config: num_jobs=$num_jobs min_free_mb=$min_free_mb mem_per_job_mb=$mem_per_job_mb max_oom_retries=$max_oom_retries"
echo "Dataset config: dataset_name=$dataset_name dataset_key=$dataset_key dataset_path=$dataset_path"
echo "Split config: valid_dataset=$valid_dataset test_dataset=${test_dataset:-<auto>} train_datasets=${train_datasets:-<auto>}"
echo "Planned jobs: $total_models (already completed: $already_done, to be run: $to_be_run)"

scheduler_loop

failed_count=0
for job_id in $(seq 1 "$total_models"); do
    if [ "${JOB_FAILED_FINAL[$job_id]:-0}" -eq 1 ]; then
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
echo "Log structure: ${progress_root}/{tmp,success,failed,csv}/"
echo "  Active logs   : ${tmp_root}/{error,log,warning,done,cache,db}/"
echo "  Success logs  : ${success_root}/{log,warning,done}/"
echo "  Failed logs   : ${failed_root}/{error,log}/"
echo "  CSVs          : ${csv_dir}/"
echo "Cumulative error log: $cumulative_error_log"
echo "Live job status: ${status_file}"
echo "Live completed runs ledger: ${completed_runs_file}"
echo "All jobs completed. Failed jobs: $failed_count/$total_models"
