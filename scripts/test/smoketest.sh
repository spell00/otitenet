#!/bin/bash
set -e

echo "Starting OtiteNet Smoke Test..."

# Run launch.sh in test mode
# Use a custom task name for the test to avoid polluting PROD logs
TASK_NAME="SMOKE_TEST_$(date +%s)"
bash launch.sh --test --task="$TASK_NAME" --jobs=2

PROGRESS_ROOT="logs/progresses/$TASK_NAME"

echo "Verifying directory structure..."
folders=("tmp" "success" "failed" "csv")
for folder in "${folders[@]}"; do
    if [ ! -d "$PROGRESS_ROOT/$folder" ]; then
        echo "ERROR: Directory $PROGRESS_ROOT/$folder missing!"
        exit 1
    fi
done

echo "Verifying CSV logs location..."
if [ ! -f "$PROGRESS_ROOT/csv/TEST_SMOKE_${TASK_NAME}_job_manifest.csv" ]; then
    echo "ERROR: Manifest file missing in csv/ folder!"
    exit 1
fi

echo "Verifying training logs moved to success/..."
# In test mode, at least one job should succeed
SUCCESS_LOGS=$(ls "$PROGRESS_ROOT/success/log/"*.log 2>/dev/null || true)
if [ -z "$SUCCESS_LOGS" ]; then
    echo "ERROR: No logs found in success/log/!"
    exit 1
fi

echo "Smoke test PASSED successfully!"
