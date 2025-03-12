#!/bin/bash

# Set default values (can be modified if needed)
START_IDX=${1:--0}       # Starting index (default: 0)
STEP_SIZE=${2:-1000}    # Range for each task (default: 1000)
NUM_TASKS=${3:-3}       # Number of tasks to run (default: 5)

CONFIG_PATH="/home/muzammal/Projects/TRIG/config/p2p_t.yaml"


# Run multiple eval.py tasks in parallel
for ((i=0; i<NUM_TASKS; i++))
do
    END_IDX=$((START_IDX + STEP_SIZE))
    echo "Launching eval.py from index $START_IDX to $END_IDX"

    CUDA_VISIBLE_DEVICES=2 python eval.py --config "$CONFIG_PATH" --start_idx "$START_IDX" --end_idx "$END_IDX" &

    START_IDX=$END_IDX
done

wait  # Wait for all background processes to complete
