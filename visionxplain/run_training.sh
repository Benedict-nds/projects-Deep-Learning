#!/bin/bash
# Training script wrapper that sets up Python path correctly

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project root
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include project root
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run the training script with all arguments
python src/training/train.py "$@"

