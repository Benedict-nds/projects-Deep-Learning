#!/bin/bash
# Training script that ensures virtual environment is used

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run training with all arguments
python src/training/train.py "$@"



