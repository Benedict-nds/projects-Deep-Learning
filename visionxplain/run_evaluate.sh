#!/bin/bash
# Evaluation script wrapper that ensures virtual environment is used

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run evaluation with all arguments
python scripts/evaluation/evaluate_model.py "$@"



