#!/bin/bash
# Master script to run the complete VisionXplain pipeline
# This script runs: tests, training, evaluation, benchmarks, and explainability

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

# Run the Python master script with all arguments
python run_all.py "$@"

