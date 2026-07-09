#!/bin/bash
echo "🧪 Running Automated Test Suite via PyTest..."
cd "$(dirname "$0")/.."

# Execute pytest through the python module interface
python3 -m pytest -v

echo "🧹 Cleaning up bytecode artifacts..."
find . -type d -name "__pycache__" -exec rm -rf {} +