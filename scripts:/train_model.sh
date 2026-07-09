#!/bin/bash
echo "🧹 Cleaning terminal workspace..."
clear

echo "🚀 Starting Production Training Pipeline..."
# Move up out of the scripts directory to the project root
cd "$(dirname "$0")/.."

# Run the training master script
python3 main.py