#!/bin/bash
echo "🏡 Launching House Price Prediction Web App..."
cd "$(dirname "$0")/.."

# Boot the streamlit engine and force open browser interface
python3 -m streamlit run app/main_interface.py --server.headless false