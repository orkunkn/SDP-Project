#!/bin/bash

# Create and activate virtual environment
python -m venv sb3_env
source sb3_env/bin/activate

# Upgrade pip and install required packages
#pip install --upgrade pip
pip install build scipy pandas stable-baselines3 sb3-contrib gymnasium

echo
echo "✅ Setup complete. To activate your environment later, run:"
#echo "source ~/sb3_env/bin/activate"
