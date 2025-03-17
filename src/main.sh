#!/bin/bash

# main.sh - Run the EZ diffusion simulate-and-recover exercise
#written using zotgpt - claude sonnet 3.7
echo "Running EZ Diffusion Simulate-and-Recover Exercise"

# Navigate to the src directory (if script is run from project root)
cd "$(dirname "$0")" || cd src

# Run the simulation
python simulate.py

echo "Simulation complete!"
