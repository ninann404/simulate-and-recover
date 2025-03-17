# src/main.sh
#!/bin/bash
# written with zotgpt - claude sonnet 3.7 
# Navigate to the script's directory
cd "$(dirname "$0")"

# Run the simulation
echo "Running EZ diffusion simulate-and-recover exercise..."
python simulate.py

echo "Simulation complete. Results saved in the results directory."
