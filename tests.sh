# test/test.sh
#!/bin/bash
# written with zotgpt - claude sonnet 3.7 
# Navigate to the script's directory
cd "$(dirname "$0")"

# Run the test suite
echo "Running test suite for EZ diffusion model..."
python -m unittest testsuite.py

echo "Tests complete."
