
#!/bin/bash
# written with zotgpt - claude sonnet 3.7 
# tests.sh - Run the test suite for EZ diffusion model

echo "Running EZ Diffusion Test Suite"

# Navigate to the test directory (if script is run from project root)
cd "$(dirname "$0")" || cd test

# Run the test suite
python testsuite.py

echo "Tests complete!"
