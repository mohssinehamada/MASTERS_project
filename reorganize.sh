#!/bin/bash

# This script reorganizes the project files into the new structure
echo "Reorganizing project files..."

# Create data directory for results
mkdir -p data

# Move result files to data directory
echo "Moving result files to data directory..."
mv results_*.json data/ 2>/dev/null
mv benchmark_report_*.json data/ 2>/dev/null
mv raw_content_*.txt data/ 2>/dev/null

# Delete old Python files (they've been reorganized into the src directory)
echo "Cleaning up old files..."
rm -f agent.py browser_tools.py opendeepsearch_tools.py extractors.py benchmark.py run_tests.py utils.py test_search.py

echo "Reorganization complete!"
echo ""
echo "The project is now structured as follows:"
echo "- src/: Source code in proper packages"
echo "- data/: Data files and benchmark results"
echo "- tests/: Test files"
echo "- main.py: Main entry point"
echo "- requirements.txt: Dependencies"
echo ""
echo "To run the agent: python main.py agent"
echo "To run benchmarks: python main.py benchmark" 