#!/bin/bash
set -e

# Print header
echo "=================================="
echo "Fake News Detection Model Setup"
echo "=================================="
echo

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed data/evaluation
mkdir -p model/checkpoints

# Download datasets
echo "Downloading datasets..."
python -m utils.download_datasets

# Prepare datasets
echo "Preparing datasets..."
python -m utils.preprocessing

echo
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo
echo "To activate the environment, run:"
echo "source .venv/bin/activate"
echo
echo "To train the model, run:"
echo "python -m model.training.train"
echo
echo "To start the API server, run:"
echo "python -m api.server"
echo 
echo "To test the model with a claim, run:"
echo "python -m model.inference.test_model --claim \"Your claim here\""
echo "==================================" 