#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"

echo "Creating virtual environment..."
python3 -m venv "$VENV_PATH"

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing dependencies..."
python3 -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "Setup complete."
