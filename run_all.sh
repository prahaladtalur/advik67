#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
MODEL_PATH="$PROJECT_DIR/model.joblib"

CSV_PATH="${1:-$PROJECT_DIR/outfits.csv}"

echo "Project: $PROJECT_DIR"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing dependencies..."
python3 -m pip install -r "$PROJECT_DIR/requirements.txt"

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV not found at $CSV_PATH"
    echo "Usage: ./run_all.sh /path/to/outfits.csv"
    exit 1
  fi
  echo "Training model..."
  python3 "$PROJECT_DIR/train_model.py" --data "$CSV_PATH" --model "$MODEL_PATH"
fi

echo "Starting app..."
python3 "$PROJECT_DIR/app.py"
