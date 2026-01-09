#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
MODEL_PATH="$PROJECT_DIR/model.joblib"

CSV_PATH="${1:-$PROJECT_DIR/outfits.csv}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

source "$VENV_PATH/bin/activate"

echo "Installing dependencies (safe to re-run)..."
python3 -m pip install -r "$PROJECT_DIR/requirements.txt"

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV not found at $CSV_PATH"
    echo "Provide a CSV path: ./start.sh /path/to/outfits.csv"
    exit 1
  fi
  echo "Training model..."
  python3 "$PROJECT_DIR/train_model.py" --data "$CSV_PATH" --model "$MODEL_PATH"
fi

echo "Starting app..."
python3 "$PROJECT_DIR/app.py"
