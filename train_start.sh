#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
MODEL_PATH="$PROJECT_DIR/model.joblib"
CSV_PATH="${1:-$PROJECT_DIR/outfits.csv}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found at $VENV_PATH"
  echo "Create one first: python3 -m venv .venv"
  exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found at $CSV_PATH"
  echo "Usage: ./train_start.sh /path/to/outfits.csv"
  exit 1
fi

source "$VENV_PATH/bin/activate"

echo "Training model..."
python3 "$PROJECT_DIR/train_model.py" --csv "$CSV_PATH" --out "$MODEL_PATH"

echo "Starting app..."
python3 "$PROJECT_DIR/app.py"
