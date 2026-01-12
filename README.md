# Advik Outfit Predictor

Train a model from a CSV and serve predictions with a Flask web app.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Train

Place your CSV in this folder and run:

```bash
./train_start.sh
```

Expected columns (case-insensitive):
- Date (optional)
- Day (optional)
- Top (optional)
- Bottom (optional)
- Outfit (optional; inferred as "Top | Bottom" when missing)

Rows with missing labels or day are ignored.

## Admin credentials

Set environment variables before running:

```bash
export ADMIN_USER="advik67"
export ADMIN_PASSWORD="advik67"
export FLASK_SECRET_KEY="change-me"
```

Optional:
```bash
export CSV_PATH="outfits.csv"
export MODEL_PATH="model.joblib"
export CONFIG_PATH="admin_config.json"
```
