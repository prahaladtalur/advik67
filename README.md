# Advik Outfit Predictor

Train a model from a CSV and serve predictions with a Flask web app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Place your CSV in this folder and run:

```bash
python train_model.py --data outfits.csv --model model.joblib
```

Expected columns (case-insensitive):
- Date (optional)
- Day (optional)
- Top (optional)
- Bottom (optional)
- Outfit (optional; inferred as "Top | Bottom" when missing)

Rows with missing labels or day are ignored.

## Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.
