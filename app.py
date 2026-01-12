from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, render_template, request

from models import load_model

app = Flask(__name__)

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
FORCE_SYNTHETIC = os.environ.get("FORCE_SYNTHETIC", "1") == "1"
BLACK_BONUS = float(os.environ.get("BLACK_BONUS", "0.35"))

SYNTHETIC_TOP_WEIGHTS = {
    "Black QZ": 0.2,
    "Black Hoodie": 0.2,
    "Camo Hoodie": 0.15,
    "Cape Cod": 0.15,
    "Tow Truck": 0.15,
    "Columbia Puffer": 0.14,
    "Blue Patagonia Pullover": 0.01,
}

SYNTHETIC_BOTTOM_WEIGHTS = {
    "Grey Sweats": 0.45,
    "Black Sweats": 0.44,
    "Tan Sweats": 0.10,
    "Navy Pants": 0.01,
}

_model_bundle: Optional[Dict[str, object]] = None
_model_error: Optional[str] = None


def get_model_bundle() -> Optional[Dict[str, object]]:
    global _model_bundle, _model_error
    if _model_bundle is not None or _model_error is not None:
        return _model_bundle
    try:
        _model_bundle = load_model(MODEL_PATH)
    except FileNotFoundError:
        _model_error = f"Model file not found at {MODEL_PATH}."
    except Exception as exc:  # noqa: BLE001
        _model_error = f"Failed to load model: {exc}"
    return _model_bundle


def _get_pipeline(model_bundle: Dict[str, object]):
    if "pipeline" in model_bundle:
        return model_bundle["pipeline"]
    if "model" in model_bundle:
        return model_bundle["model"]
    raise KeyError("Model bundle missing pipeline/model.")


def _get_feature_columns(model_bundle: Dict[str, object]) -> List[str]:
    feature_columns = model_bundle.get("feature_columns")
    if isinstance(feature_columns, list) and feature_columns:
        return feature_columns
    return ["Day"]


def _get_label_counts(model_bundle: Dict[str, object]) -> Dict[str, int]:
    label_counts = model_bundle.get("label_counts")
    if isinstance(label_counts, dict):
        return {str(k): int(v) for k, v in label_counts.items()}
    return {}


def _get_classes(pipeline) -> List[str]:
    for step_name in ("model", "clf"):
        if hasattr(pipeline, "named_steps") and step_name in pipeline.named_steps:
            return list(pipeline.named_steps[step_name].classes_)
    if hasattr(pipeline, "classes_"):
        return list(pipeline.classes_)
    raise AttributeError("Model does not expose classes_.")


def _is_black_outfit(outfit: str) -> bool:
    return "black" in outfit.lower()


def _colors_in_text(text: str) -> set:
    tokens = {"black", "grey", "gray", "tan", "navy", "blue", "camo"}
    found = set()
    lower = text.lower()
    for token in tokens:
        if token in lower:
            found.add(token)
    if "gray" in found:
        found.add("grey")
    return found


def _is_mono_color(top: str, bottom: str) -> bool:
    top_colors = _colors_in_text(top)
    bottom_colors = _colors_in_text(bottom)
    if not top_colors or not bottom_colors:
        return False
    return bool(top_colors & bottom_colors)


def _prioritize_black(
    ranked: List[tuple], top_k: int
) -> List[Dict[str, object]]:
    results = []
    for outfit, score in ranked[:top_k]:
        results.append({"outfit": outfit, "probability": float(score)})

    if any(_is_black_outfit(item["outfit"]) for item in results):
        return results

    for outfit, score in ranked[top_k:]:
        if _is_black_outfit(outfit):
            results[-1] = {"outfit": outfit, "probability": float(score)}
            break

    return results


def _history_based_outfits(
    label_counts: Dict[str, int], top_k: int
) -> List[Dict[str, object]]:
    if not label_counts:
        return []
    ranked_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    total = sum(label_counts.values()) or 1
    ranked = [(outfit, count / total) for outfit, count in ranked_counts]
    return _prioritize_black(ranked, top_k)


def predict_outfits(
    model_bundle: Dict[str, object], day: str, top_k: int
) -> List[Dict[str, object]]:
    label_counts = _get_label_counts(model_bundle)
    if FORCE_SYNTHETIC and label_counts:
        return _history_based_outfits(label_counts, top_k)
    if FORCE_SYNTHETIC:
        return generate_synthetic_outfits(top_k)

    pipeline = _get_pipeline(model_bundle)
    feature_columns = _get_feature_columns(model_bundle)
    features = {col: "Unknown" for col in feature_columns}
    for col in feature_columns:
        if col.lower() == "day":
            features[col] = day
    frame = pd.DataFrame([features])
    probabilities = pipeline.predict_proba(frame)[0]
    classes = _get_classes(pipeline)
    items = []
    for outfit, score in zip(classes, probabilities):
        if label_counts and outfit not in label_counts:
            continue
        if _is_black_outfit(outfit):
            score *= 1.0 + BLACK_BONUS
        items.append(
            {
                "outfit": outfit,
                "probability": float(score),
                "count": label_counts.get(outfit, 0) if label_counts else 0,
            }
        )

    if label_counts:
        items.sort(key=lambda item: (item["count"], item["probability"]), reverse=True)
    else:
        items.sort(key=lambda item: item["probability"], reverse=True)

    ranked = [(item["outfit"], item["probability"]) for item in items]
    return _prioritize_black(ranked, top_k)


def generate_synthetic_outfits(top_k: int) -> List[Dict[str, object]]:
    tops = list(SYNTHETIC_TOP_WEIGHTS.keys())
    top_weights = list(SYNTHETIC_TOP_WEIGHTS.values())
    bottoms = list(SYNTHETIC_BOTTOM_WEIGHTS.keys())
    bottom_weights = list(SYNTHETIC_BOTTOM_WEIGHTS.values())

    results = []
    seen = set()
    attempts = 0
    max_attempts = 50

    while len(results) < top_k and attempts < max_attempts:
        top = random.choices(tops, weights=top_weights, k=1)[0]
        bottom = random.choices(bottoms, weights=bottom_weights, k=1)[0]
        if _is_mono_color(top, bottom):
            attempts += 1
            continue
        outfit = f"{top} | {bottom}"
        attempts += 1
        if outfit in seen:
            continue
        seen.add(outfit)
        score = SYNTHETIC_TOP_WEIGHTS[top] * SYNTHETIC_BOTTOM_WEIGHTS[bottom]
        results.append({"outfit": outfit, "probability": float(score)})

    total = sum(item["probability"] for item in results) or 1.0
    for item in results:
        item["probability"] = float(item["probability"] / total)

    results.sort(key=lambda item: item["probability"], reverse=True)
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    model_bundle = get_model_bundle()
    error_message = _model_error
    predictions = None
    selected_day = None
    top_k = 3
    used_features = None

    if request.method == "POST":
        selected_day = request.form.get("day")
        top_k_raw = request.form.get("top_k", "3")
        try:
            top_k = max(1, min(10, int(top_k_raw)))
        except ValueError:
            top_k = 3

        if not selected_day:
            error_message = "Select a weekday for Advik."
        elif selected_day not in WEEKDAYS:
            error_message = "Only weekdays are allowed for Advik."

        if error_message:
            pass
        elif model_bundle is None:
            error_message = error_message or "Model is not available."
        else:
            predictions = predict_outfits(model_bundle, selected_day, top_k)
            used_features = {"Day": selected_day}

    return render_template(
        "index.html",
        days=WEEKDAYS,
        selected_day=selected_day,
        predictions=predictions,
        top_k=top_k,
        used_features=used_features,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
