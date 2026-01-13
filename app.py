from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from models import load_model

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "advik-secret-key")

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
CSV_PATH = os.environ.get("CSV_PATH", "outfits.csv")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "admin_config.json")
FORCE_SYNTHETIC_ENV = os.environ.get("FORCE_SYNTHETIC")
ADMIN_USER = os.environ.get("ADMIN_USER", "advik67")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "advik67")

DEFAULT_TOP_WEIGHTS = {
    "Black QZ": 0.2,
    "Black Hoodie": 0.2,
    "Camo Hoodie": 0.15,
    "Cape Cod": 0.15,
    "Tow Truck": 0.15,
    "Columbia Puffer": 0.14,
    "Blue Patagonia Pullover": 0.01,
}

DEFAULT_BOTTOM_WEIGHTS = {
    "Grey Sweats": 0.45,
    "Black Sweats": 0.44,
    "Tan Sweats": 0.10,
    "Navy Pants": 0.01,
}

_model_bundle: Optional[Dict[str, object]] = None
_model_error: Optional[str] = None


def _default_config() -> Dict[str, object]:
    return {
        "force_synthetic": True,
        "top_weights": DEFAULT_TOP_WEIGHTS,
        "bottom_weights": DEFAULT_BOTTOM_WEIGHTS,
    }


def _load_config() -> Dict[str, object]:
    config = _default_config()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                config.update(stored)
        except Exception:
            pass
    if FORCE_SYNTHETIC_ENV is not None:
        config["force_synthetic"] = FORCE_SYNTHETIC_ENV == "1"
    return config


def _save_config(config: Dict[str, object]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def _detect_csv_layout(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError("CSV file is empty.")
        skip_first = first_line.upper().startswith("POSSIBLE TOPS")
        header_line = handle.readline() if skip_first else first_line

    delimiter = "\t" if "\t" in header_line else ","
    headers = [col.strip() for col in header_line.rstrip("\n").split(delimiter)]
    return {
        "delimiter": delimiter,
        "headers": headers,
        "skip_first": skip_first,
        "first_line": first_line.rstrip("\n"),
    }


def _read_csv_dataframe() -> pd.DataFrame:
    layout = _detect_csv_layout(CSV_PATH)
    delimiter = layout["delimiter"]
    header = 1 if layout["skip_first"] else 0
    return pd.read_csv(CSV_PATH, sep=delimiter, engine="python", header=header)


def _write_csv_dataframe(df: pd.DataFrame, layout: Dict[str, object]) -> None:
    delimiter = layout["delimiter"]
    skip_first = layout["skip_first"]
    first_line = layout.get("first_line")
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as handle:
        if skip_first and first_line:
            handle.write(first_line + "\n")
        df.to_csv(handle, index=False, sep=delimiter)


def _append_outfit_entry(date_str: str, top: str, bottom: str) -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    parsed = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Invalid date.")

    day_name = parsed.day_name()
    layout = _detect_csv_layout(CSV_PATH)
    delimiter = layout["delimiter"]
    headers = layout["headers"]

    row = [""] * len(headers)
    for index, header in enumerate(headers):
        if header == "Date":
            row[index] = date_str
        elif header in {"Day of the Week", "Day"}:
            row[index] = day_name
        elif header in {"Hoodie/Jacket", "Top"}:
            row[index] = top
        elif header in {"Pants/Sweats Color", "Bottom"}:
            row[index] = bottom

    with open(CSV_PATH, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(row)


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


def _history_based_outfits(
    label_counts: Dict[str, int],
    top_k: int,
    last_outfit: Optional[str],
) -> List[Dict[str, object]]:
    if not label_counts:
        return []
    ranked_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    total = sum(label_counts.values()) or 1
    ranked = [(outfit, count / total) for outfit, count in ranked_counts]
    return _select_top_k(ranked, top_k, last_outfit)


def _get_last_outfit_label() -> Optional[str]:
    if not os.path.exists(CSV_PATH):
        return None

    try:
        df = _read_csv_dataframe()
    except Exception:
        return None

    if df.empty:
        return None

    rename_map = {
        "Day of the Week": "Day",
        "Hoodie/Jacket": "Top",
        "Pants/Sweats Color": "Bottom",
    }
    df = df.rename(columns=rename_map)
    df.columns = [str(col).strip() for col in df.columns]

    for _, row in df.tail(10).iloc[::-1].iterrows():
        if "Outfit" in df.columns:
            outfit = str(row.get("Outfit", "")).strip()
            if outfit:
                return outfit
        if "Top" in df.columns and "Bottom" in df.columns:
            top = str(row.get("Top", "")).strip()
            bottom = str(row.get("Bottom", "")).strip()
            if top and bottom:
                return f"{top} | {bottom}"
    return None


def _select_top_k(
    ranked: List[tuple],
    top_k: int,
    last_outfit: Optional[str],
) -> List[Dict[str, object]]:
    results = []
    for outfit, score in ranked:
        if last_outfit and outfit == last_outfit:
            continue
        results.append({"outfit": outfit, "probability": float(score)})
        if len(results) >= top_k:
            break
    return results


def predict_outfits(
    model_bundle: Dict[str, object], day: str, top_k: int
) -> List[Dict[str, object]]:
    label_counts = _get_label_counts(model_bundle)
    last_outfit = _get_last_outfit_label()
    config = _load_config()
    force_synthetic = bool(config.get("force_synthetic", True))
    if force_synthetic and label_counts:
        return _history_based_outfits(label_counts, top_k, last_outfit)
    if force_synthetic:
        return generate_synthetic_outfits(top_k, last_outfit)

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
    return _select_top_k(ranked, top_k, last_outfit)


def generate_synthetic_outfits(
    top_k: int, last_outfit: Optional[str]
) -> List[Dict[str, object]]:
    config = _load_config()
    top_weights_map = config.get("top_weights", DEFAULT_TOP_WEIGHTS)
    bottom_weights_map = config.get("bottom_weights", DEFAULT_BOTTOM_WEIGHTS)
    tops = list(top_weights_map.keys())
    top_weights = list(top_weights_map.values())
    bottoms = list(bottom_weights_map.keys())
    bottom_weights = list(bottom_weights_map.values())

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
        if outfit == last_outfit or outfit in seen:
            continue
        seen.add(outfit)
        score = top_weights_map[top] * bottom_weights_map[bottom]
        results.append({"outfit": outfit, "probability": float(score)})

    total = sum(item["probability"] for item in results) or 1.0
    for item in results:
        item["probability"] = float(item["probability"] / total)

    results.sort(key=lambda item: item["probability"], reverse=True)
    return results


def _parse_weight_lines(raw_text: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            name, weight = stripped.split(":", 1)
        elif "," in stripped:
            name, weight = stripped.split(",", 1)
        else:
            continue
        name = name.strip()
        try:
            value = float(weight.strip())
        except ValueError:
            continue
        if name:
            weights[name] = value
    return weights


def _require_admin() -> bool:
    return bool(session.get("admin"))


def _retrain_model() -> None:
    command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "train_model.py"),
        "--csv",
        CSV_PATH,
        "--out",
        MODEL_PATH,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Training failed.")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    error_message = None
    status_message = None

    if request.method == "POST":
        action = request.form.get("action", "login")
        if action == "login":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            if username == ADMIN_USER and password == ADMIN_PASSWORD:
                session["admin"] = True
                return redirect(url_for("admin"))
            error_message = "Invalid credentials."
        elif not _require_admin():
            error_message = "Login required."
        elif action == "logout":
            session.pop("admin", None)
            return redirect(url_for("admin"))
        elif action == "retrain":
            try:
                _retrain_model()
                status_message = "Model retrained."
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
        elif action == "toggle_mode":
            config = _load_config()
            config["force_synthetic"] = request.form.get("force_synthetic") == "1"
            _save_config(config)
            status_message = "Mode updated."
        elif action == "update_weights":
            config = _load_config()
            top_text = request.form.get("top_weights", "")
            bottom_text = request.form.get("bottom_weights", "")
            top_weights = _parse_weight_lines(top_text)
            bottom_weights = _parse_weight_lines(bottom_text)
            if top_weights:
                config["top_weights"] = top_weights
            if bottom_weights:
                config["bottom_weights"] = bottom_weights
            _save_config(config)
            status_message = "Weights updated."
        elif action == "upload_csv":
            upload = request.files.get("csv_file")
            if not upload or upload.filename == "":
                error_message = "Choose a CSV file."
            else:
                filename = secure_filename(upload.filename)
                if not filename.lower().endswith(".csv"):
                    error_message = "Only .csv files are allowed."
                else:
                    upload.save(CSV_PATH)
                    status_message = "CSV uploaded."
        elif action == "delete_rows":
            layout = _detect_csv_layout(CSV_PATH)
            df = _read_csv_dataframe()
            indices = request.form.getlist("row_index")
            drop_indices = []
            for raw in indices:
                try:
                    drop_indices.append(int(raw))
                except ValueError:
                    continue
            if drop_indices:
                df = df.drop(index=drop_indices, errors="ignore")
                _write_csv_dataframe(df, layout)
                status_message = "Rows deleted."
        elif action == "clear_csv":
            layout = _detect_csv_layout(CSV_PATH)
            df = _read_csv_dataframe()
            df = df.iloc[0:0]
            _write_csv_dataframe(df, layout)
            status_message = "CSV cleared."
        elif action == "reset_model":
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            status_message = "Model removed."

    config = _load_config()
    recent_rows = []
    if _require_admin() and os.path.exists(CSV_PATH):
        try:
            df = _read_csv_dataframe()
            recent_rows = df.tail(10).to_dict(orient="records")
            recent_rows = list(enumerate(recent_rows))
        except Exception:
            recent_rows = []

    def _format_weights(weights: Dict[str, float]) -> str:
        return "\n".join(f"{name}: {value}" for name, value in weights.items())

    return render_template(
        "admin.html",
        is_admin=_require_admin(),
        error_message=error_message,
        status_message=status_message,
        config=config,
        top_weights_text=_format_weights(config.get("top_weights", DEFAULT_TOP_WEIGHTS)),
        bottom_weights_text=_format_weights(
            config.get("bottom_weights", DEFAULT_BOTTOM_WEIGHTS)
        ),
        recent_rows=recent_rows,
    )


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin"))


@app.route("/", methods=["GET", "POST"])
def index():
    model_bundle = get_model_bundle()
    error_message = _model_error
    status_message = None
    predictions = None
    selected_day = None
    top_k = 3
    used_features = None
    today_value = date.today().isoformat()

    if request.method == "POST":
        action = request.form.get("action", "predict")
        if action == "log_outfit":
            entry_date = request.form.get("entry_date", "").strip()
            entry_top = request.form.get("entry_top", "").strip()
            entry_bottom = request.form.get("entry_bottom", "").strip()
            if not entry_date or not entry_top or not entry_bottom:
                error_message = "Provide date, top, and bottom."
            else:
                try:
                    _append_outfit_entry(entry_date, entry_top, entry_bottom)
                    status_message = "Saved today's outfit. Retrain to update predictions."
                except Exception as exc:  # noqa: BLE001
                    error_message = str(exc)
        else:
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
        today_value=today_value,
        status_message=status_message,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
