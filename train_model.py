"""Train an outfit prediction model from the Advik CSV.

Expected CSV columns (best-effort):
- Date (optional)
- Day (optional) : e.g., Monday
- Top (optional)
- Bottom (optional)
- Outfit (optional) : if present, should be a string label like "Top | Bottom" or any consistent label

The trainer will infer a target label column in this priority order:
1) Outfit
2) (Top + Bottom) combined into "Top | Bottom"

Features:
- Day of week (from Day column or inferred from Date)
- (Optional) other simple categorical columns if present

Saves a scikit-learn Pipeline to model.joblib.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names for robustness
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def replace_csv(
    new_csv_path: str, target_path: str = "outfits.csv", backup: bool = True
) -> Path:
    """Copy a new CSV into place (optionally backing up the old file)."""
    src = Path(new_csv_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}")

    dest = Path(target_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if backup and dest.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = dest.with_suffix(f"{dest.suffix}.bak.{timestamp}")
        shutil.copy2(dest, backup_path)

    shutil.copy2(src, dest)
    return dest


def _infer_day(df: pd.DataFrame) -> pd.Series:
    # Prefer explicit Day column
    if "Day" in df.columns:
        return df["Day"].astype(str).str.strip()

    # Try to infer from Date column
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
        return dt.dt.day_name().fillna("Unknown")

    return pd.Series(["Unknown"] * len(df))


def _infer_target(df: pd.DataFrame) -> pd.Series:
    # 1) Use Outfit if present
    if "Outfit" in df.columns:
        outfit = df["Outfit"].astype(str).str.strip()
        outfit = outfit.replace({"nan": None, "None": None, "": None, "Absent": None, "absent": None})
        return outfit

    # 2) Combine Top and Bottom if present
    if "Top" in df.columns and "Bottom" in df.columns:
        top = (
            df["Top"]
            .astype(str)
            .str.strip()
            .replace({"nan": None, "None": None, "": None, "Absent": None, "absent": None})
        )
        bottom = (
            df["Bottom"]
            .astype(str)
            .str.strip()
            .replace({"nan": None, "None": None, "": None, "Absent": None, "absent": None})
        )
        combined = top + " | " + bottom
        combined = combined.where(top.notna() & bottom.notna())
        return combined.str.strip()

    raise ValueError(
        "Could not infer target label. Provide an 'Outfit' column, or both 'Top' and 'Bottom' columns."
    )


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Build a minimal, robust set of categorical features
    features = pd.DataFrame(index=df.index)
    features["Day"] = _infer_day(df)

    # Add other simple categorical columns if they exist (excluding target-like columns)
    exclude = {"Outfit", "Top", "Bottom"}
    for col in df.columns:
        if col in exclude:
            continue
        if "prediction" in str(col).lower():
            continue
        if col == "Day":
            continue
        if col == "Date":
            # Date is converted to Day already
            continue

        # Keep only low-cardinality-ish object columns as categoricals
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            features[col] = df[col].astype(str).str.strip()

    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="model.joblib", help="Output model path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")

    if len(df.columns) == 1 and "\t" in df.columns[0]:
        df = pd.read_csv(csv_path, sep="\t", engine="python")

    if "Date" not in df.columns and any(
        str(col).upper().startswith("POSSIBLE TOPS") for col in df.columns
    ):
        df = pd.read_csv(csv_path, sep="\t", engine="python", header=1)

    rename_map = {
        "Day of the Week": "Day",
        "Hoodie/Jacket": "Top",
        "Pants/Sweats Color": "Bottom",
    }
    df = df.rename(columns=rename_map)
    df = _normalize_columns(df)

    y = _infer_target(df)
    # Drop rows with missing/empty target
    y = y.replace({"nan": None, "None": None, "": None, "Absent": None, "absent": None})
    mask = y.notna()
    mask &= ~y.astype(str).str.lower().str.contains("absent")
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        y = y.loc[df.index].reset_index(drop=True)
        non_repeat_mask = y.ne(y.shift(1))
        df = df.loc[non_repeat_mask].reset_index(drop=True)
        y = y.loc[non_repeat_mask].reset_index(drop=True)

    X = _build_feature_frame(df)

    # Categorical pipeline
    cat_cols: List[str] = list(X.columns)
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, multi_class="auto")

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])
    model.fit(X, y)

    label_counts = y.value_counts().to_dict()
    joblib.dump(
        {
            "model": model,
            "feature_columns": cat_cols,
            "label_example": str(y.iloc[0]) if len(y) else "",
            "label_counts": label_counts,
        },
        args.out,
    )

    print(f"✅ Trained on {len(df)} rows")
    print(f"✅ Saved model to {args.out}")


if __name__ == "__main__":
    main()
