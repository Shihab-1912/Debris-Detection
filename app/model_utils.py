from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "space_debris_uncleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "space_debris_model.joblib"

DROP_COLUMNS = [
    "ID",
    "Name",
    "Satellite_ID",
    "Launch_Date",
    "Last_Updated",
    "Risk_Level",
    "Notes",
    "Extra_Junk",
]

NUMERIC_INPUTS = [
    "Altitude_km",
    "Inclination_deg",
    "Eccentricity",
    "Perigee_km",
    "Apogee_km",
    "Size_m",
    "Radar_Cross_Section_m2",
    "Collision_Probability",
]

CATEGORICAL_INPUTS = [
    "Operator",
    "Country",
    "Status",
    "Material",
    "Debris_Type",
    "Tracked",
]

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        df[col] = df[col].fillna(fill_value).astype(str).str.lower().str.strip()
        df[col] = df[col].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Perigee_Apogee_Diff"] = df["Apogee_km"] - df["Perigee_km"]
    df["Orbit_Stability"] = df["Eccentricity"] * df["Inclination_deg"]
    df["Risk_Score"] = df["Collision_Probability"] * df["Radar_Cross_Section_m2"]
    df["Altitude_Risk"] = df["Altitude_km"] * df["Collision_Probability"]
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    score = (
        df["Collision_Probability"] * 0.6
        + df["Radar_Cross_Section_m2"] * 0.2
        + df["Altitude_km"] * 0.0001
        + df["Eccentricity"] * 0.1
    )
    return pd.qcut(score, q=3, labels=[0, 1, 2]).astype(int)


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(clean_data(df))
    y = build_target(df)
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    X = X.drop(columns=["Risk_Level_Final"], errors="ignore")
    return X, y


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_save_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> dict:
    raw_df = load_raw_data(data_path)
    X, y = prepare_training_frame(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = make_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted")),
        "recall": float(recall_score(y_test, predictions, average="weighted")),
        "f1": float(f1_score(y_test, predictions, average="weighted")),
    }

    artifact = {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "numeric_inputs": NUMERIC_INPUTS,
        "categorical_inputs": CATEGORICAL_INPUTS,
        "label_map": LABEL_MAP,
        "metrics": metrics,
        "training_rows": int(len(X)),
        "class_distribution": y.value_counts().sort_index().to_dict(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    return artifact


def load_model(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        return train_and_save_model(model_path=model_path)
    return joblib.load(model_path)


def build_prediction_frame(values: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {column: values.get(column) for column in feature_columns}
    frame = pd.DataFrame([row])
    return add_features(frame)
