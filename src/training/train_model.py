"""
src/training/train_model.py

Train a LightGBM credit-risk pipeline, log experiment to MLflow,
register the trained model in the MLflow Model Registry as
`credit_risk_model`, and also save a local .pkl for the FastAPI service.

Usage:
    python src/training/train_model.py
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Use SQLite as the tracking + registry backend (filesystem store is deprecated
# in MLflow 3.x and doesn't support the Model Registry).
mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "credit_risk_experiment"
REGISTERED_MODEL_NAME = "credit_risk_model"


def load_data(path="data/interim/german_credit.csv", target_candidates=None):
    """Load German Credit dataset and identify target column."""
    if target_candidates is None:
        target_candidates = ["credit_risk", "target", "default"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}.")

    df = pd.read_csv(path)
    for t in target_candidates:
        if t in df.columns:
            return df, t

    raise KeyError(f"Target column not found in {df.columns.tolist()}")


def build_preprocessor(X):
    """OneHotEncoder for categoricals + StandardScaler for numerics."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return preprocessor, categorical_cols, numeric_cols


def main():
    # 1. Data
    df, target_col = load_data()
    print(f"Loaded data: {len(df)} rows. Target: '{target_col}'")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Pipeline
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    model = LGBMClassifier(
        learning_rate=0.01,
        max_depth=10,
        num_leaves=30,
        n_estimators=200,
        subsample=1.0,
        colsample_bytree=0.8,
        scale_pos_weight=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # 3. Train + log + register
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="lgbm_credit_risk_run") as run:
        mlflow.log_params(pipeline.named_steps["model"].get_params())

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("accuracy", float(acc))
        for cls in ("0", "1"):
            if cls in report:
                mlflow.log_metric(f"precision_class_{cls}", float(report[cls]["precision"]))
                mlflow.log_metric(f"recall_class_{cls}", float(report[cls]["recall"]))
                mlflow.log_metric(f"f1_class_{cls}", float(report[cls]["f1-score"]))

        # Log + register the model in one call
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model_pipeline",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )

        # Save a local .pkl too, for the API and Docker image to load
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/credit_risk_model.pkl")

        run_id = run.info.run_id
        print(f"\n📊 Metrics — accuracy: {acc:.3f}")
        print(f"📌 MLflow run ID: {run_id}")
        print(f"🏷️  Registered model: {REGISTERED_MODEL_NAME} (check Registry for version)")
        print("💾 Local pickle:  models/credit_risk_model.pkl")
        print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
