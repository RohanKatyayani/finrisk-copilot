"""
src/training/train_model.py

Train a LightGBM pipeline, log experiment with MLflow, save pipeline to models/credit_risk_model.pkl.

Usage:
    python src/training/train_model.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def load_data(path="data/interim/german_credit.csv", target_candidates=None):
    """
    Load dataset and find target column.
    Default path is data/interim/german_credit.csv (adjust if your cleaned CSV is somewhere else).
    """
    if target_candidates is None:
        target_candidates = ["credit_risk", "target", "default"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Put your cleaned CSV there or change the path.")

    df = pd.read_csv(path)
    # find the target column (common names)
    for t in target_candidates:
        if t in df.columns:
            return df, t

    raise KeyError(f"Couldn't find target column among {target_candidates}. Columns in file: {df.columns.tolist()}")


def build_preprocessor(X):
    """
    Build ColumnTransformer with OneHotEncoder for categoricals and StandardScaler for numeric.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # OneHotEncoder: handle_unknown='ignore' so unseen categories don't break inference
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_cols),
            ("num", num_transformer, numeric_cols),
        ],
        remainder="drop",  # drop anything else
        sparse_threshold=0,
    )

    return preprocessor, categorical_cols, numeric_cols


def main():
    # 1) Load data
    df, target_col = load_data()
    print(f"Loaded data: {len(df)} rows. Using target column: '{target_col}'")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2) Train/test split (stratify to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 3) Preprocessor
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    print("Categorical cols:", cat_cols)
    print("Numerical cols:", num_cols)

    # 4) Model (use tuned params you had from experiments)
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

    # 5) Pipeline: preprocessor + model
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # 6) MLflow: start run and log everything
    mlflow.set_experiment("credit_risk_experiment")  # creates or uses experiment
    with mlflow.start_run(run_name="lgbm_credit_risk_run"):
        # Log model hyperparameters (from the classifier)
        mlflow.log_params(pipeline.named_steps["model"].get_params())

        # Fit
        pipeline.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics
        mlflow.log_metric("accuracy", float(acc))
        # log per-class precision/recall if present
        for cls in report:
            if cls in ("0", "1"):
                mlflow.log_metric(f"precision_class_{cls}", float(report[cls]["precision"]))
                mlflow.log_metric(f"recall_class_{cls}", float(report[cls]["recall"]))
                mlflow.log_metric(f"f1_class_{cls}", float(report[cls]["f1-score"]))

        # Log some artifacts: a small text summary
        summary = {
            "accuracy": acc,
            "classification_report": report,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        # Save summary locally and log as artifact
        os.makedirs("artifacts", exist_ok=True)
        summary_path = os.path.join("artifacts", "summary.txt")
        with open(summary_path, "w") as f:
            f.write(str(summary))

        mlflow.log_artifact(summary_path, artifact_path="training_summary")

        # Log the entire sklearn Pipeline (recommended)
        mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")

        # Also save pipeline to models/ for API use (joblib)
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/credit_risk_model.pkl")

        print("MLflow run finished. Run ID:", mlflow.active_run().info.run_id)

    print("✅ Training complete. Pipeline saved to models/credit_risk_model.pkl")


if __name__ == "__main__":
    main()