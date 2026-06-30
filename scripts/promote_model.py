"""
scripts/promote_model.py

Promote a registered MLflow model version to a target stage
(Staging or Production), automatically archiving any previous
version in that stage.

Examples:
    python -m scripts.promote_model --version 1 --stage Staging
    python -m scripts.promote_model --version 1 --stage Production

In a real ML platform this would be invoked by CI after validation
checks pass (e.g., accuracy gate, drift gate, fairness audits).
"""

import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "sqlite:///mlflow.db"
MODEL_NAME = "credit_risk_model"
VALID_STAGES = {"None", "Staging", "Production", "Archived"}


def list_versions(client: MlflowClient) -> None:
    """Print all versions and their current stage for visibility."""
    print(f"\nCurrent versions of '{MODEL_NAME}':")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("  (none registered yet)")
        return
    for v in sorted(versions, key=lambda x: int(x.version)):
        marker = "  ◀ promoting" if False else ""
        print(f"  v{v.version}: stage={v.current_stage:<12s} run_id={v.run_id[:8]}{marker}")


def promote(version: str, stage: str, archive_existing: bool = True) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Confirm the model + version exist
    try:
        mv = client.get_model_version(MODEL_NAME, version)
    except Exception as e:
        print(f"❌ Couldn't find {MODEL_NAME} version {version}: {e}")
        sys.exit(1)

    print(f"\nPromoting {MODEL_NAME} v{version} → {stage}")
    print(f"  current stage: {mv.current_stage}")
    print(f"  run_id:        {mv.run_id}")

    list_versions(client)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )

    print(f"\n✅ Done. v{version} is now in stage '{stage}'.")
    if archive_existing and stage in ("Staging", "Production"):
        print(f"   Any prior version in '{stage}' was automatically archived.")

    list_versions(client)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an MLflow model version.")
    parser.add_argument("--version", required=True, help="Model version (e.g., 1)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(VALID_STAGES),
        help="Target stage",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not auto-archive existing versions in the target stage",
    )
    args = parser.parse_args()

    promote(args.version, args.stage, archive_existing=not args.no_archive)


if __name__ == "__main__":
    main()
