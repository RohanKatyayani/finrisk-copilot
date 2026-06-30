"""
monitoring/jobs/compute_drift.py

Detects data drift in the German Credit feature distributions by comparing
the training data (reference) against a simulated "current production"
dataset. Produces an HTML report at monitoring/drift_report.html.

In a real bank, the `current` DataFrame would be loaded from production
prediction logs (e.g., last 24h of requests). For this portfolio project
we synthesize realistic drift so the report demonstrates detection
working end-to-end.

Run:
    python -m monitoring.jobs.compute_drift
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/interim/german_credit.csv")
REPORT_PATH = Path("monitoring/drift_report.html")

# Drift simulation knobs
RANDOM_SEED = 42
CURRENT_SAMPLE_SIZE = 500  # larger sample reduces categorical noise
AMOUNT_SHIFT_PCT = 1.50  # 150% upward shift on loan amounts (severe)
DURATION_SHIFT_MO = 18  # +18 months on loan durations (severe)
AGE_NOISE_SD = 8.0  # heavier age noise
PURPOSE_DRIFT_PROB = 0.60  # 60% chance to flip purpose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_reference() -> pd.DataFrame:
    """The training data IS the reference distribution."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Drop the target — drift detection is on features only
    return df.drop(columns=["credit_risk"]) if "credit_risk" in df.columns else df


def simulate_current(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Take a sample of reference and apply controlled distribution shifts so the
    drift report has interesting findings. This stands in for real production
    traffic logs.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    cur = reference.sample(n=CURRENT_SAMPLE_SIZE, random_state=RANDOM_SEED).copy()

    # 1. Loan amounts shifted up by 40% (e.g., post-inflation environment)
    if "amount" in cur.columns:
        cur["amount"] = (cur["amount"] * (1 + AMOUNT_SHIFT_PCT)).astype(int)

    # 2. Loan durations longer (e.g., bank pushing longer-term products)
    if "duration" in cur.columns:
        cur["duration"] = cur["duration"] + DURATION_SHIFT_MO

    # 3. Age noisier (e.g., wider customer demographic)
    if "age" in cur.columns:
        cur["age"] = (
            (cur["age"] + rng.normal(0, AGE_NOISE_SD, size=len(cur))).clip(18, 90).astype(int)
        )

    # 4. Purpose category drift (some applicants now have shifted preferences)
    if "purpose" in cur.columns:
        purposes = cur["purpose"].unique().tolist()
        flip_mask = rng.random(len(cur)) < PURPOSE_DRIFT_PROB
        cur.loc[flip_mask, "purpose"] = rng.choice(purposes, size=flip_mask.sum())

    return cur.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference data: {DATA_PATH}")
    reference = load_reference()
    print(f"  Reference shape: {reference.shape}")

    print("Simulating current production data with controlled drift...")
    current = simulate_current(reference)
    print(f"  Current shape:   {current.shape}")
    print(
        f"  Drift applied:   +{int(AMOUNT_SHIFT_PCT*100)}% amount, "
        f"+{DURATION_SHIFT_MO}mo duration, "
        f"~N(0,{AGE_NOISE_SD}) on age, "
        f"{int(PURPOSE_DRIFT_PROB*100)}% purpose flip"
    )

    print("\nRunning Evidently DataDriftPreset...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)

    snapshot.save_html(str(REPORT_PATH))
    print(f"\n✅ Drift report saved: {REPORT_PATH}")
    print(f"   Open in browser:   open {REPORT_PATH}")


if __name__ == "__main__":
    run()
