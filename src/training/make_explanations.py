import pandas as pd
import json

# Load the cleaned German dataset
df = pd.read_csv("data/interim/german_credit.csv")

# Function to generate a simple rule-based explanation
def generate_explanation(row):
    reasons = []
    if row["savings"] in ["A65", "A61"]:  # low savings
        reasons.append("low savings")
    if row["employment_duration"] in ["A71", "A72"]:  # short employment
        reasons.append("short employment duration")
    if row["amount"] > 5000:
        reasons.append("high loan amount")
    if row["age"] < 25:
        reasons.append("young applicant")

    if not reasons:
        reasons.append("stable financial profile")

    label = "bad" if row["credit_risk"] == 1 else "good"
    explanation = f"Application {'denied' if label=='bad' else 'approved'} due to " + ", ".join(reasons) + "."

    return label, explanation

# Generate JSONL
with open("data/explanations/german_credit_explanations.jsonl", "w") as f:
    for _, row in df.iterrows():
        label, explanation = generate_explanation(row)
        record = {
            "input": f"status={row['status']}, duration={row['duration']}, savings={row['savings']}, employment={row['employment_duration']}, amount={row['amount']}, age={row['age']}",
            "label": label,
            "explanation": explanation
        }
        f.write(json.dumps(record) + "\n")

print("Dataset saved to data/explanations/german_credit_explanations.jsonl")