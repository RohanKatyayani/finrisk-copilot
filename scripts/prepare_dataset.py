import json
import os

# Paths
input_path = "data/explanations/german_credit_explanations.jsonl"
output_path = "data/credit_instructions.jsonl"

def convert_to_instruction_format(input_path, output_path):
    """Convert credit explanation dataset into Alpaca-style instruction format."""
    samples = []
    with open(input_path, "r") as f:
        for line in f:
            record = json.loads(line)

            instruction = "Explain the credit risk decision for the following applicant profile."
            input_text = record["input"]
            output_text = f"Decision: {record['label']}. Explanation: {record['explanation']}"

            samples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })

    # Save new dataset
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Converted {len(samples)} records and saved to {output_path}")


if __name__ == "__main__":
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    convert_to_instruction_format(input_path, output_path)