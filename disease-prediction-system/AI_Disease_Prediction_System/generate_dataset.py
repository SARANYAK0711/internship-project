"""
generate_dataset.py
-------------------
Utility script that programmatically creates `disease_dataset.csv`.

The dataset uses a binary symptom matrix:
    - Each column (except the last) is a symptom (1 = present, 0 = absent).
    - The last column `disease` is the target label.

For every disease we define a "typical" symptom profile. We then generate
several noisy samples per disease so that the Decision Tree has enough
variation to learn meaningful decision boundaries instead of memorising a
single row per class.

Run this file only if you want to regenerate the dataset:
    python generate_dataset.py
"""

import csv
import os
import random

# Fixed seed => reproducible dataset every time the script is run.
random.seed(42)

# ---------------------------------------------------------------------------
# 1. Define the 20 symptoms (input features)
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "fever",
    "cough",
    "fatigue",
    "headache",
    "sore_throat",
    "runny_nose",
    "body_ache",
    "chills",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal_pain",
    "loss_of_taste",
    "shortness_of_breath",
    "joint_pain",
    "skin_rash",
    "dizziness",
    "chest_pain",
    "sneezing",
    "sweating",
]

# ---------------------------------------------------------------------------
# 2. Define the 10 diseases with their characteristic symptoms
# ---------------------------------------------------------------------------
DISEASE_PROFILES = {
    "Common Cold": ["cough", "sore_throat", "runny_nose", "sneezing", "headache"],
    "Influenza (Flu)": ["fever", "cough", "fatigue", "body_ache", "chills", "headache"],
    "COVID-19": ["fever", "cough", "fatigue", "loss_of_taste", "shortness_of_breath", "sore_throat"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea", "fatigue"],
    "Dengue": ["fever", "headache", "joint_pain", "skin_rash", "body_ache", "nausea"],
    "Typhoid": ["fever", "abdominal_pain", "fatigue", "headache", "diarrhea", "nausea"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever"],
    "Migraine": ["headache", "nausea", "dizziness", "fatigue"],
    "Pneumonia": ["fever", "cough", "shortness_of_breath", "chest_pain", "chills", "fatigue"],
    "Food Poisoning": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "sweating"],
}

# Number of noisy samples to generate per disease.
SAMPLES_PER_DISEASE = 60

# Probabilities that control the noise in the generated data.
P_CORE_PRESENT = 0.95   # a core symptom is almost always present
P_NOISE_PRESENT = 0.03  # a non-core symptom rarely appears


def build_rows():
    """Create the list of data rows (each row is a dict)."""
    rows = []
    for disease, core_symptoms in DISEASE_PROFILES.items():
        for _ in range(SAMPLES_PER_DISEASE):
            row = {}
            for symptom in SYMPTOMS:
                if symptom in core_symptoms:
                    row[symptom] = 1 if random.random() < P_CORE_PRESENT else 0
                else:
                    row[symptom] = 1 if random.random() < P_NOISE_PRESENT else 0
            # Safety net: ensure at least one core symptom is present so the
            # row is never an "all zeros" ambiguous sample.
            if sum(row[s] for s in core_symptoms) == 0:
                row[random.choice(core_symptoms)] = 1
            row["disease"] = disease
            rows.append(row)
    random.shuffle(rows)
    return rows


def main():
    output_path = os.path.join(os.path.dirname(__file__), "disease_dataset.csv")
    rows = build_rows()
    fieldnames = SYMPTOMS + ["disease"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset created: {output_path}")
    print(f"Total rows: {len(rows)}  |  Symptoms: {len(SYMPTOMS)}  |  Diseases: {len(DISEASE_PROFILES)}")


if __name__ == "__main__":
    main()
