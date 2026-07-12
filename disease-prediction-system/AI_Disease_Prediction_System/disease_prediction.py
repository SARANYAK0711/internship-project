"""
AI-Based Disease Prediction System Using Machine Learning
=========================================================

This program predicts the most likely disease based on the symptoms
entered by the user. It uses a Decision Tree Classifier trained on a
symptom-disease dataset.

Workflow:
    1. Load the dataset (disease_dataset.csv).
    2. Preprocess the data (features / target split).
    3. Train a Decision Tree Classifier.
    4. Accept symptoms from the user.
    5. Convert the input into the model's feature format.
    6. Predict the disease.
    7. Display the prediction in a user-friendly format.

Author : Your Name
Course : Machine Learning Major Project
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# joblib is optional – used to save / load the trained model.
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "disease_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.joblib")
TARGET_COLUMN = "disease"


# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
def load_dataset(path=DATASET_PATH):
    """Load the symptom-disease dataset from a CSV file.

    Returns a pandas DataFrame. Exits gracefully if the file is missing.
    """
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        print("        Run 'python generate_dataset.py' to create it first.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocess the data
# ---------------------------------------------------------------------------
def preprocess_data(df):
    """Split the DataFrame into feature matrix X and target vector y.

    Also performs basic cleaning:
        - Drops fully empty rows.
        - Fills any missing symptom values with 0 (symptom absent).
    """
    # Basic cleaning.
    df = df.dropna(how="all")                       # remove fully empty rows
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    df[feature_columns] = df[feature_columns].fillna(0).astype(int)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print(f"[INFO] Preprocessing complete: {len(feature_columns)} symptom features.")
    return X, y, feature_columns


# ---------------------------------------------------------------------------
# 3. Train the Decision Tree model
# ---------------------------------------------------------------------------
def train_model(X, y):
    """Train a Decision Tree Classifier and report its accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[INFO] Model trained. Test accuracy: {accuracy * 100:.2f}%")

    # Optionally persist the model for reuse.
    if JOBLIB_AVAILABLE:
        joblib.dump(model, MODEL_PATH)
        print(f"[INFO] Trained model saved to: {MODEL_PATH}")

    return model


def get_model(X, y):
    """Load a saved model if available, otherwise train a fresh one."""
    if JOBLIB_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("[INFO] Loaded existing trained model from disk.")
            return model
        except Exception:
            print("[WARN] Could not load saved model. Retraining...")
    return train_model(X, y)


# ---------------------------------------------------------------------------
# 4 & 5. Accept user input and convert it into feature format
# ---------------------------------------------------------------------------
def display_symptom_menu(feature_columns):
    """Print a numbered menu of all available symptoms."""
    print("\nAvailable symptoms:")
    print("-" * 50)
    for idx, symptom in enumerate(feature_columns, start=1):
        # Display symptoms in a clean, readable format.
        pretty = symptom.replace("_", " ").title()
        print(f"  {idx:>2}. {pretty}")
    print("-" * 50)


def parse_user_input(raw_input_str, feature_columns):
    """Convert a comma-separated string of symptom numbers/names into a
    binary feature vector.

    Accepts either symptom numbers (e.g. "1, 4, 7") or symptom names
    (e.g. "fever, cough"). Returns (feature_vector, selected_symptoms).

    Raises ValueError on completely invalid input.
    """
    if not raw_input_str or not raw_input_str.strip():
        raise ValueError("No symptoms were entered.")

    # Build lookup helpers.
    name_to_index = {name.lower(): i for i, name in enumerate(feature_columns)}

    feature_vector = [0] * len(feature_columns)
    selected_symptoms = []
    invalid_tokens = []

    for token in raw_input_str.split(","):
        token = token.strip().lower()
        if not token:
            continue

        matched_index = None
        if token.isdigit():                         # user entered a number
            num = int(token)
            if 1 <= num <= len(feature_columns):
                matched_index = num - 1
        elif token in name_to_index:                # user entered a name
            matched_index = name_to_index[token]
        else:                                       # try name with spaces
            normalized = token.replace(" ", "_")
            if normalized in name_to_index:
                matched_index = name_to_index[normalized]

        if matched_index is None:
            invalid_tokens.append(token)
        else:
            feature_vector[matched_index] = 1
            selected_symptoms.append(feature_columns[matched_index])

    if invalid_tokens:
        print(f"[WARN] Ignored unrecognised entries: {', '.join(invalid_tokens)}")

    if sum(feature_vector) == 0:
        raise ValueError("None of the entered symptoms were valid.")

    return feature_vector, selected_symptoms


# ---------------------------------------------------------------------------
# 6 & 7. Predict the disease and display the result
# ---------------------------------------------------------------------------
def predict_disease(model, feature_vector, feature_columns):
    """Return the predicted disease and per-class probabilities."""
    input_df = pd.DataFrame([feature_vector], columns=feature_columns)
    prediction = model.predict(input_df)[0]

    # Probability estimates give the user extra confidence context.
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return prediction, ranked


def display_result(selected_symptoms, prediction, ranked):
    """Show the prediction in a clear, user-friendly format."""
    pretty_symptoms = [s.replace("_", " ").title() for s in selected_symptoms]

    print("\n" + "=" * 55)
    print("               DISEASE PREDICTION RESULT")
    print("=" * 55)
    print(f"  Symptoms entered : {', '.join(pretty_symptoms)}")
    print(f"  Predicted disease: >>> {prediction} <<<")
    print("-" * 55)
    print("  Top possibilities (confidence):")
    for disease, prob in ranked[:3]:
        if prob > 0:
            bar = "#" * int(prob * 20)
            print(f"    - {disease:<20} {prob * 100:5.1f}%  {bar}")
    print("=" * 55)
    print("  NOTE: This is an ML-based prediction for educational")
    print("        purposes only. Please consult a qualified doctor.")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
def interactive_mode(model, feature_columns):
    """Continuously accept symptoms from the user until they quit."""
    display_symptom_menu(feature_columns)

    while True:
        print("\nEnter symptom numbers or names separated by commas.")
        print("(e.g. '1, 2, 5' or 'fever, cough') | type 'quit' to exit.")
        raw = input("Your symptoms: ").strip()

        if raw.lower() in {"quit", "exit", "q"}:
            print("Goodbye! Stay healthy.")
            break

        try:
            feature_vector, selected = parse_user_input(raw, feature_columns)
            prediction, ranked = predict_disease(model, feature_vector, feature_columns)
            display_result(selected, prediction, ranked)
        except ValueError as err:
            print(f"[ERROR] {err} Please try again.")


def run_demo(model, feature_columns):
    """Run a few sample predictions (useful when no keyboard input is
    available, e.g. automated grading or screenshots)."""
    print("\n[DEMO MODE] Running sample predictions...\n")
    samples = [
        ["fever", "cough", "loss_of_taste", "shortness_of_breath"],
        ["headache", "nausea", "dizziness"],
        ["fever", "chills", "sweating", "headache"],
        ["nausea", "vomiting", "diarrhea", "abdominal_pain"],
    ]
    for symptoms in samples:
        feature_vector = [1 if col in symptoms else 0 for col in feature_columns]
        prediction, ranked = predict_disease(model, feature_vector, feature_columns)
        display_result(symptoms, prediction, ranked)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "*" * 55)
    print("   AI-BASED DISEASE PREDICTION SYSTEM (Decision Tree)")
    print("*" * 55)

    # Steps 1-3: load, preprocess, train.
    df = load_dataset()
    X, y, feature_columns = preprocess_data(df)
    model = get_model(X, y)

    # Steps 4-7: interactive prediction.
    # If '--demo' is passed OR there is no interactive terminal, run the demo.
    if "--demo" in sys.argv or not sys.stdin.isatty():
        run_demo(model, feature_columns)
    else:
        interactive_mode(model, feature_columns)


if __name__ == "__main__":
    main()
