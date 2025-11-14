from pathlib import Path

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "marks_prediction_model.pkl"
DATASET_PATH = BASE_DIR / "students_dataset.csv"

# Define the order of features expected by the model.
FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "internet_access",
    "play_hours",
    "assignments_completed",
    "sleep_hours",
]


def train_and_save_model() -> Pipeline:
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["rollno", "student_id", "name"])
    df["internet_access"] = df["internet_access"].map({"Yes": 1, "No": 0})

    X = df[FEATURE_COLUMNS]
    y = df["marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.5)),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    # Persist the trained model so subsequent runs skip training.
    joblib.dump(model_pipeline, MODEL_PATH)

    return model_pipeline


def load_or_train_model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Required dataset not found at {DATASET_PATH}. "
            "Place students_dataset.csv next to app.py."
        )

    return train_and_save_model()


# Load the trained model once when the app starts.
model = load_or_train_model()


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_marks = None
    error_message = None
    status = None

    if request.method == "POST":
        try:
            # Collect form data and ensure the correct types.
            form_values = {
                "study_hours": float(request.form.get("study_hours", 0)),
                "attendance": float(request.form.get("attendance", 0)),
                "internet_access": 1
                if request.form.get("internet_access") == "yes"
                else 0,
                "play_hours": float(request.form.get("play_hours", 0)),
                "assignments_completed": float(
                    request.form.get("assignments_completed", 0)
                ),
                "sleep_hours": float(request.form.get("sleep_hours", 0)),
            }

            # Validate ranges
            if not 1 <= form_values["study_hours"] <= 10:
                raise ValueError("Study hours must be between 1 and 10.")
            if not 0 <= form_values["attendance"] <= 100:
                raise ValueError("Attendance must be between 0 and 100.")
            if not 0 <= form_values["play_hours"] <= 5:
                raise ValueError("Play hours must be between 0 and 5.")
            if not 0 <= form_values["assignments_completed"] <= 20:
                raise ValueError("Assignments completed must be between 0 and 20.")
            if not 1 <= form_values["sleep_hours"] <= 10:
                raise ValueError("Sleep hours must be between 1 and 10.")

            # Arrange data in a DataFrame aligned with the model's expectations.
            new_data = pd.DataFrame([form_values], columns=FEATURE_COLUMNS)

            prediction = model.predict(new_data)
            predicted_marks = round(float(prediction[0]), 2)
            status = "Pass" if predicted_marks >= 33 else "Fail"
        except ValueError:
            error_message = "Please provide valid numeric values for all inputs."
        except Exception as exc:  # pragma: no cover
            error_message = f"Unexpected error: {exc}"

    return render_template(
        "index.html",
        predicted_marks=predicted_marks,
        status=status,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)

