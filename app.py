from flask import Flask, render_template, request, send_file

import os
import joblib
import pandas as pd
import numpy as np

from utils.ocr import read_report
from utils.extractor import extract_values
from report_generator import create_report


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)


# ============================================================
# Folders
# ============================================================

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# ============================================================
# Load Trained Model
# ============================================================

model = joblib.load("heart_model.pkl")


# ============================================================
# Load Model Information
# ============================================================

try:
    model_info = joblib.load("heart_model_info.pkl")

except FileNotFoundError:

    print("WARNING: heart_model_info.pkl not found.")

    model_info = {
        "best_algorithm": "Random Forest",
        "best_accuracy_percent": 0.0,
        "algorithm_comparison": []
    }


# ============================================================
# Feature Order
# ============================================================

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope"
]


# ============================================================
# Store Last Generated PDF
# ============================================================

pdf_file_path = None


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# Upload Page
# ============================================================

@app.route("/upload-page")
def upload_page():

    return render_template("upload.html")


# ============================================================
# Helper - Check Valid Extracted Value
# ============================================================

def is_valid_extracted_value(value, key=None):

    if value is None:
        return False

    if isinstance(value, str):

        cleaned = value.strip().lower()

        if cleaned in ["", "none", "not found", "n/a", "na"]:
            return False

    # These values should not normally be zero.
    # Zero means the extractor did not find the value.
    if key in ["trestbps", "chol", "thalach"]:

        try:

            if float(value) == 0:
                return False

        except (ValueError, TypeError):

            return False

    return True


# ============================================================
# Upload Medical Reports
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    reports = request.files.getlist("reports")

    if not reports:

        return """
        <h3>No report uploaded.</h3>
        <a href="/upload-page">Go Back</a>
        """

    merged_data = {}

    # ========================================================
    # Process Multiple Reports
    # ========================================================

    for report in reports:

        if report.filename == "":
            continue

        # ----------------------------------------------------
        # Save Uploaded Report
        # ----------------------------------------------------

        filepath = os.path.join(
            UPLOAD_FOLDER,
            report.filename
        )

        report.save(filepath)

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = read_report(filepath)

        # ----------------------------------------------------
        # Extract Values
        # ----------------------------------------------------

        data = extract_values(text)

        # ----------------------------------------------------
        # Print Individual Extraction
        # ----------------------------------------------------

        print("\n========================================")
        print("EXTRACTED DATA")
        print("========================================")

        for key, value in data.items():

            print(f"{key}: {value}")

        print("========================================\n")

        # ----------------------------------------------------
        # Merge Extracted Data
        #
        # Only valid values are allowed to overwrite previous
        # values. This prevents values such as:
        #
        # None
        # Not Found
        # 0
        #
        # from overwriting valid values from another report.
        # ----------------------------------------------------

        for key, value in data.items():

            if is_valid_extracted_value(value, key):

                # Store if not already present
                if key not in merged_data:

                    merged_data[key] = value

                # Replace only if the new value is valid
                else:

                    current_value = merged_data[key]

                    if not is_valid_extracted_value(
                        current_value,
                        key
                    ):

                        merged_data[key] = value

    # ========================================================
    # ECG Heart Rate Mapping
    # ========================================================
    #
    # For your project test reports, if ECG HR exists and
    # thalach is missing, use ECG HR.
    #
    # NOTE:
    # Clinically, resting ECG heart rate and maximum heart
    # rate achieved (thalach) are different measurements.
    # This mapping is only for your synthetic/project reports.
    # ========================================================

    if (
        not is_valid_extracted_value(
            merged_data.get("thalach"),
            "thalach"
        )
        and is_valid_extracted_value(
            merged_data.get("ecg_hr")
        )
    ):

        merged_data["thalach"] = merged_data["ecg_hr"]

    # ========================================================
    # Print Final Merged Data
    # ========================================================

    print("\n========================================")
    print("FINAL MERGED DATA")
    print("========================================")

    for key, value in merged_data.items():

        print(f"{key}: {value}")

    print("========================================\n")

    # ========================================================
    # Send Data To Review Page
    # ========================================================

    return render_template(
        "review.html",
        data=merged_data,
        missing_features=[],
        error=None
    )


# ============================================================
# Helper Function - Integer Values
# ============================================================

def get_int_value(field_name):

    value = request.form.get(
        field_name,
        ""
    ).strip()

    if value == "":
        return np.nan

    try:

        return int(float(value))

    except (ValueError, TypeError):

        return np.nan


# ============================================================
# Helper Function - Float Values
# ============================================================

def get_float_value(field_name):

    value = request.form.get(
        field_name,
        ""
    ).strip()

    if value == "":
        return np.nan

    try:

        return float(value)

    except (ValueError, TypeError):

        return np.nan


# ============================================================
# Helper Function - Display Values
# ============================================================

def display_value(value):

    if value is None:
        return "Not Available"

    try:

        if pd.isna(value):
            return "Not Available"

    except Exception:
        pass

    return value


# ============================================================
# Helper Function - Check Missing ML Features
# ============================================================

def find_missing_features(input_data):

    missing_features = []

    for feature in FEATURES:

        value = input_data.iloc[0][feature]

        # ----------------------------------------------------
        # None
        # ----------------------------------------------------

        if value is None:

            missing_features.append(feature)
            continue

        # ----------------------------------------------------
        # NaN
        # ----------------------------------------------------

        try:

            if pd.isna(value):

                missing_features.append(feature)
                continue

        except Exception:
            pass

        # ----------------------------------------------------
        # Empty String
        # ----------------------------------------------------

        if isinstance(value, str):

            if value.strip() == "":

                missing_features.append(feature)
                continue

        # ----------------------------------------------------
        # Invalid default values
        #
        # These fields should not be zero in the application.
        # ----------------------------------------------------

        if feature in [
            "trestbps",
            "chol",
            "thalach"
        ]:

            try:

                if float(value) == 0:

                    missing_features.append(feature)

            except (ValueError, TypeError):

                missing_features.append(feature)

    return list(dict.fromkeys(missing_features))


# ============================================================
# Helper Function - Feature Display Names
# ============================================================

def feature_display_name(feature):

    names = {

        "age":
            "Age",

        "sex":
            "Gender",

        "cp":
            "Chest Pain Type",

        "trestbps":
            "Resting Blood Pressure",

        "chol":
            "Cholesterol",

        "fbs":
            "Fasting Blood Sugar",

        "restecg":
            "Resting ECG",

        "thalach":
            "Maximum Heart Rate",

        "exang":
            "Exercise-Induced Angina",

        "oldpeak":
            "Oldpeak",

        "slope":
            "ST Slope"
    }

    return names.get(
        feature,
        feature
    )


# ============================================================
# Helper Function - Safely Get Model Probability
# ============================================================

def get_risk_probability(input_data, prediction_value):

    # --------------------------------------------------------
    # If model supports probability
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = list(model.classes_)

        # ----------------------------------------------------
        # Find probability corresponding specifically to
        # class 1 = Heart Disease
        # ----------------------------------------------------

        if 1 in classes:

            risk_index = classes.index(1)

            risk_probability = (
                float(probabilities[risk_index]) * 100
            )

        else:

            risk_probability = (
                100.0
                if int(prediction_value) == 1
                else 0.0
            )

        confidence = (
            float(max(probabilities)) * 100
        )

        return (
            probabilities,
            confidence,
            risk_probability
        )

    # --------------------------------------------------------
    # Model without predict_proba
    # --------------------------------------------------------

    probabilities = np.array([])

    confidence = 100.0

    risk_probability = (
        100.0
        if int(prediction_value) == 1
        else 0.0
    )

    return (
        probabilities,
        confidence,
        risk_probability
    )


# ============================================================
# Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    global pdf_file_path

    # ========================================================
    # Read Patient Information
    # ========================================================

    patient_name = request.form.get(
        "patient_name",
        "Not Found"
    ).strip()

    if patient_name == "":
        patient_name = "Not Found"

    hospital_name = request.form.get(
        "hospital_name",
        "Not Found"
    ).strip()

    if hospital_name == "":
        hospital_name = "Not Found"

    report_date = request.form.get(
        "report_date",
        "Not Found"
    ).strip()

    if report_date == "":
        report_date = "Not Found"

    # ========================================================
    # Read Clinical Values
    # ========================================================

    age = get_int_value("age")

    sex = get_int_value("sex")

    cp = get_int_value("cp")

    trestbps = get_int_value("trestbps")

    chol = get_int_value("chol")

    fbs = get_int_value("fbs")

    restecg = get_int_value("restecg")

    thalach = get_int_value("thalach")

    exang = get_int_value("exang")

    oldpeak = get_float_value("oldpeak")

    slope = get_int_value("slope")

    # ========================================================
    # Create Model Input
    # ========================================================

    input_data = pd.DataFrame([{

        "age": age,

        "sex": sex,

        "cp": cp,

        "trestbps": trestbps,

        "chol": chol,

        "fbs": fbs,

        "restecg": restecg,

        "thalach": thalach,

        "exang": exang,

        "oldpeak": oldpeak,

        "slope": slope

    }])

    # ========================================================
    # Exact Training Feature Order
    # ========================================================

    input_data = input_data[FEATURES]

    # ========================================================
    # Print Input Data
    # ========================================================

    print("\n========================================")
    print("INPUT DATA")
    print("========================================")
    print(input_data)
    print("========================================\n")

    # ========================================================
    # Check Missing Features
    # ========================================================

    missing_features = find_missing_features(
        input_data
    )

    # ========================================================
    # Stop Prediction If Data Is Incomplete
    # ========================================================

    if missing_features:

        missing_names = [
            feature_display_name(feature)
            for feature in missing_features
        ]

        print("\n========================================")
        print("PREDICTION STOPPED")
        print("========================================")

        print(
            "Missing ML Features:",
            ", ".join(missing_names)
        )

        print("========================================\n")

        error_message = (
            "Some required medical values are missing. "
            "Please enter or verify all required values "
            "before prediction."
        )

        review_data = {

            "patient_name":
                patient_name,

            "hospital_name":
                hospital_name,

            "report_date":
                report_date,

            "age":
                display_value(age),

            "sex":
                display_value(sex),

            "cp":
                display_value(cp),

            "trestbps":
                display_value(trestbps),

            "chol":
                display_value(chol),

            "fbs":
                display_value(fbs),

            "restecg":
                display_value(restecg),

            "thalach":
                display_value(thalach),

            "exang":
                display_value(exang),

            "oldpeak":
                display_value(oldpeak),

            "slope":
                display_value(slope)
        }

        return render_template(
            "review.html",
            data=review_data,
            missing_features=missing_features,
            missing_feature_names=missing_names,
            error=error_message
        )

    # ========================================================
    # Final NaN Check
    # ========================================================

    if input_data.isnull().any().any():

        print(
            "\nPrediction stopped because NaN values remain."
        )

        return render_template(
            "review.html",
            data=input_data.iloc[0].to_dict(),
            missing_features=FEATURES,
            missing_feature_names=[
                feature_display_name(feature)
                for feature in FEATURES
            ],
            error=(
                "Invalid or incomplete medical data. "
                "Please verify all required values."
            )
        )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    try:

        prediction_value = model.predict(
            input_data
        )[0]

    except Exception as e:

        print("\n========================================")
        print("PREDICTION ERROR")
        print("========================================")
        print(e)
        print("========================================\n")

        return """
        <h3>Prediction Error</h3>

        <p>
        Unable to process the entered medical values.
        Please check the Review Page and try again.
        </p>

        <a href="/upload-page">
            Upload Again
        </a>
        """

    # Convert NumPy value to normal Python integer
    prediction_value = int(prediction_value)

    # ========================================================
    # MODEL PROBABILITY
    # ========================================================

    (
        probability_values,
        confidence,
        risk_probability
    ) = get_risk_probability(
        input_data,
        prediction_value
    )

    confidence = round(
        confidence,
        2
    )

    risk_probability = round(
        risk_probability,
        2
    )

    # ========================================================
    # IMPORTANT DEBUG OUTPUT
    # ========================================================

    print("\n========================================")
    print("MODEL PREDICTION:", prediction_value)

    if hasattr(model, "classes_"):

        print(
            "MODEL CLASSES:",
            model.classes_
        )

    if len(probability_values) > 0:

        print(
            "MODEL PROBABILITIES:",
            probability_values
        )

    print(
        "RISK PROBABILITY:",
        risk_probability,
        "%"
    )

    print(
        "MODEL CONFIDENCE:",
        confidence,
        "%"
    )

    print("========================================\n")

    # ========================================================
    # Prediction Result
    # ========================================================

    if prediction_value == 1:

        prediction = (
            "High Risk of Heart Disease"
        )

        risk_level = "High Risk"

    else:

        prediction = (
            "Low Risk of Heart Disease"
        )

        risk_level = "Low Risk"

    # ========================================================
    # Model Information
    # ========================================================

    best_algorithm = model_info.get(
        "best_algorithm",
        "Random Forest"
    )

    best_accuracy = model_info.get(
        "best_accuracy_percent",
        0.0
    )

    algorithm_comparison = model_info.get(
        "algorithm_comparison",
        []
    )

    # ========================================================
    # Do Not Display Fake 0% Accuracy
    # ========================================================

    if best_accuracy is None:

        best_accuracy_display = "Not Available"

    else:

        try:

            if float(best_accuracy) <= 0:

                best_accuracy_display = "Not Available"

            else:

                best_accuracy_display = best_accuracy

        except (ValueError, TypeError):

            best_accuracy_display = "Not Available"

    # ========================================================
    # Print Final Prediction
    # ========================================================

    print("\n========================================")
    print(
        "Prediction:",
        prediction
    )

    print(
        "Prediction Value:",
        prediction_value
    )

    print(
        "Confidence:",
        confidence,
        "%"
    )

    print(
        "Heart Disease Probability:",
        risk_probability,
        "%"
    )

    print(
        "Best Algorithm:",
        best_algorithm
    )

    print(
        "Best Accuracy:",
        best_accuracy_display
    )

    print("========================================\n")

    # ========================================================
    # Data For Result Page
    # ========================================================

    data = {

        "patient_name":
            patient_name,

        "hospital_name":
            hospital_name,

        "report_date":
            report_date,

        "age":
            display_value(age),

        "sex":
            display_value(sex),

        "cp":
            display_value(cp),

        "trestbps":
            display_value(trestbps),

        "chol":
            display_value(chol),

        "fbs":
            display_value(fbs),

        "restecg":
            display_value(restecg),

        "thalach":
            display_value(thalach),

        "exang":
            display_value(exang),

        "oldpeak":
            display_value(oldpeak),

        "slope":
            display_value(slope)
    }

    # ========================================================
    # Patient Data For PDF
    # ========================================================

    patient_data = {

        "Patient Name":
            patient_name,

        "Hospital":
            hospital_name,

        "Report Date":
            report_date,

        "Age":
            display_value(age),

        "Gender":

            (
                "Male"
                if sex == 1

                else
                "Female"
                if sex == 0

                else
                "Not Available"
            ),

        "Blood Pressure":
            display_value(trestbps),

        "Cholesterol":
            display_value(chol),

        "Blood Sugar":

            (
                "High"
                if fbs == 1

                else
                "Normal"
                if fbs == 0

                else
                "Not Available"
            ),

        "Heart Rate":
            display_value(thalach),

        "Chest Pain":
            display_value(cp),

        "Rest ECG":
            display_value(restecg),

        "Exercise Angina":

            (
                "Yes"
                if exang == 1

                else
                "No"
                if exang == 0

                else
                "Not Available"
            ),

        "Oldpeak":
            display_value(oldpeak),

        "Slope":
            display_value(slope)
    }

    # ========================================================
    # Generate PDF
    # ========================================================
    #
    # Current report_generator.py uses:
    #
    # create_report(patient_data, prediction, probability)
    #
    # Therefore only these THREE arguments are passed.
    # ========================================================

    try:

        generated_pdf = create_report(
            patient_data,
            prediction_value,
            risk_probability
        )

        pdf_file_path = generated_pdf

        print("\n========================================")
        print("PDF generated successfully.")
        print("PDF Path:", pdf_file_path)
        print("========================================\n")

    except Exception as e:

        print("\n========================================")
        print("PDF Generation Error:")
        print(e)
        print("========================================\n")

        pdf_file_path = None

    # ========================================================
    # Result Page
    # ========================================================

    return render_template(
        "result.html",

        data=data,

        prediction=prediction_value,

        risk_level=risk_level,

        probability=confidence,

        risk_probability=risk_probability,

        best_algorithm=best_algorithm,

        best_accuracy=best_accuracy_display,

        algorithm_comparison=algorithm_comparison
    )


# ============================================================
# Download PDF
# ============================================================

@app.route("/download")
def download():

    global pdf_file_path

    if pdf_file_path is None:

        return """
        <h3>No report available.</h3>

        <p>
        Please complete the prediction first.
        </p>

        <a href="/upload-page">
            Start Analysis
        </a>
        """

    if not os.path.isfile(pdf_file_path):

        return """
        <h3>Report file not found.</h3>

        <a href="/upload-page">
            Start New Analysis
        </a>
        """

    return send_file(
        pdf_file_path,
        as_attachment=True,
        download_name="Heart_Disease_Prediction_Report.pdf",
        mimetype="application/pdf"
    )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )