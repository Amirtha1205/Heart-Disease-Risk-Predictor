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
# Upload Medical Reports
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    reports = request.files.getlist("reports")

    if not reports:

        return """

        <h3>No report uploaded.</h3>

        <a href="/upload-page">
            Go Back
        </a>

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
        # Merge Extracted Data
        # ----------------------------------------------------

        for key, value in data.items():

            # Keep extracted values
            if value is not None:

                merged_data[key] = value


    # ========================================================
    # Print Extracted Data
    # ========================================================

    print("\n========================================")
    print("EXTRACTED DATA")
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

        return int(value)

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
# Helper Function - Display Missing Values
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
        # Empty string
        # ----------------------------------------------------

        if isinstance(value, str) and value.strip() == "":

            missing_features.append(feature)

            continue


        # ----------------------------------------------------
        # Invalid default values
        #
        # These fields cannot realistically be 0.
        # Therefore 0 means missing in this application.
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


    return missing_features


# ============================================================
# Helper Function - Convert Feature Names For Display
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
    # Keep Exact Training Feature Order
    # ========================================================

    input_data = input_data[FEATURES]


    # ========================================================
    # Print Input
    # ========================================================

    print("\n========================================")
    print("INPUT DATA")
    print("========================================")

    print(input_data)

    print("========================================\n")


    # ========================================================
    # CHECK MISSING VALUES BEFORE PREDICTION
    # ========================================================

    missing_features = find_missing_features(

        input_data

    )


    # ========================================================
    # STOP PREDICTION IF DATA IS INCOMPLETE
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


        # ----------------------------------------------------
        # Send back to Review Page
        # ----------------------------------------------------

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
    # FINAL NaN CHECK
    # ========================================================

    if input_data.isnull().any().any():

        print("\nPrediction stopped because NaN values remain.")

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
    # Prediction
    # ========================================================

    try:

        prediction_value = model.predict(

            input_data

        )[0]

    except Exception as e:

        print("\nPrediction Error:")
        print(e)


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


    # ========================================================
    # Prediction Probability
    # ========================================================

    if hasattr(model, "predict_proba"):

        probability_values = model.predict_proba(

            input_data

        )[0]


        # ----------------------------------------------------
        # Overall Model Confidence
        # ----------------------------------------------------

        confidence = round(

            max(probability_values) * 100,

            2

        )


        # ----------------------------------------------------
        # Heart Disease Probability
        # Class 1 = Heart Disease
        # ----------------------------------------------------

        if len(probability_values) > 1:

            risk_probability = round(

                probability_values[1] * 100,

                2

            )

        else:

            risk_probability = (

                100.0

                if prediction_value == 1

                else 0.0

            )


    else:

        confidence = 0.0

        risk_probability = (

            100.0

            if prediction_value == 1

            else 0.0

        )


    # ========================================================
    # Prediction Result
    # ========================================================

    if prediction_value == 1:

        prediction = "High Risk of Heart Disease"

    else:

        prediction = "Low Risk of Heart Disease"


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
    # Print Prediction Result
    # ========================================================

    print("\n========================================")

    print(

        "Prediction:",

        prediction

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

    pdf_file_path = os.path.join(

        REPORT_FOLDER,

        "Heart_Disease_Report.pdf"

    )


    try:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # probability is passed ONLY ONCE.
        #
        # Previous code passed "confidence" as the
        # 4th positional argument and also passed
        # probability=risk_probability.
        #
        # That caused:
        #
        # create_report() got multiple values
        # for argument 'probability'
        # ----------------------------------------------------

        create_report(

            pdf_file_path,

            patient_data,

            prediction,

            probability=risk_probability,

            best_algorithm=best_algorithm,

            best_accuracy=best_accuracy_display,

            algorithm_comparison=algorithm_comparison

        )


        print("\nPDF generated successfully.")


    except Exception as e:

        print("\nPDF Generation Error:")
        print(e)

        pdf_file_path = None


    # ========================================================
    # Result Page
    # ========================================================

    return render_template(

        "result.html",

        data=data,

        prediction=prediction,

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

    if pdf_file_path is None:

        return """

        <h3>
            No report available.
        </h3>

        <p>
            Please perform a prediction first.
        </p>

        <a href="/upload-page">
            Start Analysis
        </a>

        """


    if not os.path.exists(pdf_file_path):

        return """

        <h3>
            Report file not found.
        </h3>

        <a href="/upload-page">
            Start New Analysis
        </a>

        """


    return send_file(

        pdf_file_path,

        as_attachment=True

    )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )