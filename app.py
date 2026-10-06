
from flask import Flask, render_template, request, send_file
import os
import joblib
import pandas as pd

from utils.ocr import read_report
from utils.extractor import extract_values
from report_generator import create_report


app = Flask(__name__)

# --------------------------------------------------
# Folder Configuration
# --------------------------------------------------

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

model = joblib.load("heart_model.pkl")


# --------------------------------------------------
# Global PDF File Path
# --------------------------------------------------

pdf_file_path = None


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# Upload Page
# --------------------------------------------------

@app.route("/upload-page")
def upload_page():
    return render_template("upload.html")


# --------------------------------------------------
# Upload Medical Reports
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():

    global pdf_file_path

    files = request.files.getlist("reports")

    if not files or all(file.filename == "" for file in files):
        return "No files selected."

    combined_text = ""

    for file in files:

        if file and file.filename:

            filename = file.filename

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            try:

                extracted_text = read_report(filepath)

                if extracted_text:
                    combined_text += (
                        "\n\n--- REPORT ---\n\n"
                        + extracted_text
                    )

            except Exception as e:

                print("OCR Error:", e)

    # --------------------------------------------------
    # Extract Patient Information and Medical Values
    # --------------------------------------------------

    try:

        data = extract_values(combined_text)

    except Exception as e:

        print("Extraction Error:", e)

        data = {}

    # Store extracted data temporarily
    # and send it to the review page.

    return render_template(
        "review.html",
        data=data
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    global pdf_file_path

    try:

        # --------------------------------------------------
        # Patient Information
        # --------------------------------------------------

        patient_name = request.form.get(
            "patient_name",
            "Not Found"
        )

        hospital_name = request.form.get(
            "hospital_name",
            "Not Found"
        )

        report_date = request.form.get(
            "report_date",
            "Not Found"
        )

        # --------------------------------------------------
        # 11 Model Features
        # --------------------------------------------------

        age = int(
            request.form.get("age", 0)
        )

        sex = int(
            request.form.get("sex", 0)
        )

        cp = int(
            request.form.get("cp", 0)
        )

        trestbps = int(
            request.form.get("trestbps", 0)
        )

        chol = int(
            request.form.get("chol", 0)
        )

        fbs = int(
            request.form.get("fbs", 0)
        )

        restecg = int(
            request.form.get("restecg", 0)
        )

        thalach = int(
            request.form.get("thalach", 0)
        )

        exang = int(
            request.form.get("exang", 0)
        )

        oldpeak = float(
            request.form.get("oldpeak", 0)
        )

        slope = int(
            request.form.get("slope", 0)
        )

    except (ValueError, TypeError):

        return (
            "Invalid patient data. "
            "Please check the entered values."
        )


    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    input_data = pd.DataFrame(
        [{
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
        }]
    )


    # --------------------------------------------------
    # Model Prediction
    # --------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        risk_probability = float(
            probabilities[1] * 100
        )

    except Exception as e:

        print("Prediction Error:", e)

        return (
            "Prediction failed. "
            "Please check the input values and model."
        )


    # --------------------------------------------------
    # Risk Classification
    # --------------------------------------------------

    if prediction == 1:

        risk_level = "High Risk"

    else:

        risk_level = "Low Risk"


    # --------------------------------------------------
    # Result Data
    # --------------------------------------------------

    result_data = {

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

    }


    # --------------------------------------------------
    # Patient Data for PDF Report
    # --------------------------------------------------

    patient_data = {

        "Patient Name": patient_name,

        "Hospital Name": hospital_name,

        "Report Date": report_date,

        "Age": age,

        "Gender":
            "Male"
            if sex == 1
            else "Female",

        "Blood Pressure": trestbps,

        "Cholesterol": chol,

        "Blood Sugar":
            "High"
            if fbs == 1
            else "Normal",

        "Heart Rate": thalach,

        "Chest Pain": cp,

        "Rest ECG": restecg,

        "Exercise Angina": exang,

        "Oldpeak": oldpeak,

        "ST Slope": slope

    }


    # --------------------------------------------------
    # Generate PDF Report
    # --------------------------------------------------

    try:

        pdf_file_path = create_report(
            patient_data,
            prediction,
            risk_probability
        )

    except Exception as e:

        print("PDF Generation Error:", e)

        pdf_file_path = None


    # --------------------------------------------------
    # Result Page
    # --------------------------------------------------

    return render_template(
        "result.html",

        prediction=prediction,

        risk_level=risk_level,

        probability=round(
            risk_probability,
            2
        ),

        patient_name=patient_name,

        hospital_name=hospital_name,

        report_date=report_date,

        data=result_data
    )


# --------------------------------------------------
# Download PDF Report
# --------------------------------------------------

@app.route("/download-report")
def download_report():

    global pdf_file_path

    if pdf_file_path and os.path.exists(
        pdf_file_path
    ):

        return send_file(
            pdf_file_path,
            as_attachment=True
        )

    return "Report not available."


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run()