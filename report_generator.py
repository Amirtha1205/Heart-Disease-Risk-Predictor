from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

import os


def create_report(patient_data, prediction, probability):

    # ======================================================
    # REPORT FOLDER
    # ======================================================

    report_folder = "reports"

    os.makedirs(
        report_folder,
        exist_ok=True
    )


    # ======================================================
    # REPORT FILE
    # ======================================================

    filename = os.path.join(
        report_folder,
        "heart_disease_prediction_report.pdf"
    )


    # ======================================================
    # DOCUMENT
    # ======================================================

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )


    # ======================================================
    # STYLES
    # ======================================================

    styles = getSampleStyleSheet()


    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=15
    )


    heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )


    normal = ParagraphStyle(
        "NormalText",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=5
    )


    result_style = ParagraphStyle(
        "ResultStyle",
        parent=styles["BodyText"],
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        spaceBefore=8,
        spaceAfter=8
    )


    # ======================================================
    # STORY
    # ======================================================

    story = []


    # ======================================================
    # TITLE
    # ======================================================

    story.append(
        Paragraph(
            "AI Heart Disease Risk Prediction Report",
            title
        )
    )


    story.append(
        Paragraph(
            "AI-Based Heart Disease Risk Prediction System",
            ParagraphStyle(
                "Subtitle",
                parent=normal,
                alignment=TA_CENTER,
                fontSize=10,
                spaceAfter=20
            )
        )
    )


    # ======================================================
    # PATIENT INFORMATION
    # ======================================================

    story.append(
        Paragraph(
            "Patient Information",
            heading
        )
    )


    patient_rows = []


    for key, value in patient_data.items():

        patient_rows.append(
            [
                Paragraph(
                    f"<b>{key}</b>",
                    normal
                ),

                Paragraph(
                    str(value),
                    normal
                )
            ]
        )


    patient_table = Table(
        patient_rows,
        colWidths=[
            55 * mm,
            105 * mm
        ]
    )


    patient_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )


    story.append(
        patient_table
    )


    story.append(
        Spacer(1, 20)
    )


    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    story.append(
        Paragraph(
            "Prediction Result",
            heading
        )
    )


    # Convert model output into readable result

    if prediction == 1:

        risk_level = "High Risk"

        result_text = (
            '<font color="red">'
            '<b>High Risk of Heart Disease</b>'
            '</font>'
        )

        recommendations = [

            "Consult a qualified medical professional for further evaluation.",

            "Monitor blood pressure regularly.",

            "Maintain healthy cholesterol levels.",

            "Follow a balanced and heart-healthy diet.",

            "Exercise regularly according to medical advice.",

            "Avoid smoking and other harmful habits.",

            "Follow the recommendations provided by your doctor."
        ]

    else:

        risk_level = "Low Risk"

        result_text = (
            '<font color="green">'
            '<b>Low Risk of Heart Disease</b>'
            '</font>'
        )

        recommendations = [

            "Maintain a balanced and healthy diet.",

            "Exercise regularly.",

            "Monitor blood pressure periodically.",

            "Maintain a healthy body weight.",

            "Continue regular health check-ups."
        ]


    story.append(
        Paragraph(
            result_text,
            result_style
        )
    )


    # ======================================================
    # RISK PROBABILITY
    # ======================================================

    story.append(
        Paragraph(
            f"<b>Risk Probability:</b> {probability:.2f}%",
            normal
        )
    )


    story.append(
        Paragraph(
            f"<b>Risk Level:</b> {risk_level}",
            normal
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # ======================================================
    # RECOMMENDATIONS
    # ======================================================

    story.append(
        Paragraph(
            "Recommendations",
            heading
        )
    )


    for item in recommendations:

        story.append(
            Paragraph(
                "• " + item,
                normal
            )
        )


    story.append(
        Spacer(1, 20)
    )


    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    story.append(
        Paragraph(
            "Prediction Model Information",
            heading
        )
    )


    story.append(
        Paragraph(
            "The prediction is generated using a trained "
            "Random Forest machine learning model based on "
            "11 clinical input features.",
            normal
        )
    )


    story.append(
        Paragraph(
            "Input features include age, sex, chest pain type, "
            "resting blood pressure, cholesterol, fasting blood "
            "sugar, resting ECG, maximum heart rate, "
            "exercise-induced angina, ST depression and ST slope.",
            normal
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # ======================================================
    # DISCLAIMER
    # ======================================================

    story.append(
        Paragraph(
            "Medical Disclaimer",
            heading
        )
    )


    story.append(
        Paragraph(
            "This report is generated by an AI-based prediction "
            "system for educational and research purposes. "
            "The prediction should not be considered a medical "
            "diagnosis. Patients should consult a qualified "
            "healthcare professional for proper medical evaluation "
            "and treatment.",
            normal
        )
    )


    story.append(
        Spacer(1, 20)
    )


    # ======================================================
    # FOOTER
    # ======================================================

    story.append(
        Paragraph(
            "<b>Generated by AI Heart Disease Risk Predictor</b>",
            ParagraphStyle(
                "Footer",
                parent=normal,
                alignment=TA_CENTER,
                fontSize=9
            )
        )
    )


    # ======================================================
    # BUILD PDF
    # ======================================================

    doc.build(
        story
    )


    # Return generated PDF path

    return filename