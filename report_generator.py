from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

import os


# ============================================================
# Create PDF Report
# ============================================================

def create_report(
    patient_data,
    prediction,
    probability
):

    # ========================================================
    # Report Folder
    # ========================================================

    report_folder = "reports"

    os.makedirs(
        report_folder,
        exist_ok=True
    )


    # ========================================================
    # PDF File Path
    # ========================================================

    filename = os.path.join(

        report_folder,

        "heart_disease_prediction_report.pdf"

    )


    # ========================================================
    # PDF Document
    # ========================================================

    doc = SimpleDocTemplate(

        filename,

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=20 * mm,

        bottomMargin=20 * mm

    )


    # ========================================================
    # Styles
    # ========================================================

    styles = getSampleStyleSheet()


    title = ParagraphStyle(

        "ReportTitle",

        parent=styles["Heading1"],

        alignment=TA_CENTER,

        fontSize=18,

        leading=22,

        spaceAfter=15

    )


    subtitle = ParagraphStyle(

        "Subtitle",

        parent=styles["BodyText"],

        alignment=TA_CENTER,

        fontSize=10,

        leading=14,

        spaceAfter=20

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


    footer_style = ParagraphStyle(

        "Footer",

        parent=normal,

        alignment=TA_CENTER,

        fontSize=9

    )


    # ========================================================
    # Story
    # ========================================================

    story = []


    # ========================================================
    # Title
    # ========================================================

    story.append(

        Paragraph(

            "AI Heart Disease Risk Prediction Report",

            title

        )

    )


    story.append(

        Paragraph(

            "AI-Based Heart Disease Risk Prediction System",

            subtitle

        )

    )


    # ========================================================
    # Patient Information
    # ========================================================

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

        colWidths=[55 * mm, 105 * mm]

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


    story.append(patient_table)

    story.append(
        Spacer(1, 20)
    )


    # ========================================================
    # Prediction Result
    # ========================================================

    story.append(

        Paragraph(

            "Prediction Result",

            heading

        )

    )


    # ========================================================
    # Risk Level
    # ========================================================

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


    # ========================================================
    # Result
    # ========================================================

    story.append(

        Paragraph(

            result_text,

            result_style

        )

    )


    # ========================================================
    # Probability
    # ========================================================

    story.append(

        Paragraph(

            f"<b>Heart Disease Probability:</b> "
            f"{probability:.2f}%",

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


    # ========================================================
    # Recommendations
    # ========================================================

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


    # ========================================================
    # Prediction Model Information
    # ========================================================

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


    # ========================================================
    # Medical Disclaimer
    # ========================================================

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


    # ========================================================
    # Footer
    # ========================================================

    story.append(

        Paragraph(

            "<b>Generated by AI Heart Disease Risk Predictor</b>",

            footer_style

        )

    )


    # ========================================================
    # Build PDF
    # ========================================================

    doc.build(story)


    # ========================================================
    # Return Generated PDF Path
    # ========================================================

    return filename