import re


def extract_values(text):

    data = {}

    # ======================================================
    # PATIENT NAME
    # ======================================================

    patient_patterns = [

        r"Patient\s*Name\s*[:\-]?\s*(.*?)(?=\s*Age|\s*Gender|\s*Sex|\s*DOB|\s*Date|\s*Hospital|$)",

        r"Name\s*[:\-]?\s*(.*?)(?=\s*Age|\s*Gender|\s*Sex|\s*DOB|\s*Date|\s*Hospital|$)",

        r"Patient\s*[:\-]?\s*(.*?)(?=\s*Age|\s*Gender|\s*Sex|\s*DOB|\s*Date|\s*Hospital|$)"
    ]

    for pattern in patient_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            name = match.group(1).strip()

            name = re.sub(
                r"\b(Age|Gender|Sex|DOB|Date|Hospital)\b.*",
                "",
                name,
                flags=re.IGNORECASE
            ).strip()

            if len(name) > 2:

                data["patient_name"] = name

                break


    # ======================================================
    # HOSPITAL NAME
    # ======================================================

    hospital_patterns = [

        r"([A-Za-z][A-Za-z &.'-]*\sHospital)",

        r"([A-Za-z][A-Za-z &.'-]*\sMedical Centre)",

        r"([A-Za-z][A-Za-z &.'-]*\sMedical Center)",

        r"([A-Za-z][A-Za-z &.'-]*\sClinic)",

        r"([A-Za-z][A-Za-z &.'-]*\sDiagnostics)",

        r"([A-Za-z][A-Za-z &.'-]*\sLaboratory)"
    ]

    for pattern in hospital_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            data["hospital_name"] = match.group(1).strip()

            break


    # ======================================================
    # REPORT DATE
    # ======================================================

    date_patterns = [

        r"\b(\d{2}/\d{2}/\d{4})\b",

        r"\b(\d{2}-\d{2}-\d{4})\b",

        r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",

        r"\b(\d{1,2}-\d{1,2}-\d{2,4})\b"
    ]

    for pattern in date_patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            data["report_date"] = match.group(1)

            break


    # ======================================================
    # AGE
    # ======================================================

    age_patterns = [

        r"\bAge\s*[:\-]?\s*(\d{1,3})\b",

        r"\b(\d{1,3})\s*Years?\s*Old\b",

        r"\b(\d{1,3})\s*Years?\b"
    ]

    for pattern in age_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            age = int(match.group(1))

            if 1 <= age <= 120:

                data["age"] = age

                break


    # ======================================================
    # GENDER / SEX
    # ======================================================

    gender_patterns = [

        r"\bGender\s*[:\-]?\s*(Male|Female)\b",

        r"\bSex\s*[:\-]?\s*(Male|Female)\b"
    ]

    for pattern in gender_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            gender = match.group(1).lower()

            if gender == "male":

                data["sex"] = 1

            elif gender == "female":

                data["sex"] = 0

            break


    # If gender label is not found, check standalone values

    if "sex" not in data:

        if re.search(
            r"\bMale\b",
            text,
            re.IGNORECASE
        ):

            data["sex"] = 1

        elif re.search(
            r"\bFemale\b",
            text,
            re.IGNORECASE
        ):

            data["sex"] = 0


    # ======================================================
    # BLOOD PRESSURE
    # ======================================================

    bp_patterns = [

        r"Blood\s*Pressure\s*[:\-]?\s*(\d{2,3})",

        r"\bBP\s*[:\-]?\s*(\d{2,3})",

        r"Resting\s*Blood\s*Pressure\s*[:\-]?\s*(\d{2,3})"
    ]

    for pattern in bp_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            bp = int(match.group(1))

            if 50 <= bp <= 250:

                data["trestbps"] = bp

                break


    # ======================================================
    # CHOLESTEROL
    # ======================================================

    chol_patterns = [

        r"Total\s*Cholesterol\s*[:\-]?\s*(\d{2,4})",

        r"Cholesterol\s*[:\-]?\s*(\d{2,4})",

        r"Chol\s*[:\-]?\s*(\d{2,4})"
    ]

    for pattern in chol_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            cholesterol = int(match.group(1))

            if 50 <= cholesterol <= 700:

                data["chol"] = cholesterol

                break


    # ======================================================
    # FASTING BLOOD SUGAR
    # ======================================================

    sugar_patterns = [

        r"Fasting\s*Blood\s*Sugar\s*[:\-]?\s*(\d+(?:\.\d+)?)",

        r"Blood\s*Sugar\s*[:\-]?\s*(\d+(?:\.\d+)?)",

        r"Glucose\s*[:\-]?\s*(\d+(?:\.\d+)?)"
    ]

    for pattern in sugar_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            sugar = float(match.group(1))

            data["blood_sugar"] = sugar

            # Dataset definition:
            # 0 = Normal
            # 1 = High (>120)

            data["fbs"] = 1 if sugar > 120 else 0

            break


    # ======================================================
    # HEART RATE
    # ======================================================

    heart_patterns = [

        r"Heart\s*Rate\s*[:\-]?\s*(\d{2,3})",

        r"Pulse\s*Rate\s*[:\-]?\s*(\d{2,3})",

        r"Pulse\s*[:\-]?\s*(\d{2,3})",

        r"Maximum\s*Heart\s*Rate\s*[:\-]?\s*(\d{2,3})",

        r"Max\s*Heart\s*Rate\s*[:\-]?\s*(\d{2,3})",

        r"Thalach\s*[:\-]?\s*(\d{2,3})"
    ]

    for pattern in heart_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            heart_rate = int(match.group(1))

            if 40 <= heart_rate <= 250:

                data["thalach"] = heart_rate

                break


    # ======================================================
    # CHEST PAIN TYPE
    # ======================================================

    cp_patterns = {

        "Typical Angina": 0,

        "Atypical Angina": 1,

        "Non-anginal Pain": 2,

        "Non-anginal": 2,

        "Asymptomatic": 3
    }

    for key, value in cp_patterns.items():

        if re.search(
            re.escape(key),
            text,
            re.IGNORECASE
        ):

            data["cp"] = value

            break


    # Numeric chest pain type

    if "cp" not in data:

        cp_numeric_patterns = [

            r"Chest\s*Pain\s*(?:Type)?\s*[:\-]?\s*([0-3])",

            r"\bCP\s*[:\-]?\s*([0-3])"
        ]

        for pattern in cp_numeric_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                data["cp"] = int(match.group(1))

                break


    # ======================================================
    # RESTING ECG
    # ======================================================

    restecg_patterns = [

        (
            r"Rest\s*ECG\s*[:\-]?\s*Normal",
            0
        ),

        (
            r"ECG\s*Result\s*[:\-]?\s*Normal",
            0
        ),

        (
            r"Resting\s*ECG\s*[:\-]?\s*Normal",
            0
        ),

        (
            r"ST[-\s]?T\s*Wave\s*Abnormality",
            1
        ),

        (
            r"Left\s*Ventricular\s*Hypertrophy",
            2
        )
    ]

    for pattern, value in restecg_patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):

            data["restecg"] = value

            break


    # Numeric ECG value

    if "restecg" not in data:

        restecg_numeric_patterns = [

            r"Rest\s*ECG\s*[:\-]?\s*([0-2])",

            r"Resting\s*ECG\s*[:\-]?\s*([0-2])",

            r"ECG\s*Result\s*[:\-]?\s*([0-2])"
        ]

        for pattern in restecg_numeric_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                data["restecg"] = int(
                    match.group(1)
                )

                break


    # ======================================================
    # EXERCISE-INDUCED ANGINA
    # ======================================================

    exang_patterns = [

        r"Exercise\s*Angina\s*[:\-]?\s*(Yes|No)",

        r"Exercise\s*Induced\s*Angina\s*[:\-]?\s*(Yes|No)",

        r"Exercise[-\s]*Induced\s*Angina\s*[:\-]?\s*(Yes|No)",

        r"Exang\s*[:\-]?\s*(Yes|No)"
    ]

    for pattern in exang_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            if match.group(1).lower() == "yes":

                data["exang"] = 1

            else:

                data["exang"] = 0

            break


    # Numeric exercise angina

    if "exang" not in data:

        exang_numeric_patterns = [

            r"Exercise\s*Angina\s*[:\-]?\s*([01])",

            r"Exang\s*[:\-]?\s*([01])"
        ]

        for pattern in exang_numeric_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                data["exang"] = int(
                    match.group(1)
                )

                break


    # ======================================================
    # ST DEPRESSION / OLDPEAK
    # ======================================================

    oldpeak_patterns = [

        r"Oldpeak\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",

        r"ST\s*Depression\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",

        r"ST[-\s]*Segment\s*Depression\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)"
    ]

    for pattern in oldpeak_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            data["oldpeak"] = float(
                match.group(1)
            )

            break


    # ======================================================
    # ST SLOPE
    # ======================================================

    slope_patterns = {

        "Upsloping": 0,

        "Flat": 1,

        "Downsloping": 2
    }

    for key, value in slope_patterns.items():

        if re.search(
            re.escape(key),
            text,
            re.IGNORECASE
        ):

            data["slope"] = value

            break


    # Numeric slope

    if "slope" not in data:

        slope_numeric_patterns = [

            r"ST\s*Slope\s*[:\-]?\s*([0-2])",

            r"Slope\s*[:\-]?\s*([0-2])"
        ]

        for pattern in slope_numeric_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                data["slope"] = int(
                    match.group(1)
                )

                break


    # ======================================================
    # DEFAULT VALUES
    # ======================================================

    defaults = {

        "patient_name": "Not Found",

        "hospital_name": "Not Found",

        "report_date": "Not Found",

        "age": 0,

        "sex": 0,

        "trestbps": 0,

        "chol": 0,

        "blood_sugar": 0,

        "fbs": 0,

        "thalach": 0,

        "cp": None,

        "restecg": None,

        "exang": None,

        "oldpeak": None,

        "slope": None
    }


    for key, value in defaults.items():

        if key not in data:

            data[key] = value


    # ======================================================
    # DEBUG OUTPUT
    # ======================================================

    print("\n")
    print("=" * 60)
    print("EXTRACTED VALUES")
    print("=" * 60)

    for key, value in data.items():

        print(f"{key} : {value}")

    print("=" * 60)
    print("\n")


    # ======================================================
    # RETURN
    # ======================================================

    return data