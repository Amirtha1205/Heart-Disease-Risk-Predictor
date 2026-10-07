import re


def extract_values(text):
    """
    Extract patient, hospital, clinical and ECG information
    from medical reports.

    Supports:
    - Digital PDF text
    - Tesseract OCR text
    - Labels and values on the same line
    - Labels and values on separate lines
    - ECG report formats
    - General medical/laboratory report formats

    Main ML features:
        age
        sex
        cp
        trestbps
        chol
        fbs
        restecg
        thalach
        exang
        oldpeak
        slope

    Additional information:
        patient_name
        patient_id
        hospital_name
        report_date
        acquired_at
        reported_at
        blood_sugar
        ecg_hr
        vr
        pr_interval
        qrs_duration
        qt_interval
        qtc
        p_axis
        r_axis
        t_axis
    """

    data = {}

    # ==========================================================
    # CLEAN INPUT TEXT
    # ==========================================================

    if not text:
        text = ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    lines = []

    for line in text.split("\n"):

        line = re.sub(r"[ ]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    normalized_text = "\n".join(lines)

    # ==========================================================
    # HELPER FUNCTIONS
    # ==========================================================

    def clean(value):

        if value is None:
            return None

        value = value.strip()

        value = re.sub(r"^[\s:;\-]+", "", value)
        value = re.sub(r"[\s:;\-]+$", "", value)

        value = re.sub(r"\s+", " ", value)

        return value.strip()

    def get_next_value(label_patterns):

        """
        Supports:

        Age
        45 Years

        Age 45 Years

        Age: 45 Years
        """

        for i, line in enumerate(lines):

            current = line.strip()

            for pattern in label_patterns:

                # --------------------------------------------------
                # SAME LINE
                # --------------------------------------------------

                match = re.match(
                    r"^" + pattern + r"\s*[:\-]?\s*(.+)$",
                    current,
                    re.IGNORECASE
                )

                if match:

                    value = clean(match.group(1))

                    if value:
                        return value

                # --------------------------------------------------
                # NEXT LINE
                # --------------------------------------------------

                if re.fullmatch(
                    pattern,
                    current,
                    re.IGNORECASE
                ):

                    if i + 1 < len(lines):

                        next_value = clean(lines[i + 1])

                        if next_value:
                            return next_value

        return None

    def get_numeric(patterns, min_value=None, max_value=None):

        for pattern in patterns:

            match = re.search(
                pattern,
                normalized_text,
                re.IGNORECASE
            )

            if match:

                try:

                    value = float(match.group(1))

                    if min_value is not None and value < min_value:
                        continue

                    if max_value is not None and value > max_value:
                        continue

                    return value

                except (ValueError, TypeError):

                    pass

        return None

    # ==========================================================
    # HOSPITAL / DIAGNOSTIC CENTER / SCAN CENTER
    # ==========================================================

    hospital_name = None

    # ----------------------------------------------------------
    # 1. Explicit labels
    # ----------------------------------------------------------

    hospital_name = get_next_value([
        r"Hospital Name",
        r"Diagnostic Center Name",
        r"Diagnostic Centre Name",
        r"Medical Center Name",
        r"Medical Centre Name",
        r"Clinic Name",
        r"Scan Center Name",
        r"Scan Centre Name"
    ])

    # ----------------------------------------------------------
    # 2. Search organization name
    # ----------------------------------------------------------

    if not hospital_name:

        for line in lines:

            clean_line = clean(line)
            lower = clean_line.lower()

            # Ignore report title
            if "sample cardiology" in lower:
                continue

            if "laboratory report" in lower:
                continue

            if "cardiology & lab" in lower:
                continue

            # Ignore disclaimer
            if "synthetic sample" in lower:
                continue

            if "not a real medical record" in lower:
                continue

            # Ignore headings
            if lower in [
                "patient information",
                "patient details",
                "biochemistry",
                "cardiology",
                "clinical comment",
                "recommendation",
                "test",
                "result",
                "unit",
                "parameter",
                "parameter result",
                "test result unit"
            ]:
                continue

            # Ignore common ECG headings
            if lower in [
                "ecg",
                "electrocardiogram",
                "12 lead ecg",
                "12-lead ecg"
            ]:
                continue

            # Ignore clinical sections
            if lower.startswith("clinical"):
                continue

            # --------------------------------------------------
            # Organization keywords
            # --------------------------------------------------

            if re.search(
                r"\bdiagnostics?\b",
                lower,
                re.IGNORECASE
            ):

                hospital_name = clean_line
                break

            if re.search(
                r"\bhospital\b",
                lower,
                re.IGNORECASE
            ):

                hospital_name = clean_line
                break

            if re.search(
                r"\bmedical\s+center\b",
                lower,
                re.IGNORECASE
            ):

                hospital_name = clean_line
                break

            if re.search(
                r"\bmedical\s+centre\b",
                lower,
                re.IGNORECASE
            ):

                hospital_name = clean_line
                break

            if re.search(
                r"\bclinic\b",
                lower,
                re.IGNORECASE
            ):

                hospital_name = clean_line
                break

            # --------------------------------------------------
            # NEW: SCAN / SCANS
            # Example:
            # Gengaa Scans
            # --------------------------------------------------

            if re.search(
                r"\bscans?\b",
                lower,
                re.IGNORECASE
            ):

                hospital_name = clean_line
                break

    # ----------------------------------------------------------
    # 3. Specific fallback for current sample report
    # ----------------------------------------------------------

    if not hospital_name:

        for line in lines:

            if line.strip().upper() == "HEART CARE DIAGNOSTICS":

                hospital_name = "HEART CARE DIAGNOSTICS"

                break

    # ----------------------------------------------------------
    # Save hospital name
    # ----------------------------------------------------------

    if hospital_name:

        data["hospital_name"] = hospital_name

    # ==========================================================
    # PATIENT NAME
    # ==========================================================

    patient_name = None

    # ----------------------------------------------------------
    # SAME-LINE FORMAT
    #
    # Patient Name Arun Kumar Report ID SAMPLE-CR-1021
    #
    # Patient Name: Mrs.Susmitha sri
    #
    # Patient Name Mrs.Susmitha sri
    # ----------------------------------------------------------

    match = re.search(
        r"\bPatient\s*Name\s*[:\-]?\s*(.+?)"
        r"(?=\s+Report\s*ID\b"
        r"|\s+Patient\s*ID\b"
        r"|\s+Age\b"
        r"|\s+Gender\b"
        r"|\s+Sex\b"
        r"|\s+Report\s*Date\b"
        r"|\s+Department\b"
        r"|\s+Acquired\s*At\b"
        r"|\s+Reported\s*At\b"
        r"|$)",
        normalized_text,
        re.IGNORECASE
    )

    if match:

        patient_name = clean(match.group(1))

    # ----------------------------------------------------------
    # MULTI-LINE FORMAT
    #
    # Patient Name
    # Arun Kumar
    # ----------------------------------------------------------

    if not patient_name:

        patient_name = get_next_value([
            r"Patient\s*Name"
        ])

    # ----------------------------------------------------------
    # CLEAN PATIENT NAME
    # ----------------------------------------------------------

    if patient_name:

        # Remove accidental labels
        patient_name = re.sub(
            r"\bReport\s*ID\b.*$",
            "",
            patient_name,
            flags=re.IGNORECASE
        )

        patient_name = re.sub(
            r"\bPatient\s*ID\b.*$",
            "",
            patient_name,
            flags=re.IGNORECASE
        )

        patient_name = re.sub(
            r"\bAge\b.*$",
            "",
            patient_name,
            flags=re.IGNORECASE
        )

        patient_name = re.sub(
            r"\bGender\b.*$",
            "",
            patient_name,
            flags=re.IGNORECASE
        )

        patient_name = clean(patient_name)

    # ----------------------------------------------------------
    # INVALID PATIENT NAME
    # ----------------------------------------------------------

    if patient_name:

        invalid_patient_names = [
            "patient",
            "patient name",
            "name",
            "information",
            "details",
            "data",
            "not found",
            "n/a",
            "na"
        ]

        if patient_name.lower() in invalid_patient_names:

            patient_name = None

    # ----------------------------------------------------------
    # SAVE PATIENT NAME
    # ----------------------------------------------------------

    if patient_name:

        data["patient_name"] = patient_name

    # ==========================================================
    # PATIENT ID / REPORT ID
    # ==========================================================

    patient_id = None

    match = re.search(
        r"\b(?:Patient\s*ID|Patient\s*No|Patient\s*Number|Report\s*ID)"
        r"\s*[:\-]?\s*([A-Za-z0-9\-_]+)",
        normalized_text,
        re.IGNORECASE
    )

    if match:

        patient_id = clean(match.group(1))

    if not patient_id:

        patient_id = get_next_value([
            r"Patient\s*ID",
            r"Patient\s*No",
            r"Patient\s*Number",
            r"Report\s*ID"
        ])

    if patient_id:

        data["patient_id"] = patient_id

    # ==========================================================
    # AGE
    # ==========================================================

    age_value = get_next_value([
        r"Age"
    ])

    # ----------------------------------------------------------
    # NEW ECG FORMAT
    #
    # Age / Gender: 20/Female
    #
    # Age/Gender: 20/Male
    #
    # Age Gender: 20/Female
    # ----------------------------------------------------------

    if not age_value:

        match = re.search(
            r"\bAge\s*[/|]?\s*Gender"
            r"\s*[:\-]?\s*"
            r"(\d{1,3})"
            r"\s*(?:Years?|Yrs?)?"
            r"\s*[/|]\s*"
            r"(?:Male|Female|M|F)\b",
            normalized_text,
            re.IGNORECASE
        )

        if match:

            age_value = match.group(1)

    # ----------------------------------------------------------
    # Extract numeric age
    # ----------------------------------------------------------

    if age_value:

        match = re.search(
            r"\b(\d{1,3})\s*(?:Years?|Yrs?)?\b",
            age_value,
            re.IGNORECASE
        )

        if match:

            try:

                age = int(match.group(1))

                if 1 <= age <= 120:

                    data["age"] = age

            except ValueError:

                pass

    # ==========================================================
    # GENDER / SEX
    # ==========================================================

    gender_value = get_next_value([
        r"Gender",
        r"Sex"
    ])

    # ----------------------------------------------------------
    # NEW ECG FORMAT
    #
    # Age / Gender: 20/Female
    # ----------------------------------------------------------

    if not gender_value:

        match = re.search(
            r"\bAge\s*[/|]?\s*Gender"
            r"\s*[:\-]?\s*"
            r"\d{1,3}"
            r"\s*(?:Years?|Yrs?)?"
            r"\s*[/|]\s*"
            r"(Male|Female|M|F)\b",
            normalized_text,
            re.IGNORECASE
        )

        if match:

            gender_value = match.group(1)

    if gender_value:

        gender = gender_value.lower().strip()

        # Female first because "female" contains "male"
        if re.search(r"\bfemale\b", gender):

            data["sex"] = 0

        elif re.search(r"\bmale\b", gender):

            data["sex"] = 1

        elif re.fullmatch(r"f", gender):

            data["sex"] = 0

        elif re.fullmatch(r"m", gender):

            data["sex"] = 1

    # ==========================================================
    # REPORT DATE
    # ==========================================================

    report_date = None

    # ----------------------------------------------------------
    # Same-line format
    # ----------------------------------------------------------

    match = re.search(
        r"\bReport\s+Date\s*[:\-]?\s*(.+?)"
        r"(?=\s+Department\b|$)",
        normalized_text,
        re.IGNORECASE
    )

    if match:

        report_date = clean(match.group(1))

    # ----------------------------------------------------------
    # Multi-line format
    # ----------------------------------------------------------

    if not report_date:

        report_date = get_next_value([
            r"Report Date",
            r"Date of Report",
            r"Report Dt"
        ])

    if report_date:

        data["report_date"] = report_date

    # ==========================================================
    # ACQUIRED DATE / TIME
    # ==========================================================

    acquired_at = get_next_value([
        r"Acquired At",
        r"Acquired On",
        r"Acquisition Time"
    ])

    if acquired_at:

        data["acquired_at"] = acquired_at

    # ==========================================================
    # REPORTED DATE / TIME
    # ==========================================================

    reported_at = get_next_value([
        r"Reported At",
        r"Reported On",
        r"Report Time"
    ])

    if reported_at:

        data["reported_at"] = reported_at

    # ==========================================================
    # BLOOD PRESSURE
    # ==========================================================

    bp_patterns = [

        r"(?:Blood Pressure|Resting Blood Pressure|BP)"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})\s*/\s*(\d{2,3})"

    ]

    for pattern in bp_patterns:

        match = re.search(
            pattern,
            normalized_text,
            re.IGNORECASE
        )

        if match:

            try:

                systolic = int(match.group(1))

                if 50 <= systolic <= 250:

                    data["trestbps"] = systolic

                    break

            except ValueError:

                pass

    # ==========================================================
    # CHOLESTEROL
    # ==========================================================

    cholesterol_patterns = [

        r"Total\s+Cholesterol"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)",

        r"Cholesterol"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)"

    ]

    chol = get_numeric(
        cholesterol_patterns,
        50,
        600
    )

    if chol is not None:

        data["chol"] = int(chol)

    # ==========================================================
    # FASTING BLOOD SUGAR
    # ==========================================================

    sugar_patterns = [

        r"Fasting\s+Blood\s+Sugar"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)",

        r"Blood\s+Sugar"
        r"\s*\(\s*Fasting\s*\)"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)",

        r"Fasting\s+Sugar"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)",

        r"\bFBS"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)"

    ]

    blood_sugar = get_numeric(
        sugar_patterns,
        0,
        600
    )

    if blood_sugar is not None:

        data["blood_sugar"] = int(blood_sugar)

        data["fbs"] = 1 if blood_sugar > 120 else 0

    # ==========================================================
    # HEART RATE
    # ==========================================================

    heart_rate_patterns = [

        r"Heart\s+Rate"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})"
        r"\s*(?:bpm)?",

        r"Maximum\s+Heart\s+Rate"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})"
        r"\s*(?:bpm)?",

        r"Max\s+Heart\s+Rate"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})"
        r"\s*(?:bpm)?",

        r"\bHR"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})"
        r"\s*(?:bpm)?"

    ]

    heart_rate = get_numeric(
        heart_rate_patterns,
        30,
        250
    )

    if heart_rate is not None:

        data["thalach"] = int(heart_rate)

        data["ecg_hr"] = int(heart_rate)

    # ==========================================================
    # ECG VENTRICULAR RATE / VR
    # ==========================================================

    vr_patterns = [

        r"\bVR"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})"
        r"\s*(?:bpm)?",

        r"Ventricular\s+Rate"
        r"\s*[:\-]?\s*"
        r"(\d{2,3})"
        r"\s*(?:bpm)?"

    ]

    vr = get_numeric(
        vr_patterns,
        30,
        250
    )

    if vr is not None:

        data["vr"] = int(vr)

    # ==========================================================
    # CHEST PAIN
    # ==========================================================

    chest_pain = get_next_value([
        r"Chest Pain Type",
        r"Chest Pain"
    ])

    if chest_pain:

        cp_text = chest_pain.lower().strip()

        if (
            "no typical angina" in cp_text
            or cp_text in [
                "no",
                "none",
                "not available",
                "not reported"
            ]
        ):

            data["cp"] = None

        elif "atypical angina" in cp_text:

            data["cp"] = 1

        elif "typical angina" in cp_text:

            data["cp"] = 0

        elif (
            "non-anginal" in cp_text
            or "non anginal" in cp_text
        ):

            data["cp"] = 2

        elif "asymptomatic" in cp_text:

            data["cp"] = 3

    # ----------------------------------------------------------
    # Numeric CP
    # ----------------------------------------------------------

    if "cp" not in data:

        match = re.search(
            r"(?:Chest\s+Pain|CP)"
            r"\s*[:\-]\s*"
            r"([0-3])\b",
            normalized_text,
            re.IGNORECASE
        )

        if match:

            data["cp"] = int(match.group(1))

    # ==========================================================
    # RESTING ECG
    # ==========================================================

    rest_ecg = get_next_value([
        r"Rest ECG",
        r"Resting ECG"
    ])

    if rest_ecg:

        ecg_text = rest_ecg.lower().strip()

        if ecg_text == "normal":

            data["restecg"] = 0

        elif (
            "st-t" in ecg_text
            or "st t" in ecg_text
            or "abnormality" in ecg_text
            or "abnormal" in ecg_text
        ):

            data["restecg"] = 1

        elif (
            "lvh" in ecg_text
            or "left ventricular hypertrophy" in ecg_text
        ):

            data["restecg"] = 2

    # ----------------------------------------------------------
    # Numeric ECG
    # ----------------------------------------------------------

    if "restecg" not in data:

        match = re.search(
            r"(?:Rest\s+ECG|Resting\s+ECG)"
            r"\s*[:\-]\s*"
            r"([0-2])\b",
            normalized_text,
            re.IGNORECASE
        )

        if match:

            data["restecg"] = int(match.group(1))

    # ==========================================================
    # EXERCISE-INDUCED ANGINA
    # ==========================================================

    exercise_angina = get_next_value([
        r"Exercise Induced Angina",
        r"Exercise-Induced Angina",
        r"Exercise Angina",
        r"Exang"
    ])

    if exercise_angina:

        exang_text = exercise_angina.lower().strip()

        if exang_text in [
            "yes",
            "present",
            "positive"
        ]:

            data["exang"] = 1

        elif exang_text in [
            "no",
            "absent",
            "negative"
        ]:

            data["exang"] = 0

    # ----------------------------------------------------------
    # Numeric exang
    # ----------------------------------------------------------

    if "exang" not in data:

        match = re.search(
            r"(?:Exercise\s+Induced\s+Angina|"
            r"Exercise-Induced\s+Angina|"
            r"Exercise\s+Angina|"
            r"Exang)"
            r"\s*[:\-]\s*"
            r"([01])\b",
            normalized_text,
            re.IGNORECASE
        )

        if match:

            data["exang"] = int(match.group(1))

    # ==========================================================
    # OLDPEAK
    # ==========================================================

    oldpeak_patterns = [

        r"Oldpeak"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)",

        r"ST\s+Depression"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)"

    ]

    oldpeak = get_numeric(
        oldpeak_patterns,
        0,
        10
    )

    if oldpeak is not None:

        data["oldpeak"] = oldpeak

    # ==========================================================
    # ST SLOPE
    # ==========================================================

    slope = get_next_value([
        r"ST Slope",
        r"Slope"
    ])

    if slope:

        slope_text = slope.lower().strip()

        if "upsloping" in slope_text:

            data["slope"] = 0

        elif "flat" in slope_text:

            data["slope"] = 1

        elif "downsloping" in slope_text:

            data["slope"] = 2

    # ----------------------------------------------------------
    # Numeric slope
    # ----------------------------------------------------------

    if "slope" not in data:

        match = re.search(
            r"(?:ST\s+Slope|Slope)"
            r"\s*[:\-]\s*"
            r"([0-2])\b",
            normalized_text,
            re.IGNORECASE
        )

        if match:

            data["slope"] = int(match.group(1))

    # ==========================================================
    # ECG ADDITIONAL VALUES
    # ==========================================================

    # ----------------------------------------------------------
    # PR INTERVAL
    # Supports:
    #
    # PR Interval: 138 ms
    # PRI: 138ms
    # ----------------------------------------------------------

    pr_patterns = [

        r"\bPR\s+Interval"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms",

        r"\bPRI"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms"

    ]

    pr = get_numeric(
        pr_patterns,
        0,
        500
    )

    if pr is not None:

        data["pr_interval"] = pr

    # ----------------------------------------------------------
    # QRS
    #
    # QRS Duration: 66 ms
    # QRSD: 66ms
    # ----------------------------------------------------------

    qrs_patterns = [

        r"\bQRS\s+Duration"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms",

        r"\bQRSD"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms"

    ]

    qrs = get_numeric(
        qrs_patterns,
        0,
        500
    )

    if qrs is not None:

        data["qrs_duration"] = qrs

    # ----------------------------------------------------------
    # QT
    #
    # QT Interval: 336 ms
    # QT: 336ms
    # ----------------------------------------------------------

    qt_patterns = [

        r"\bQT\s+Interval"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms",

        r"\bQT"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms"

    ]

    qt = get_numeric(
        qt_patterns,
        0,
        1000
    )

    if qt is not None:

        data["qt_interval"] = qt

    # ----------------------------------------------------------
    # QTc
    #
    # QTcB: 427 ms
    # QTc: 427 ms
    # ----------------------------------------------------------

    qtc_patterns = [

        r"\bQTcB"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms",

        r"\bQTc"
        r"\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*ms"

    ]

    qtc = get_numeric(
        qtc_patterns,
        0,
        1000
    )

    if qtc is not None:

        data["qtc"] = qtc

    # ==========================================================
    # ECG AXIS VALUES
    # ==========================================================

    axis_fields = {

        "p_axis": [
            r"\bP\s+Axis"
            r"\s*[:\-]?\s*"
            r"(-?\d+(?:\.\d+)?)"
        ],

        "r_axis": [
            r"\bR\s+Axis"
            r"\s*[:\-]?\s*"
            r"(-?\d+(?:\.\d+)?)"
        ],

        "t_axis": [
            r"\bT\s+Axis"
            r"\s*[:\-]?\s*"
            r"(-?\d+(?:\.\d+)?)"
        ]

    }

    for field, patterns in axis_fields.items():

        value = get_numeric(
            patterns,
            -180,
            180
        )

        if value is not None:

            data[field] = value

    # ==========================================================
    # P-R-T COMBINED AXIS
    #
    # Example:
    #
    # P-R-T: 36° 50° 28°
    #
    # P-R-T 36 50 28
    # ==========================================================

    prt_match = re.search(
        r"\bP\s*[-/]\s*R\s*[-/]\s*T"
        r"\s*[:\-]?\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*[°º]?\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*[°º]?\s*"
        r"(-?\d+(?:\.\d+)?)",
        normalized_text,
        re.IGNORECASE
    )

    if prt_match:

        try:

            p_axis = float(prt_match.group(1))
            r_axis = float(prt_match.group(2))
            t_axis = float(prt_match.group(3))

            data["p_axis"] = p_axis
            data["r_axis"] = r_axis
            data["t_axis"] = t_axis

        except ValueError:

            pass

    # ==========================================================
    # DEFAULT VALUES
    # ==========================================================

    defaults = {

        "patient_name": "Not Found",
        "hospital_name": "Not Found",
        "patient_id": "Not Found",

        "report_date": "Not Found",
        "acquired_at": "Not Found",
        "reported_at": "Not Found",

        "age": 0,
        "sex": 0,

        "trestbps": 0,
        "chol": 0,

        "fbs": 0,
        "blood_sugar": 0,

        "thalach": 0,

        "cp": None,
        "restecg": None,
        "exang": None,
        "oldpeak": None,
        "slope": None,

        "ecg_hr": 0,
        "vr": 0,

        "pr_interval": None,
        "qrs_duration": None,
        "qt_interval": None,
        "qtc": None,

        "p_axis": None,
        "r_axis": None,
        "t_axis": None
    }

    for key, default_value in defaults.items():

        if key not in data:

            data[key] = default_value

    # ==========================================================
    # DEBUG OUTPUT
    # ==========================================================

    print()
    print("========================================")
    print("EXTRACTED DATA")
    print("========================================")

    for key, value in data.items():

        print(f"{key}: {value}")

    print("========================================")
    print()

    return data