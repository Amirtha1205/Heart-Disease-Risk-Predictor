import os
import cv2
import numpy as np
import pytesseract
import pymupdf

from PIL import Image


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Basic image preprocessing for medical report OCR.
    """

    if image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def deskew_image(image):
    """
    Correct small rotations in scanned medical reports.
    """

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find coordinates of foreground pixels
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Avoid unnecessary rotation
    if abs(angle) < 0.5:
        return image

    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def threshold_image(image):
    """
    Create a high-contrast black-and-white image.
    """

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return threshold


# ============================================================
# TESSERACT OCR
# ============================================================

def perform_ocr(image):
    """
    Perform OCR using Tesseract.
    """

    if image is None:
        return ""

    try:

        # Tesseract configuration
        config = r"--oem 3 --psm 6"

        text = pytesseract.image_to_string(
            image,
            config=config
        )

        return text

    except Exception as e:

        print("Tesseract OCR Error:", e)

        return ""


# ============================================================
# MEDICAL IMAGE OCR
# ============================================================

def ocr_medical_image(file_path):
    """
    OCR for medical report images.

    Multiple regions are processed because medical reports
    often contain patient details at the top and ECG values
    in the middle or bottom sections.
    """

    try:

        image = cv2.imread(file_path)

        if image is None:
            print("Unable to read image:", file_path)
            return ""

        # ----------------------------------------------------
        # Resize small images
        # ----------------------------------------------------

        height, width = image.shape[:2]

        if width < 1600:

            scale = 1600 / width

            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # Deskew
        # ----------------------------------------------------

        image = deskew_image(image)

        # ----------------------------------------------------
        # Full image OCR
        # ----------------------------------------------------

        results = []

        full_text = perform_ocr(image)

        if full_text:
            results.append(full_text)

        # ----------------------------------------------------
        # TOP SECTION
        # Patient details
        # ----------------------------------------------------

        top_end = int(height * 0.40)

        top_section = image[
            0:top_end,
            0:width
        ]

        top_text = perform_ocr(top_section)

        if top_text:
            results.append(top_text)

        # ----------------------------------------------------
        # MIDDLE SECTION
        # ECG measurements
        # ----------------------------------------------------

        middle_start = int(height * 0.20)
        middle_end = int(height * 0.80)

        middle_section = image[
            middle_start:middle_end,
            0:width
        ]

        middle_text = perform_ocr(middle_section)

        if middle_text:
            results.append(middle_text)

        # ----------------------------------------------------
        # BOTTOM SECTION
        # Report / interpretation
        # ----------------------------------------------------

        bottom_start = int(height * 0.60)

        bottom_section = image[
            bottom_start:height,
            0:width
        ]

        bottom_text = perform_ocr(bottom_section)

        if bottom_text:
            results.append(bottom_text)

        # ----------------------------------------------------
        # Thresholded image OCR
        # ----------------------------------------------------

        threshold = threshold_image(image)

        threshold_text = perform_ocr(threshold)

        if threshold_text:
            results.append(threshold_text)

        # ----------------------------------------------------
        # Combine OCR results
        # ----------------------------------------------------

        combined_text = "\n".join(results)

        # Remove excessive blank lines
        lines = []

        for line in combined_text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        final_text = "\n".join(lines)

        return final_text

    except Exception as e:

        print("Image OCR Error:", e)

        return ""


# ============================================================
# PDF DIRECT TEXT EXTRACTION
# ============================================================

def extract_pdf_text(document):
    """
    Extract existing text directly from a digital PDF.

    This is useful for digitally generated ECG reports because
    Tesseract may lose table values that already exist as text.
    """

    pages = []

    try:

        for page in document:

            text = page.get_text("text")

            if text:
                pages.append(text)

    except Exception as e:

        print("PDF text extraction error:", e)

    return "\n".join(pages)


# ============================================================
# PDF OCR
# ============================================================

def ocr_pdf(document):
    """
    OCR scanned/image-based PDF pages.
    """

    results = []

    try:

        for page_number, page in enumerate(document):

            print(
                f"Processing PDF page {page_number + 1}..."
            )

            # Render PDF page
            matrix = pymupdf.Matrix(
                250 / 72,
                250 / 72
            )

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            )

            image = image.reshape(
                pix.height,
                pix.width,
                pix.n
            )

            if pix.n == 4:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGBA2BGR
                )

            else:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )

            # OCR full page
            text = perform_ocr(image)

            if text:
                results.append(text)

            # OCR thresholded page
            threshold = threshold_image(image)

            threshold_text = perform_ocr(threshold)

            if threshold_text:
                results.append(threshold_text)

    except Exception as e:

        print("PDF OCR Error:", e)

    return "\n".join(results)


# ============================================================
# MAIN REPORT READER
# ============================================================

def read_report(file_path):
    """
    Main OCR function.

    Supports:
        - JPG
        - JPEG
        - PNG
        - BMP
        - TIFF
        - PDF

    Digital PDFs are first read directly.
    Scanned PDFs are processed using OCR.
    """

    print("\n========================================")
    print("READING MEDICAL REPORT")
    print("========================================")

    print("File:", file_path)

    if not os.path.exists(file_path):

        print("File does not exist.")

        return ""

    extension = os.path.splitext(
        file_path
    )[1].lower()

    extracted_text = ""

    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        try:

            document = pymupdf.open(file_path)

            # ----------------------------------------------
            # First try direct PDF text extraction
            # ----------------------------------------------

            direct_text = extract_pdf_text(
                document
            )

            medical_keywords = [
                "patient",
                "age",
                "gender",
                "heart",
                "cholesterol",
                "blood",
                "pressure",
                "ecg",
                "heart rate",
                "hr",
                "qt",
                "qtc",
                "qrs",
                "pr"
            ]

            keyword_count = sum(
                keyword.lower() in
                direct_text.lower()
                for keyword in medical_keywords
            )

            # Use direct text if sufficient
            if (
                len(direct_text.strip()) >= 50
                and keyword_count >= 3
            ):

                print(
                    "Digital PDF text detected."
                )

                extracted_text = direct_text

            else:

                print(
                    "PDF appears scanned/image-based."
                )

                extracted_text = ocr_pdf(
                    document
                )

            document.close()

        except Exception as e:

            print("PDF processing error:", e)

            extracted_text = ""

    # ========================================================
    # IMAGE
    # ========================================================

    elif extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    ]:

        extracted_text = ocr_medical_image(
            file_path
        )

    else:

        print(
            "Unsupported file format:",
            extension
        )

        return ""

    # ========================================================
    # PRINT OCR RESULT
    # ========================================================

    print("\n========================================")
    print("OCR / PDF TEXT")
    print("========================================")

    print(extracted_text)

    print("========================================\n")

    return extracted_text