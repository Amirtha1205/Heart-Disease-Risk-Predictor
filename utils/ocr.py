import os
import fitz
import pytesseract
from PIL import Image


def read_report(file_path):
    """
    Extract text from an image or PDF medical report.
    Uses Tesseract OCR for lightweight deployment.
    """

    extracted_text = ""

    try:

        # PDF file
        if file_path.lower().endswith(".pdf"):

            document = fitz.open(file_path)

            for page_number in range(len(document)):

                page = document[page_number]

                pix = page.get_pixmap(
                    dpi=200,
                    alpha=False
                )

                temp_image = f"temp_page_{page_number}.png"

                pix.save(temp_image)

                try:
                    image = Image.open(temp_image)

                    text = pytesseract.image_to_string(
                        image,
                        config="--psm 6"
                    )

                    if text:
                        extracted_text += text + "\n"

                finally:

                    if os.path.exists(temp_image):
                        os.remove(temp_image)

            document.close()

        # Image file
        else:

            image = Image.open(file_path)

            extracted_text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

    except Exception as e:

        print("OCR Error:", e)
        return ""

    return extracted_text.strip()
