import os
import fitz
import pytesseract
from PIL import Image


def read_report(file_path):
    """
    Lightweight OCR for image and PDF medical reports.
    Designed for low-memory deployment.
    """

    extracted_text = ""

    try:

        # -----------------------------
        # PDF FILE
        # -----------------------------
        if file_path.lower().endswith(".pdf"):

            document = fitz.open(file_path)

            for page_number in range(len(document)):

                page = document[page_number]

                # Lower DPI to reduce memory usage
                pix = page.get_pixmap(
                    dpi=100,
                    alpha=False
                )

                temp_image = f"/tmp/temp_page_{page_number}.png"

                pix.save(temp_image)

                try:

                    image = Image.open(temp_image)

                    # Convert to grayscale
                    image = image.convert("L")

                    # Reduce image size if too large
                    max_width = 1600

                    if image.width > max_width:

                        ratio = max_width / image.width

                        new_height = int(
                            image.height * ratio
                        )

                        image = image.resize(
                            (max_width, new_height)
                        )

                    text = pytesseract.image_to_string(
                        image,
                        config="--psm 6"
                    )

                    if text:
                        extracted_text += text + "\n"

                    image.close()

                finally:

                    if os.path.exists(temp_image):
                        os.remove(temp_image)

            document.close()

        # -----------------------------
        # IMAGE FILE
        # -----------------------------
        else:

            image = Image.open(file_path)

            image = image.convert("L")

            max_width = 1600

            if image.width > max_width:

                ratio = max_width / image.width

                new_height = int(
                    image.height * ratio
                )

                image = image.resize(
                    (max_width, new_height)
                )

            extracted_text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            image.close()

    except Exception as e:

        print("OCR Error:", e)
        return ""

    return extracted_text.strip()