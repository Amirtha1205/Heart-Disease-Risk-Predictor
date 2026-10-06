import easyocr
import fitz
import os
import re

reader = easyocr.Reader(['en'], gpu=False)


def clean_text(text):
    text = text.replace("|", " ")
    text = text.replace("_", " ")
    text = text.replace(";", ":")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_report(filepath):

    extracted_text = ""

    extension = os.path.splitext(filepath)[1].lower()

    # PDF
    if extension == ".pdf":

        pdf = fitz.open(filepath)

        for page_no in range(len(pdf)):

            page = pdf.load_page(page_no)

            pix = page.get_pixmap(dpi=300)

            image_path = f"temp_page_{page_no}.png"

            pix.save(image_path)

            result = reader.readtext(
                image_path,
                detail=0,
                paragraph=True
            )

            extracted_text += "\n".join(result)
            extracted_text += "\n"

            if os.path.exists(image_path):
                os.remove(image_path)

        pdf.close()

    # Image
    else:

        result = reader.readtext(
            filepath,
            detail=0,
            paragraph=True
        )

        extracted_text = "\n".join(result)

    extracted_text = clean_text(extracted_text)

    print("=" * 60)
    print("OCR OUTPUT")
    print("=" * 60)
    print(extracted_text)
    print("=" * 60)

    return extracted_text