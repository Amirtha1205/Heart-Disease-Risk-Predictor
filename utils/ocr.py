import os

# Reduce CPU memory/thread usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

import easyocr
import fitz


# Initialize EasyOCR only once
reader = easyocr.Reader(
    ['en'],
    gpu=False,
    verbose=False
)


def read_report(file_path):
    """
    Extract text from an image or PDF medical report.
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

                    result = reader.readtext(
                        temp_image,
                        detail=0,
                        paragraph=True
                    )

                    if result:
                        extracted_text += "\n".join(result) + "\n"

                finally:

                    if os.path.exists(temp_image):
                        os.remove(temp_image)

            document.close()

        # Image file
        else:

            result = reader.readtext(
                file_path,
                detail=0,
                paragraph=True
            )

            if result:
                extracted_text = "\n".join(result)

    except Exception as e:

        print("OCR Error:", e)
        return ""

    return extracted_text.strip()
