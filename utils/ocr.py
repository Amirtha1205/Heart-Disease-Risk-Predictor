import os

# Limit CPU thread usage for low-memory Render deployment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

# Limit PyTorch CPU threads
torch.set_num_threads(1)

import easyocr
import fitz


# Initialize EasyOCR in CPU-only mode
reader = easyocr.Reader(
    ['en'],
    gpu=False
)


def read_report(file_path):
    """
    Extract text from image or PDF medical reports.

    PDF files:
        Each page is converted into an image and processed using EasyOCR.

    Image files:
        The image is directly processed using EasyOCR.

    Returns:
        Extracted OCR text as a string.
    """

    extracted_text = ""

    # -------------------------------------------------
    # PDF REPORT
    # -------------------------------------------------
    if file_path.lower().endswith(".pdf"):

        pdf_document = fitz.open(file_path)

        try:
            for page_number in range(len(pdf_document)):

                page = pdf_document[page_number]

                # Convert PDF page to image
                pix = page.get_pixmap(dpi=300)

                # Temporary image for OCR
                temp_image = f"temp_page_{page_number}.png"

                pix.save(temp_image)

                try:
                    # Perform OCR using CPU
                    result = reader.readtext(
                        temp_image,
                        detail=0
                    )

                    if result:
                        extracted_text += "\n".join(result)
                        extracted_text += "\n"

                finally:
                    # Delete temporary image
                    if os.path.exists(temp_image):
                        os.remove(temp_image)

        finally:
            pdf_document.close()

    # -------------------------------------------------
    # IMAGE REPORT
    # -------------------------------------------------
    else:

        result = reader.readtext(
            file_path,
            detail=0
        )

        if result:
            extracted_text = "\n".join(result)

    return extracted_text
