import pdfplumber
import fitz
import os
from PIL import Image
from pytesseract import image_to_string

        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text   

def extract_images_from_pdf(LLM\pdfs\[PROGRAMMING][Clean Code by Robert C Martin].pdf, images):
    if not os.path.exists(images):
        os.makedirs(images)

    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_extension = base_image["ext"]
            image_filename = f"page_{page_num + 1}_image_{img_index + 1}.{image_extension}"

            with open(os.path.join(images, image_filename), "wb") as img_file:
                img_file.write(image_bytes)

    print(f"Görseller {images} klasörüne kaydedildi.")

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return image_to_string(image)