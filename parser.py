import pdfplumber
import fitz
import os
from PIL import Image
from pytesseract import image_to_string

pdf_path = r"C:\Users\ozgek\OneDrive\Masaüstü\LLM\pdfs\clean_code.pdf"
image_output_folder = r"C:\Users\ozgek\OneDrive\Masaüstü\LLM\images"

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text

    def extract_images(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pdf_document = fitz.open(self.pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_extension = base_image["ext"]
                image_filename = f"page_{page_num + 1}_image_{img_index + 1}.{image_extension}"

                with open(os.path.join(output_folder, image_filename), "wb") as img_file:
                    img_file.write(image_bytes)

        print(f"Görseller {output_folder} klasörüne kaydedildi.")

class ImageProcessor:
    @staticmethod
    def extract_text(image_path):
        image = Image.open(image_path)
        return image_to_string(image)

if __name__ == "__main__":
    pdf_path = "clean_code.pdf"
    image_output_folder = "extracted_images"

    pdf_processor = PDFProcessor(pdf_path)
    text = pdf_processor.extract_text()
    print("Extracted Text:\n", text)

    pdf_processor.extract_images(image_output_folder)

    # OCR İşlemi
    for image_file in os.listdir(image_output_folder):
        image_path = os.path.join(image_output_folder, image_file)
        extracted_text = ImageProcessor.extract_text(image_path)
        print(f"OCR Extracted Text from {image_file}:\n", extracted_text)
