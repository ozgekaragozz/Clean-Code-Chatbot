from parser import PDFProcessor
from cleaner import clean_text, segment_text
from embedding import embed_with_multiple_models
from indexing import build_faiss_index
from query_handler import handle_query

pdf_path = "./pdfs/clean_code.pdf"
pdf_processor = PDFProcessor(pdf_path)

print("PDF is being processed.")
text = pdf_processor.extract_text()

print("Extracting images from PDF...")
image_output_folder = "./extracted_images"
pdf_processor.extract_images(image_output_folder)

print("Text is cleaning and segmenting.")
cleaned_text = clean_text(text)
segments = segment_text(cleaned_text)

print("Embedding is creating.")
embeddings = embed_with_multiple_models(segments)

print("Index of FAISS is creating.")
index = build_faiss_index(embeddings)

while True:
    query = input("Please enter your question. If you want to exit, press 'q': ").strip()
    if query.lower() == 'q':
        print("Exiting...")
        break

    response = handle_query(query, index, segments)
    print("\nAnswer:")
    print(response)
