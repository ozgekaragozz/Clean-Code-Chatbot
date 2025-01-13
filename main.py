from extract_text_from_pdf import extract_text_from_pdf
from cleaner import clean_text, segment_text
from embedding import embed_text
from indexing import build_faiss_index
from query_handler import handle_query

pdf_path = "./pdfs/clean_code.pdf"
print("PDF is being processed.")
text = extract_text_from_pdf(pdf_path)

print("Text is cleaning and segmenting.")
cleaned_text = clean_text(text)
segments  =segment_text(cleaned_text)

print("Embedding is creating.")
embedding = embed_text(segments)

print("Index of FAISS is creating.")
index = build_faiss_index(embeddings)

while True:
    query = input("Please enter the your question. If you want to exit, press q. ").strip()
    if query.lower() == 'q':
        print("Exit is in progress.")
        break
    
    response = handle_query(query, index, segments)
    print("\nAnswer")
    print(response)