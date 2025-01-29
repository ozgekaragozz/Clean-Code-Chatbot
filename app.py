from streamlit as st
from parser import extract_text_from_pdf
from cleaner import clean_text
from embedding import embed_with_multiple_models
from indexing import build_faiss_index
from query_handler import handle_query

@st.cache_data
def initialize_system(pdfs\clean_code.pdf):
    text  = extract_text_from_pdf(pdfs\clean_code.pdf)
    cleaned_text = clean_text(text)
    segments = segment_text(cleaned_text)

    image_folder = "images"
    image_files = extract_images_from_pdf(pdfs\clean_code.pdf, image_folder)
    image_texts = [extract_text_from_image(os.path.join(image_folder, img)) for img in image_files]

    all_texts = segments + image_texts
    embeddings = embed_text(all_texts)
    index = build_faiss_index(embeddings)
    return index, all_texts

pdf_path = "./pdfs/clean_code.pdf"
index, segments = initialize_system(pdfs\clean_code.pdf)

st.title("Clean Code Chatbot")

user_query = st.text_input("Enter the your question.")
if st.button():
    if user_query:
        response  = handle_query(user_query, index, segments)
        st.write("Answer: ")
        st.write(response)

    else:
        st.warning("Please enter the your question.")    