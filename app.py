from streamlit as st
from extract_text_from_pdf import extract_text_from_pdf
from cleaner import clean_text
from embedding import embed_text
from indexing import build_faiss_index
from query_handler import handle_query

@st.cache_data
def initialize_system(pdf_path):
    text  = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    segments = segment_text(cleaned_text)
    embeddings = embed_text(segments)
    index = build_faiss_index(embeddings)
    return index, segments

pdf_path = "./pdfs/clean_code.pdf"
index, segments = initialize_system(pdf_path)

st.title("Clean Code Chatbot")

user_query = st.text_input("Enter the your question.")
if st.button():
    if user_query:
        response  = handle_query(user_query, index, segments)
        st.write("Answer: ")
        st.write(response)

    else:
        st.warning("Please enter the your question.")    