import os
import streamlit as st
from parser import PDFProcessor, ImageProcessor
from cleaner import clean_text, segment_text
from embedding import HybridEmbedder
from indexing import HybridIndexer
from query_handler import handle_query, save_feedback

@st.cache_data
def initialize_system(pdf_path):

    pdf_processor = PDFProcessor(pdf_path)
    text  = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)

    try: 
        from cleaner import segment_text
        segments = segment_text(cleaned_text)

    except ImportError:
        segments = [cleaned_text]

    image_folder = "images"
    os.makedirs(image_folder, exist_ok=True)

    pdf_processor.extract_images(image_folder)
    image_texts = [ImageProcessor.extract_text(os.path.join(image_folder, img)) for img in os.listdir(image_folder)]

    all_texts = segments + image_texts

    embedder = HybridEmbedder()
    embeddings = embedder.hybrid_embedding(all_texts)

    indexer = HybridIndexer()
    indexer.build_faiss_index(embeddings["dense"]['all-MiniLM-L6-v2'])
    indexer.build_tfidf_index(all_texts)

    return indexer, all_texts

pdf_path = "./pdfs/clean_code.pdf"
indexer, segments = initialize_system(pdf_path)

st.title("Clean Code Chatbot")

user_query = st.text_input("Enter the your question.")

if st.button():
    
    if user_query:
        response  = handle_query(user_query, index, segments)
        st.write("Answer: ")
        st.write(response)

        feedback = st.radio("Was this answer helpful?", ["Yes", "No"])
        if st.button("Save the feedback"):
            save_feedback(user_query, response, feedback)
            st.success("Your feedback has been saved.")

    else:
        st.warning("Please enter the your question.")    