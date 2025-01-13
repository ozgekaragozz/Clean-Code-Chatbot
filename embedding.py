from sentence_transformers import SentenceTransformer

def embed_text(text_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text_list)