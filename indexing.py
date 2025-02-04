import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridIndexer:
    def __init__(self):
        self.faiss_index = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def build_faiss_index(self, dense_embeddings):
        dimension = dense_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(dense_embeddings)
        return self.faiss_index

    def build_tfidf_index(self, texts):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return self.tfidf_matrix
    
    def search(self, query_embedding, query_text, top_k=5, alpha=0.7):
        _, dense_indices = self.faiss_index.search(query_embedding, top_k)

        sparse_query_vector = self.tfidf_vectorizer.transform([query_text])
        sparse_scores = np.dot(self.tfidf_matrix, sparse_query_vector.T).toarray().flatten()
        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]

        hybrid_results = {}
        for idx in range(top_k):
            dense_score = 1 / (1 + _[0][idx])
            sparse_score = sparse_scores[sparse_indices[idx]]
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score
            hybrid_results[sparse_indices[idx]] = combined_score

        sorted_results = sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results    

