from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridEmbedder:
    def __init__(self, model_names=None):
        if model_names is None:
            models = ['all-MiniLM-L6-v2', 'paraphrase-mpnet-base-v2', 'distiluse-base-multilingual-cased-v2']

        self.models = {name: SentenceTransformer(name) for name in model_names}
        self.vectorizer = TfidfVectorizer()

    def embed_with_dense_models(self, texts):
        dense_results = {}
        for model_name, model in self.models.items():
            embeddings = model.encode(texts)
            dense_results[model_name] = embeddings
        return dense_results

    def embed_with_sparse_model(self, texts):
        sparse_embeddings = self.vectorizer.fit_transform(texts)
        return sparse_embeddings.toarray()

    def hybrid_embedding(self, texts):
        dense_results = self.embed_with_dense_models(texts)
        sparse_results = self.embed_with_sparse_model(texts)   

        return = {
            "dense": dense_results,
            "sparse": sparse_results
        }
