from sentence_transformers import SentenceTransformer

def embed_with_multiple_models(text_list, model_names):
    results = {}
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(text_list)
        results[model_name] = embeddings
    return results

models = ['all-MiniLM-L6-v2', 'paraphrase-mpnet-base-v2', 'distiluse-base-multilingual-cased-v2']

embeddings_results = embed_with_multiple_models(texts, models)
for model_name, embeddings in embeddings_results.items():
    print(f"Model: {model_name}, Embedding Shape: {len(embeddings)} x {len(embeddings[0])}")
