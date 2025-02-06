from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME
from indexing import HybridIndexer
from sentence_transformers import CrossEncoder

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

indexer = HybridIndexer()

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6v2')

def rerank_results(query, retrieved_docs):

    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked_results]

def handle_query(query):

    query_embedding = llm.embed_query()
    search_results = indexer.search(query_embedding, query, top_k=5, alpha=0.7)

    retrieved_docs = [doc for doc, _ in search_results]
    reranked_docs = rerank_results(query, retrieved_docs)

    retriever = FAISS.load_local("faiss_index_path", llm)

    qa_chain = RetrievalQA(retriever=retriever, llm=llm)

    return qa_chain.run(reranked_docs[0])