from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME
from indexing import HybridIndexer
from sentence_transformers import CrossEncoder, SentenceTransformer
import json
import re
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

indexer = HybridIndexer()

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6v2')

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

DB_FILE = "feedback.db"

def init_db():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback(7
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def save_feedback(query, response, feedback):

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (query, response, feedback)
        VALUES (?, ?, ?)
    ''', (query, response, feedback))

    conn.commit()
    conn.close()
    return "Your feedback has been recorded."  

def analyze_feedback():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT query, feedback, COUNT(*) as count
        FROM feedback
        GROUP BY query, feedback
        ORDER BY count DESC
    ''')

    results = cursor.fetchall()
    conn.close()

    report = "Analysis of Feedback:\n"
    for row in results:
        report += f"Soru: {row[0]}\nGeri Bildirim: {row[1]}\nTekrar Sayısı: {row[2]}\n\n"

    return report 

def get_feedback_data():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT query, feedback, timestamp FROM feedback")    
    data = cursor.fetchall()
    conn.close()

    return pd.DataFrame(data, columns=["query", "feedback", "timestamp"])

def plot_feedback_distribution():
    
    df = get_feedback_data()

    if df.empty:
        print("There is no feedback data yet.")
        return

    feedback_counts = df["feedback"].value_counts

    plt.figure(figsize=(8, 5))
    feedback_counts.plot(kind="bar", color=["green", "red"])
    plt.xlabel("Type of Feedback")
    plt.ylabel("Piece")
    plt.title("Distribution of Feedback")
    plt.xticks(rotation=45)
    plt.show()

def plot_most_problematic_queries():

    df = get_feedback_data()
    
    if df.empty:
        print("There is no feedback data yet.")
        return

    problematic_queries = df[df["feedback"] == "Wrong"].groupby("query").count().sort_values("feedback", ascending=False)
    
    plt.figure(figsize=(10, 5))
    problematic_queries["feedback"][:10].plot(kind="bar", color="red")
    plt.xlabel("Question")
    plt.ylabel("Number of Wrong Feedback")
    plt.title("Questions Most Commonly Answered Wrongly")
    plt.xticks(rotation=45)
    plt.show()

def plot_feedback_over_time():
    
    df = get_feedback_data()

    if df.empty:
        print("There is no feedback data yet.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    feedback_over_time = df.resample("D").count()["feedback"]

    plt.figure(figsize=(10, 5))
    feedback_over_time.plot(kind="line", marker="o", linestyle="-", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Number of Feedback")
    plt.title("Feedback Distribution Over Time ")
    plt.grid()
    plt.show()   

def expand_query(query):

    prompt = f"Make the query more descriptive: {query}"
    expanded_query = llm.predict(prompt)

    return expanded_query

def rerank_results(query, retrieved_docs):

    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked_results][:5]

def filter_top_docs_with_similarity(query, top_docs):

    query_embedding = similarity_model.encode([query])
    doc_embeddings = similarity_model.encode(top_docs)

    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    sorted_indices = np.argsort(similarities)[::-1][:3]

    return [top_docs[i] for i in sorted_indices]

def split_sentences(text):
   
    return re.split(r'(?<=[.!?]) +', text)

def verify_sentence_with_sources(sentence, sources):

    verification_prompt = f"""Can you verify the following sentence according to the given sources?

    Sentence: {sentence}
    Sources: {sources[0][:500]}

    If the sentence can be verified, write 'Verified'. 
    If this information is not found in the sources, write 'False Information'.
    """ 
    verification_result = llm.predict(verification_prompt)

    return verification_result.strip()

def handle_query(query):

    expanded_query = expand_query(query)

    query_embedding = llm.embed_query(expanded_query)
    search_results = indexer.search(query_embedding, expanded_query, top_k=5, alpha=0.7)

    retrieved_docs = [doc for doc, _ in search_results]

    if not retrieved_docs:
        return "**I don't have that information. Can you ask it in a different way?**"

    top_5_docs = rerank_results(expanded_query, retrieved_docs)
    
    final_docs = filter_top_docs_with_similarity(expanded_query, top_5_docs)

    retriever = FAISS.load_local("faiss_index_path", llm)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever.as_retriever(),
        memory=memory
    )

    response = conversation_chain({"question": query})

    sentences = split_sentences(response['answer'])
    verified_sentences = [f"{sentence} → {verify_sentence_with_sources(sentence, reranked_docs)}" for sentence in sentences]

    correct_count = sum(1 for s in verified_sentences if "Verified" in s)
    total_sentences = len(verified_sentences)
    accuracy = (correct_count / total_sentences) * 100 if total_sentences > 0 else 0 

    sources = [f"{doc[:100]}..." for doc in reranked_docs[:3]]

    markdown_response = f"""
    ###Response: 
    {'\n'.join(verified_sentences)}

    **Accuracy Score: {accuracy:.2f}%**

    ###Sources:
    {"\n".join(sources)}
    """
    return markdown_response

init_db()    
