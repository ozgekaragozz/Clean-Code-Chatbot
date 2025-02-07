from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME
from indexing import HybridIndexer
from sentence_transformers import CrossEncoder
import json
import re

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

indexer = HybridIndexer()

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6v2')

FEEDBACK_FILE = "feedback_log.json"

def expand_query(query):

    prompt = f"Make the query more descriptive: {query}"
    expanded_query = llm.predict(prompt)

    return expanded_query

def rerank_results(query, retrieved_docs):

    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked_results]

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

    reranked_docs = rerank_results(expanded_query, retrieved_docs)

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

def save_feedback(query, response, feedback):
    
    feedback_entry = {
        "query": query,
        "response": response,
        "feedback": feedback
    }

    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(feedback_entry)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return "Your feedback has been recorded."          