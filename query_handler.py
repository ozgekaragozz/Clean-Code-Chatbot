from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT_NAME

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

def handle_query(query, index, texts):
    retriever = FAISS(index, texts)
    qa_chain = RetrievalQA(retriever=retriever, llm=llm)
    return qa_chain.run(query)