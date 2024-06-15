# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'
import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings


def save_to_vector_db(message):
    docs = [f"{message}"]
    os.environ["AZURE_OPENAI_API_KEY"] =  os.getenv("azure_api_key")
    embeddings=AzureOpenAIEmbeddings(deployment="embedding-ada-002",
                                model= "text-embedding-ada-002",
                                azure_endpoint="https://tensurfbrain1.openai.azure.com/",
                                openai_api_type="azure",
                                chunk_size=100000)
    db = FAISS.from_texts(docs, embeddings)
    db.save_local("faiss_index")


def retrieve_from_vector_db(query):
    embeddings=AzureOpenAIEmbeddings(deployment="embedding-ada-002",
                                model= "text-embedding-ada-002",
                                azure_endpoint="https://tensurfbrain1.openai.azure.com/",
                                openai_api_type="azure",
                                chunk_size=100000)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    docs = retriever.invoke(query)
    return docs[0].page_content
