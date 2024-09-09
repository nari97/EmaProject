from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.documents import Document
from langchain.storage import LocalFileStore


def store_documents(processed_documents: Dict[str, List[Document]]):
    store = LocalFileStore("./cache/")
    core_embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model,
                                                      store,
                                                      namespace="sentence-transformers/all-MiniLM-L6-v2")
    all_docs = [doc for docs in processed_documents.values() for doc in docs]
    vectorstore = FAISS.from_documents(all_docs, embedder)

    return vectorstore
