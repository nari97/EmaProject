from langchain_community.retrievers import BM25Retriever


def get_bm25_store(processed_documents):
    docs = processed_documents["unstructured"]

    all_docs = []

    for doc in docs:
        all_docs.append(doc)

    retriever = BM25Retriever.from_documents(all_docs, k=3)
    return retriever
