from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict


def process_documents(documents: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    processed_documents = {}
    for source, docs in documents.items():
        split_docs = text_splitter.split_documents(docs)

        for doc in split_docs:
            doc.metadata["source"] = source
        processed_documents[source] = split_docs

    return processed_documents
