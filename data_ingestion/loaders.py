from typing import List

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders import DirectoryLoader
import os


class RecursiveFileSystemLoader(BaseLoader):
    def __init__(self, path):
        self.path = path

    def get_recursive_paths(self):
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                file_paths.append(os.path.join(dirpath, filename))
        return file_paths

    def load(self) -> List[Document]:
        paths = self.get_recursive_paths()
        loader = UnstructuredLoader(paths, chunking_strategy="by_title", overlap_all=True, max_characters=1000, show_progress=True)
        docs = loader.load()

        # for doc in docs:
        #     doc.page_content = doc.metadata["filename"] + "\n" + doc.page_content

        return docs


def load_all_data():
    loaders = {"unstructured": DirectoryLoader(r"C:\Users\nk1581\Downloads\bbc\tech/")}

    all_documents = {}
    for source, loader in loaders.items():
        all_documents[source] = loader.load()

    return all_documents
