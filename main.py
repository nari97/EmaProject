from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from data_ingestion import loaders
from data_processing import processor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from storage import vector_store, bm_25_store
from prompts import prompts
import warnings

warnings.filterwarnings("ignore")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main(retrieval="hybrid"):
    print("Loading data...")
    docs = loaders.load_all_data()
    print(f"Loaded {len(docs['unstructured'])} documents")
    print("Processing data...")
    processed_docs = processor.process_documents(docs)
    print(f"Split data into {len(processed_docs['unstructured'])} chunks")
    print("Storing data...")
    vectorstore = vector_store.store_documents(processed_docs)
    vector_ret = vectorstore.as_retriever(search_kwargs={"k": 5})

    if retrieval == "hybrid":
        bm25_retriever = bm_25_store.get_bm25_store(processed_docs)
        ensemble_retriever = EnsembleRetriever(retrievers=[vector_ret, bm25_retriever],
                                               weights=[0.3, 0.7])
        retriever = ensemble_retriever
    else:
        retriever = vector_ret

    compressor = FlashrankRerank(top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    llm1 = ChatOllama(model="mistral")
    llm2 = ChatOllama(model="mistral")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    rephrase_prompt = PromptTemplate.from_template(prompts.return_rephrase_prompt())

    rephrase_chain = (
            {"chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
             "question": RunnablePassthrough()}
            | rephrase_prompt
            | llm1
            | StrOutputParser()
    )

    answer_prompt = hub.pull("rlm/rag-prompt")

    answer_chain = (
            {"context": compression_retriever | format_docs, "question": rephrase_chain}
            | answer_prompt
            | llm2
            | StrOutputParser()
    )

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        rephrased_question = rephrase_chain.invoke(question)
        print("Rephrased question:", rephrased_question)
        response = answer_chain.invoke(question)
        print("AI:", response)

        # Update the memory with the new interaction
        memory.save_context({"input": rephrased_question}, {"output": response})


if __name__ == "__main__":
    main()
