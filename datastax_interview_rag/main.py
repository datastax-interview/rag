import os
from llama_index.embeddings.openai import OpenAIEmbedding
from astrapy import DataAPIClient
from dotenv import load_dotenv

from rag_system import RAGSystem
import pymupdf


load_dotenv()

def read_pdf(file_path):
    doc = pymupdf.open(file_path) # open a document
    doc_text = ""
    for page in doc: # iterate the document pages
        text = page.get_text()
        doc_text += "/n" + text
    return doc_text

def read_directory(directory_path):
    documents = []
    def process_directory(path):
        for entry in os.scandir(path):
            if entry.is_file() and entry.name.lower().endswith('.pdf'):
                document_text = read_pdf(entry.path)
                documents.append((document_text, {"file_name": entry.name}))
            elif entry.is_dir():
                process_directory(entry.path)
    
    process_directory(directory_path)
    return documents

def main():

    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

    documents, metadatas = zip(*read_directory("./finance-bench/source_files"))
    documents, metadatas = documents[:2], metadatas[:2]
    print(f"Number of loaded documents: {len(documents)}")

    rag_system = RAGSystem(embed_model, "patronus_ai_finance_bench")

    rag_system.insert_documents(documents, metadatas)
    response = rag_system.query("What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.")
    print(response)
    # astra_db_store = AstraDBVectorStore(
    #     token=os.getenv("DATASTAX_TOKEN"),
    #     api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    #     collection_name="default_collection",
    #     embedding_dimension=1536,
    # )
    

    # storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

    # index = VectorStoreIndex.from_documents(
    #     documents, storage_context=storage_context, embed_model=embed_model
    # )

    # query_engine = index.as_query_engine()
    # response = query_engine.query("Why did the author choose to work on AI?")

    # print(response.response)


if __name__ == "__main__":
    main()
