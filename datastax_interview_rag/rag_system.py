import os
from typing import Dict, Any, List, Optional
from astrapy import DataAPIClient

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from tqdm import tqdm

from llm_client import OpenAIClient

DEFAULT_PROMPT = "You are a assistant meant to perform retrieval augmented generation. You will be given a query and a set of context documents. You should use the context documents to answer the query. \
Try to rely only on the context documents to answer the question. When necessary, perform reasoning to answer the question but show your reasoning steps clearly and citing the necessary information from the context documents.\
If you cannot answer the question based on the context documents, say so. "

class RAGSystem:
    def __init__(self, embedding_model: BaseEmbedding, collection_name: str, drop_collection: bool = True, system_prompt: str = DEFAULT_PROMPT, model_name: str = "gpt-4o-mini"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.summary_collection_name = f"{collection_name}_summary"
        self.llm_client = OpenAIClient(model_name=model_name, system_prompt=system_prompt)

        self.vector_dimension = len(embedding_model.get_query_embedding("test"))
        print(f"Vector dimension: {self.vector_dimension}")
        
        self.client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
        self.database = self.client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
        print(f"Collections: {self.database.list_collection_names()}")
        if collection_name in self.database.list_collection_names():
            if drop_collection:
                print(f"Collection {collection_name} already exists, deleting...")
                self.database.drop_collection(collection_name)
                self.database.drop_collection(self.summary_collection_name)
                self.collection = self.database.create_collection(
                    self.collection_name,
                    dimension=self.vector_dimension,
                    metric="cosine",
                    check_exists=True
                )
                self.summary_collection = self.database.create_collection(
                    self.summary_collection_name,
                    dimension=self.vector_dimension,
                    metric="cosine",
                    check_exists=True
                )
            else:
                print(f"Collection {collection_name} already exists, loading collection...")
                self.collection = self.database.get_collection(collection_name)
                self.summary_collection = self.database.get_collection(self.summary_collection_name)
        else:
            print(f"Collection {collection_name} does not exist, creating...")
            self.collection = self.database.create_collection(
                self.collection_name,
                dimension=self.vector_dimension,
                metric="cosine",
                check_exists=True
            )
            self.summary_collection = self.database.create_collection(
                self.summary_collection_name,
                dimension=self.vector_dimension,
                metric="cosine",
                check_exists=True
            )
        self.chunker = SentenceSplitter(chunk_size=700, chunk_overlap=300)

    def insert_document(self, text: str, filename: str, metadata: Dict[str, Any] = None) -> None:
        metadata = metadata if metadata is not None else {}
        if "file_name" not in metadata:
            metadata["file_name"] = filename
        chunks = self.chunker.split_text(text)
        documents = []
        embeddings = self.embedding_model.get_text_embedding_batch(chunks, show_progress=False)
        first_5_chunks = "\n".join(chunks[:5])
        doc_summary = self.llm_client.generate_llm_completion(f"Summarize the following document in one paragraph: {first_5_chunks}\n\nSummary: ")
        for chunk, embedding in zip(chunks, embeddings):
            document = metadata.copy()
            document["text"] = doc_summary + "\n\n" + chunk
            document["original_text"] = chunk
            document["$vector"] = embedding
            documents.append(document)
        self.collection.insert_many(documents)
        # now add summary to summary collection
        document = metadata.copy()
        document["text"] = doc_summary
        document["$vector"] = self.embedding_model.get_text_embedding(doc_summary)
        self.summary_collection.insert_one(document)


    def insert_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        for document, metadata in tqdm(zip(documents, metadatas), total=len(documents)):
            self.insert_document(document, metadata)

    def _gather_contexts_traditional_rag(self, query_text: str, num_context_docs: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.get_query_embedding(query_text)
        find_options = {
            "sort": {"$vector": query_embedding},
            "limit": num_context_docs,
            "include_similarity": True
        }
        
        results = list(self.collection.find(**find_options))
        return "\n".join([result["text"] for result in results])
    
    def _gather_contexts_two_stage_rag(self, query_text: str, num_context_docs: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.get_query_embedding(query_text)
        find_options = {
            "sort": {"$vector": query_embedding},
            "limit": 1,
            "include_similarity": True
        }
        results = list(self.summary_collection.find(**find_options))
        top_document = results[0]["file_name"]
        document_summary = "Document summary: " + results[0]["text"] + "\n\n"
        # limit search to only top document's chunks
        find_options = {
            "sort": {"$vector": query_embedding},
            "limit": num_context_docs,
            "include_similarity": True,
            "filter": {"file_name": top_document}
        }
        results = list(self.collection.find(**find_options))
        return document_summary + "\n".join([result["original_text"] for result in results])
    
    def _reflection(self, query_text: str, answer: str) -> bool:
        input_string = "Query: " + query_text + "\n\nProvided answer: " + answer + "\n\n Does the provided answer provide useful information about the query? Return json with the key anwer and only the words yes or no\
        Return no if the answer does not provide useful information about the query. "
        response = self.llm_client.generate_json_completion(input_string)
        # If LLM does not explicitly answer no, then we'll just use traditional pipeline.
        if "answer" not in response or response["answer"] != "no":
            return True
        else:
            return False

    def query(self, query_text: str, num_context_docs: int = 10) -> List[Dict[str, Any]]:
        contexts = self._gather_contexts_traditional_rag(query_text, num_context_docs)
        input_string = "Query: " + query_text + "\n\nContexts: " + contexts + "\n\nAnswer: "
        response = self.llm_client.generate_llm_completion(input_string)
        if self._reflection(query_text, response):
            return response
        else:
            contexts = self._gather_contexts_two_stage_rag(query_text, num_context_docs)
            input_string = "Query: " + query_text + "\n\nContexts: " + contexts + "\n\nAnswer: "
            response = self.llm_client.generate_llm_completion(input_string)
            return response

    def clear(self) -> None:
        self.database.delete_collection(self.collection_name)
        self.collection = self.database.create_collection(
            self.collection_name,
            dimension=1536,
            metric="cosine",
            check_exists=False
        )

