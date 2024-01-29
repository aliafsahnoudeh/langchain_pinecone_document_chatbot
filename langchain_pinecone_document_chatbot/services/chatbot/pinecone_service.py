import time
import uuid

import pandas as pd  # type: ignore
from langchain.schema import Document
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone_document_chatbot.models.similarity_metric import (
    SimilarityMetric,
)
from langchain_pinecone_document_chatbot.services.logger import logger
from langchain_pinecone_document_chatbot.settings import settings
from pinecone import Pinecone, PodSpec  # type: ignore
from tqdm.auto import tqdm  # type: ignore


class PineconeService:
    pc: Pinecone

    def __init__(self):
        self.pinecone_api_key = settings.pinecone_api_key
        self.environment = "eu-west4-gcp"
        self.index_name = f"test-{settings.environment}"
        self.batch_size = 100
        self.text_field = "text"
        self.embed_model = OpenAIEmbeddings(
            api_key=settings.open_ai_api_key,
            model="text-embedding-ada-002",
        )
        self.pc = Pinecone(api_key=self.pinecone_api_key, environment=self.environment)

    def check_index_exists(self):
        logger.info(
            {
                "message": "check_index_exists",
                "service": "PineconeService",
                "method": "check_index_exists",
            },
        )
        indexes = self.pc.list_indexes()
        for index in indexes:
            if index.name == self.index_name:
                return True
        return False

    def create_index_if_not_exists(self):
        logger.info(
            {
                "message": "create_index_if_not_exists",
                "service": "PineconeService",
                "method": "create_index_if_not_exists",
            },
        )

        if not self.check_index_exists():
            spec = PodSpec(environment=self.environment)
            self.pc.create_index(
                self.index_name,
                dimension=1536,
                metric=SimilarityMetric.COSINE,
                spec=spec,
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

    def upsert_documents(self, data: pd.DataFrame):
        logger.info(
            {
                "message": "upsert_documents",
                "service": "PineconeService",
                "method": "upsert_documents",
            },
        )
        index = self.pc.Index(self.index_name)
        for i in tqdm(range(0, len(data), self.batch_size)):  # type: ignore
            i_end = min(len(data), i + self.batch_size)
            batch = data.iloc[i:i_end]
            ids = [str(uuid.uuid4()) for _ in range(len(batch))]
            texts = [x[0][1] for _, x in batch.iterrows()]  # type: ignore
            embeds = self.embed_model.embed_documents(texts)
            metadata = [
                {
                    "text": x[0][1],
                    "source": x[1][1]["source"],
                    "title": x[1][1]["title"],
                }
                for _, x in batch.iterrows()  # type: ignore
            ]
            index.upsert(vectors=zip(ids, embeds, metadata))
        logger.info(
            {
                "message": "documents successfully upserted",
                "service": "PineconeService",
                "method": "upsert_documents",
            },
        )

    def delete_vecotors(self):
        logger.info(
            {
                "message": "delete_vecotors",
                "service": "PineconeService",
                "method": "delete_vecotors",
            },
        )
        index = self.pc.Index(self.index_name)
        index.delete(deleteAll=True)

    def seed(self, documents: list[Document]):
        self.create_index_if_not_exists()
        self.delete_vecotors()
        data = pd.DataFrame(documents)
        self.upsert_documents(data)

    def get_index(self):
        return self.pc.Index(self.index_name)

    def search(self, query: str, top_k: int = 5):
        index = self.get_index()
        vectorstore = VectorStorePinecone(index, self.embed_model, text_key="text")
        return vectorstore.similarity_search(query, k=top_k)

    def get_retriever(self, search_kwarg):
        self.create_index_if_not_exists()
        index = self.get_index()
        vectorstore = VectorStorePinecone(index, self.embed_model, text_key="text")
        return vectorstore.as_retriever(
            search_type="mmr",  # similarity mmr
            search_kwargs=search_kwarg,
        )
