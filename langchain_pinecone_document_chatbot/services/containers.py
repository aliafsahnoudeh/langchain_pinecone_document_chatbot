"""Containers module."""

from dependency_injector import containers, providers
from langchain_pinecone_document_chatbot.services.chatbot.document_chain import (
    DocumentChain,
)
from langchain_pinecone_document_chatbot.services.chatbot.pinecone_service import (
    PineconeService,
)
from langchain_pinecone_document_chatbot.services.logger import logger

from . import feedback_service
from .chatbot.main import Chatbot


class Container(containers.DeclarativeContainer):
    logger.info({"Container": "initializing"})

    wiring_config = containers.WiringConfiguration(
        modules=[
            "langchain_pinecone_document_chatbot.web.api.chat.views",
        ],
    )

    config = providers.Configuration(yaml_files=["config.yml"])

    pinecone_service = providers.Singleton(PineconeService)

    document_chain = providers.Singleton(
        DocumentChain,
        pinecone_service=pinecone_service,
    )

    chat_bot = providers.Singleton(
        Chatbot,
        document_chain=document_chain,
    )

    feedback_service = providers.Singleton(feedback_service.FeedbackService)

    logger.info({"Container": "initialized"})
