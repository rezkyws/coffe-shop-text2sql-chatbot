import logging

import asyncpg
from fastapi import Depends

from app.core.database import (
    get_coffee_db_dependency,
    get_db_dependency,
)
from app.core.llm_client import LLMClient
from app.core.memory_manager import MemoryManager
from app.core.query_processor import QueryProcessor
from app.core.vector_store import VectorStoreManager
from app.services.clarification_service import ClarificationService
from app.services.retrieval_service import RetrievalService
from app.services.schema_extractor import SchemaExtractor
from app.services.sql_executor import SQLExecutor

logger = logging.getLogger(__name__)


# Database connection dependencies
async def get_db_connection():
    """Get a connection from the main database pool."""
    try:
        return await get_db_dependency()
    except Exception as e:
        logger.error(f"Failed to get main DB connection: {e}")
        raise


async def get_coffee_db_connection():
    """Get a connection from the coffee shop database pool."""
    try:
        return await get_coffee_db_dependency()
    except Exception as e:
        logger.error(f"Failed to get coffee DB connection: {e}")
        raise


# Singleton LLM client (reuse across requests)
_llm_client_instance = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance


async def get_memory_manager(
    db: asyncpg.Connection = Depends(get_db_connection),
    llm_client: LLMClient = Depends(get_llm_client),
) -> MemoryManager:
    """Dependency for memory manager."""
    return MemoryManager(db, llm_client)


async def get_vector_store(
    db: asyncpg.Connection = Depends(get_db_connection),
) -> VectorStoreManager:
    """Dependency for vector store manager."""
    return VectorStoreManager(db)


async def get_query_processor(
    llm_client: LLMClient = Depends(get_llm_client),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> QueryProcessor:
    """Dependency for query processor."""
    return QueryProcessor(llm_client, memory_manager, vector_store)


async def get_sql_executor(
    coffee_db: asyncpg.Connection = Depends(get_coffee_db_connection),
    main_db: asyncpg.Connection = Depends(get_db_connection),
) -> SQLExecutor:
    """Dependency for SQL executor."""
    return SQLExecutor(coffee_db, main_db)


async def get_schema_extractor(
    coffee_db: asyncpg.Connection = Depends(get_coffee_db_connection),
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> SchemaExtractor:
    """Dependency for schema extractor."""
    return SchemaExtractor(coffee_db, vector_store)


async def get_retrieval_service(
    vector_store: VectorStoreManager = Depends(get_vector_store),
    llm_client: LLMClient = Depends(get_llm_client),
) -> RetrievalService:
    """Dependency for retrieval service."""
    return RetrievalService(vector_store, llm_client)


async def get_clarification_service(
    llm_client: LLMClient = Depends(get_llm_client),
    memory_manager: MemoryManager = Depends(get_memory_manager),
) -> ClarificationService:
    """Dependency for clarification service."""
    return ClarificationService(llm_client, memory_manager)
