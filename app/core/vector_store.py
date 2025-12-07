import json
import logging
from typing import List, Optional

import asyncpg
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.models.schemas import SimilarQuery

settings = get_settings()
logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            # Disable parallel processing to avoid semaphore leaks
            import os
            import threading

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["OMP_NUM_THREADS"] = "1"

            # Thread-safe initialization
            with threading.Lock():
                if self._embedding_model is None:
                    logger.info("Loading embedding model...")
                    self._embedding_model = SentenceTransformer(
                        settings.EMBEDDING_MODEL,
                        device="cpu",  # Force CPU to avoid GPU memory issues
                    )
                    logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        return self._embedding_model

    def cleanup(self):
        if self._embedding_model is not None:
            # Clean up any multiprocessing resources
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self._embedding_model = None

    def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        try:
            if is_query:
                # Use query prompt for better query embeddings
                embedding = self.embedding_model.encode(
                    text, prompt_name="query", normalize_embeddings=True
                )
            else:
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Fallback to dummy embedding if model fails
            return [0.0] * settings.EMBEDDING_DIMENSION

    async def store_query_example(
        self,
        question: str,
        sql_query: str,
        metadata: Optional[dict] = None,
    ) -> None:
        # Check if vector column exists
        vector_available = False
        try:
            await self.conn.fetchval("SELECT embedding FROM query_vector_store LIMIT 1")
            vector_available = True
        except:
            pass

        if vector_available:
            embedding = self.generate_embedding(question)

            await self.conn.execute(
                """
                INSERT INTO query_vector_store
                (question, sql_query, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                question,
                sql_query,
                embedding,
                json.dumps(metadata or {}),
            )
        else:
            await self.conn.execute(
                """
                INSERT INTO query_vector_store
                (question, sql_query, metadata)
                VALUES ($1, $2, $3)
                """,
                question,
                sql_query,
                json.dumps(metadata or {}),
            )
        logger.info(f"Stored query example: {question[:50]}...")

    async def search_similar_queries(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[SimilarQuery]:
        top_k = top_k if top_k is not None else settings.VECTOR_SEARCH_TOP_K

        # Check if vector column exists and has proper vector data
        vector_available = False
        try:
            sample = await self.conn.fetchval("SELECT embedding FROM query_vector_store LIMIT 1")
            # Check if it's actual vector data (not dummy embedding of all zeros)
            vector_available = (
                sample
                and isinstance(sample, str)
                and sample.startswith("[")
                and "," in sample
                and not sample.startswith("[0,0,0")
            )  # Exclude dummy embeddings
        except:
            pass

        if vector_available:
            query_embedding = self.generate_embedding(query_text, is_query=True)

            # Use pgvector's cosine distance operator
            rows = await self.conn.fetch(
                """
                SELECT
                    question,
                    sql_query,
                    1 - (embedding <=> $1) AS similarity
                FROM query_vector_store
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                str(query_embedding),
                top_k,
            )

            return [
                SimilarQuery(
                    question=row["question"],
                    sql_query=row["sql_query"],
                    similarity=float(row["similarity"]),
                )
                for row in rows
            ]
        else:
            # Fallback: simple keyword matching
            query_lower = query_text.lower()

            rows = await self.conn.fetch(
                """
                SELECT
                    question,
                    sql_query,
                    0.5 AS similarity
                FROM query_vector_store
                ORDER BY created_at DESC
                LIMIT $1
                """,
                top_k,
            )

            # Simple keyword matching for relevance scoring
            results = []
            for row in rows:
                question = row["question"].lower()

                # Calculate simple relevance score
                score = 0.5  # base score
                query_words = query_lower.split()
                question_words = question.split()

                # Count matching words
                matches = len(set(query_words) & set(question_words))
                if matches > 0:
                    score += 0.3 * (matches / len(query_words))

                results.append(
                    SimilarQuery(
                        question=row["question"],
                        sql_query=row["sql_query"],
                        similarity=min(score, 1.0),
                    )
                )

            # Sort by relevance score
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:top_k]

    async def update_query_success(self, question: str) -> None:
        await self.conn.execute(
            """
            UPDATE query_vector_store
            SET success_count = success_count + 1, updated_at = NOW()
            WHERE question = $1
            """,
            question,
        )

    async def cache_schema_info(
        self, table_name: str, schema_info: dict, row_count: Optional[int] = None
    ) -> None:
        # Check if vector column exists
        vector_available = False
        try:
            await self.conn.fetchval("SELECT embedding FROM schema_cache LIMIT 1")
            vector_available = True
        except:
            pass

        if vector_available:
            # Create searchable description for embedding
            description = self._create_schema_description(table_name, schema_info)
            embedding = self.generate_embedding(description)

            # Check if exists and update/insert accordingly
            existing = await self.conn.fetchval(
                "SELECT table_name FROM schema_cache WHERE table_name = $1", table_name
            )

            if existing:
                await self.conn.execute(
                    """
                    UPDATE schema_cache
                    SET schema_info = $2, embedding = $3, row_count = $4, last_updated = NOW()
                    WHERE table_name = $1
                    """,
                    table_name,
                    json.dumps(schema_info),
                    str(embedding),
                    row_count,
                )
            else:
                await self.conn.execute(
                    """
                    INSERT INTO schema_cache (table_name, schema_info, embedding, row_count)
                    VALUES ($1, $2, $3, $4)
                    """,
                    table_name,
                    json.dumps(schema_info),
                    str(embedding),
                    row_count,
                )
        else:
            # Handle without vector column
            existing = await self.conn.fetchval(
                "SELECT table_name FROM schema_cache WHERE table_name = $1", table_name
            )

            if existing:
                await self.conn.execute(
                    """
                    UPDATE schema_cache
                    SET schema_info = $2, row_count = $3, last_updated = NOW()
                    WHERE table_name = $1
                    """,
                    table_name,
                    json.dumps(schema_info),
                    row_count,
                )
            else:
                await self.conn.execute(
                    """
                    INSERT INTO schema_cache (table_name, schema_info, row_count)
                    VALUES ($1, $2, $3)
                    """,
                    table_name,
                    json.dumps(schema_info),
                    row_count,
                )

    async def search_relevant_tables(self, query_text: str, top_k: int = 5) -> List[dict]:
        # Check if vector column exists and has proper vector data
        vector_available = False
        try:
            sample = await self.conn.fetchval("SELECT embedding FROM schema_cache LIMIT 1")
            # Check if it's actual vector data (not dummy embedding of all zeros)
            vector_available = (
                sample
                and isinstance(sample, str)
                and sample.startswith("[")
                and "," in sample
                and not sample.startswith("[0,0,0")
            )  # Exclude dummy embeddings
        except:
            pass

        if vector_available:
            query_embedding = self.generate_embedding(query_text, is_query=True)

            rows = await self.conn.fetch(
                """
                SELECT
                    table_name,
                    schema_info,
                    row_count,
                    1 - (embedding <=> $1) AS similarity
                FROM schema_cache
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                str(query_embedding),
                top_k,
            )

            return [
                {
                    "table_name": row["table_name"],
                    "schema_info": json.loads(row["schema_info"])
                    if isinstance(row["schema_info"], str)
                    else row["schema_info"],
                    "row_count": row["row_count"],
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ]
        else:
            # Fallback: return all tables (simpler approach)
            logger.debug(f"Using fallback search for tables, top_k={top_k}")
            rows = await self.conn.fetch(
                """
                SELECT
                    table_name,
                    schema_info,
                    row_count,
                    0.8 AS similarity
                FROM schema_cache
                ORDER BY table_name
                LIMIT $1
                """,
                top_k,
            )
            logger.debug(f"Fallback search returned {len(rows)} tables")

            return [
                {
                    "table_name": row["table_name"],
                    "schema_info": json.loads(row["schema_info"])
                    if isinstance(row["schema_info"], str)
                    else row["schema_info"],
                    "row_count": row["row_count"],
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ]

    def _create_schema_description(self, table_name: str, schema_info: dict) -> str:
        parts = [f"Table: {table_name}"]

        if "columns" in schema_info:
            parts.append(
                "Columns: "
                + ", ".join(
                    [
                        f"{col['name']} ({col.get('type', 'unknown')})"
                        for col in schema_info["columns"]
                    ]
                )
            )

        if "description" in schema_info:
            parts.append(f"Description: {schema_info['description']}")

        return " | ".join(parts)
