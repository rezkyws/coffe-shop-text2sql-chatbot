import asyncio
import logging
from contextlib import asynccontextmanager

import asyncpg
from pgvector.asyncpg import register_vector

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global connection pools
main_pool: asyncpg.Pool | None = None
coffee_pool: asyncpg.Pool | None = None


async def init_vector_support(conn):
    try:
        await register_vector(conn)
    except Exception as e:
        logger.warning(f"Failed to register vector type: {e}")
        # Continue without vector support if pgvector is not installed


async def create_pools():
    global main_pool, coffee_pool

    # Build database URLs from individual components
    main_db_url = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME_USER}"
    coffee_db_url = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

    # Main database pool (for chat history, vectors, etc.)
    main_pool = await asyncpg.create_pool(
        main_db_url,
        min_size=2,
        max_size=10,
        init=init_vector_support,
        command_timeout=30,
        max_inactive_connection_lifetime=300,
    )

    # Coffee shop database pool (read-only for queries)
    coffee_pool = await asyncpg.create_pool(
        coffee_db_url,
        min_size=2,
        max_size=8,
        command_timeout=30,
        max_inactive_connection_lifetime=300,
    )

    logger.info("Database connection pools created successfully")


async def close_pools():
    global main_pool, coffee_pool
    try:
        if main_pool:
            await asyncio.wait_for(main_pool.close(), timeout=10.0)
        if coffee_pool:
            await asyncio.wait_for(coffee_pool.close(), timeout=10.0)
        logger.info("Database connection pools closed")
    except asyncio.TimeoutError:
        logger.warning("Pool close timed out, forcing shutdown")
        # Force close by setting pools to None
        main_pool = None
        coffee_pool = None


@asynccontextmanager
async def get_db_connection():
    global main_pool
    if not main_pool:
        await create_pools()
    if main_pool is None:
        raise RuntimeError("Failed to create main database pool")

    async with main_pool.acquire() as conn:
        try:
            yield conn
        except Exception:
            # asyncpg connections don't have explicit rollback in pool context
            raise


@asynccontextmanager
async def get_coffee_db_connection():
    global coffee_pool
    if not coffee_pool:
        await create_pools()
    if coffee_pool is None:
        raise RuntimeError("Failed to create coffee database pool")

    async with coffee_pool.acquire() as conn:
        try:
            yield conn
        except Exception:
            # asyncpg connections don't have explicit rollback in pool context
            raise


async def get_db_dependency():
    global main_pool
    if not main_pool:
        await create_pools()
    if main_pool is None:
        raise RuntimeError("Failed to create main database pool")
    return await main_pool.acquire()


async def get_coffee_db_dependency():
    global coffee_pool
    if not coffee_pool:
        await create_pools()
    if coffee_pool is None:
        raise RuntimeError("Failed to create coffee database pool")
    return await coffee_pool.acquire()


async def init_db():
    if not main_pool:
        await create_pools()

    if main_pool is None:
        raise RuntimeError("Failed to create main database pool")

    async with main_pool.acquire() as conn:
        # Enable required extensions
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}")

        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        except Exception as e:
            logger.warning(f"Could not create pgcrypto extension: {e}")

        # Create tables using raw SQL (mimicking the Alembic schema)
        await create_tables(conn)

    logger.info("Database initialized successfully")


async def create_tables(conn):
    # Check if vector extension is available
    vector_available = False
    try:
        result = await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        vector_available = result is not None
    except:
        pass

    # Create conversation_history table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_created
        ON conversation_history (session_id, created_at)
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_conversation_history_session_id
        ON conversation_history (session_id)
    """)

    # Create conversation_summary table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summary (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            total_interactions INTEGER DEFAULT 0,
            last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_conversation_summary_session_id
        ON conversation_summary (session_id)
    """)

    # Create query_vector_store table (without vector columns if extension not available)
    if vector_available:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS query_vector_store (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                question TEXT NOT NULL,
                sql_query TEXT NOT NULL,
                embedding vector({settings.EMBEDDING_DIMENSION}) NOT NULL,
                success_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create HNSW index for query embeddings
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_embedding_hnsw
                ON query_vector_store
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")
    else:
        # Create table without vector column
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_vector_store (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                question TEXT NOT NULL,
                sql_query TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

    # Create schema_cache table (without vector columns if extension not available)
    if vector_available:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS schema_cache (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                table_name VARCHAR(255) NOT NULL UNIQUE,
                schema_info JSONB NOT NULL,
                embedding vector({settings.EMBEDDING_DIMENSION}) NOT NULL,
                row_count INTEGER,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # Create HNSW index for schema embeddings
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_schema_embedding_hnsw
                ON schema_cache
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")
    else:
        # Create table without vector column
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_cache (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                table_name VARCHAR(255) NOT NULL UNIQUE,
                schema_info JSONB NOT NULL,
                row_count INTEGER,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

    # Create query_execution_log table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS query_execution_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) NOT NULL,
            user_question TEXT NOT NULL,
            generated_sql TEXT NOT NULL,
            execution_status VARCHAR(50) NOT NULL,
            error_message TEXT,
            result_rows INTEGER,
            execution_time_ms INTEGER,
            retry_count INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_execution_session_status
        ON query_execution_log (session_id, execution_status)
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_query_execution_log_session_id
        ON query_execution_log (session_id)
    """)
