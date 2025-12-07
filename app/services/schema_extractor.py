import logging
from typing import Any, Dict, List

import asyncpg

from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class SchemaExtractor:
    def __init__(self, coffee_db: asyncpg.Connection, vector_store: VectorStoreManager):
        self.coffee_db = coffee_db
        self.vector_store = vector_store

    async def get_all_tables(self) -> List[str]:
        logger.debug("Retrieving all tables from database")
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            AND table_name NOT IN ('conversation_history', 'conversation_summary', 'query_execution_log', 'query_vector_store', 'schema_cache')
            ORDER BY table_name
        """

        try:
            tables = [row["table_name"] for row in await self.coffee_db.fetch(query)]
            logger.info(
                f"Found {len(tables)} tables in database: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}"
            )
            logger.debug(f"Complete table list: {tables}")
            return tables
        except Exception as e:
            logger.error(f"Failed to retrieve tables: {e}", exc_info=True)
            raise

    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        logger.debug(f"Retrieving schema for table: {table_name}")

        try:
            # Get columns
            logger.debug(f"Getting columns for {table_name}")
            column_query = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = $1
                ORDER BY ordinal_position
            """

            columns = []
            column_rows = await self.coffee_db.fetch(column_query, table_name)
            for row in column_rows:
                columns.append(
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                        "default": row["column_default"],
                        "max_length": row["character_maximum_length"],
                    }
                )
            logger.debug(f"Found {len(columns)} columns for {table_name}")

            # Get primary keys
            logger.debug(f"Getting primary keys for {table_name}")
            pk_query = """
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = $1::regclass AND i.indisprimary
            """

            primary_keys = [
                row["attname"] for row in await self.coffee_db.fetch(pk_query, table_name)
            ]
            logger.debug(f"Found {len(primary_keys)} primary keys for {table_name}")

            # Get foreign keys
            logger.debug(f"Getting foreign keys for {table_name}")
            fk_query = """
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = $1
            """

            foreign_keys = []
            fk_rows = await self.coffee_db.fetch(fk_query, table_name)
            for row in fk_rows:
                foreign_keys.append(
                    {
                        "column": row["column_name"],
                        "references_table": row["foreign_table_name"],
                        "references_column": row["foreign_column_name"],
                    }
                )
            logger.debug(f"Found {len(foreign_keys)} foreign keys for {table_name}")

            # Get row count estimate
            logger.debug(f"Getting row count for {table_name}")
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            row_count = await self.coffee_db.fetchval(count_query)
            logger.debug(f"Row count for {table_name}: {row_count}")

            schema_info = {
                "table_name": table_name,
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "row_count": row_count,
            }

            logger.info(
                f"Successfully retrieved schema for {table_name}: {len(columns)} columns, {row_count} rows"
            )
            return schema_info

        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}", exc_info=True)
            raise

    async def get_relevant_schemas(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        logger.debug(f"Searching for relevant schemas for query: '{query[:50]}...', top_k={top_k}")

        try:
            relevant_tables = await self.vector_store.search_relevant_tables(query, top_k=top_k)
            logger.info(f"Found {len(relevant_tables)} relevant tables for query")
            for table in relevant_tables:
                table_name = table.get("table_name", "unknown")
                score = table.get("similarity", 0)
                logger.debug(f"  - {table_name}: {score:.3f}")
            return relevant_tables
        except Exception as e:
            logger.error(f"Failed to get relevant schemas: {e}", exc_info=True)
            raise

    async def cache_all_schemas(self) -> None:
        logger.info("Starting schema caching process")

        try:
            tables = await self.get_all_tables()
            logger.info(f"Starting to cache {len(tables)} table schemas")

            cached_count = 0
            for table in tables:
                try:
                    logger.debug(f"Caching schema for table: {table}")
                    schema_info = await self.get_table_schema(table)
                    await self.vector_store.cache_schema_info(
                        table_name=table,
                        schema_info=schema_info,
                        row_count=schema_info.get("row_count"),
                    )
                    cached_count += 1
                    logger.info(
                        f"Successfully cached schema for table: {table} ({cached_count}/{len(tables)})"
                    )
                except Exception as e:
                    logger.error(f"Failed to cache schema for {table}: {e}", exc_info=True)

            logger.info(
                f"Schema caching completed: {cached_count}/{len(tables)} tables cached successfully"
            )

        except Exception as e:
            logger.error(f"Schema caching process failed: {e}", exc_info=True)
            raise

    def format_schema_for_llm(self, schemas: List[Dict[str, Any]]) -> str:
        formatted = []

        for schema in schemas:
            if isinstance(schema, dict) and "schema_info" in schema:
                schema_info = schema["schema_info"]
                table_name = schema["table_name"]
            else:
                schema_info = schema
                table_name = schema.get("table_name", "unknown")

            parts = [f"Table: {table_name}"]

            # Add columns
            if "columns" in schema_info:
                parts.append("Columns:")
                for col in schema_info["columns"]:
                    col_str = f"  - {col['name']} ({col['type']})"
                    if not col.get("nullable", True):
                        col_str += " NOT NULL"
                    parts.append(col_str)

            # Add primary keys
            if schema_info.get("primary_keys"):
                parts.append(f"Primary Keys: {', '.join(schema_info['primary_keys'])}")

            # Add foreign keys
            if schema_info.get("foreign_keys"):
                parts.append("Foreign Keys:")
                for fk in schema_info["foreign_keys"]:
                    parts.append(
                        f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}"
                    )

            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)
