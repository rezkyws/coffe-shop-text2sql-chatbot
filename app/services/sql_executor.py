import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import asyncpg

from app.core.config import get_settings
from app.models.schemas import QueryExecutionResult

settings = get_settings()
logger = logging.getLogger(__name__)


class SQLExecutor:
    def __init__(self, coffee_conn: asyncpg.Connection, main_conn: asyncpg.Connection):
        self.coffee_conn = coffee_conn
        self.main_conn = main_conn

    async def execute_query(
        self, sql_query: str, session_id: str, user_question: str, max_retries: Optional[int] = None
    ) -> QueryExecutionResult:
        # Handle case where parameters might be passed as dict
        # Convert to string if needed to prevent asyncpg errors
        sql_query = str(sql_query) if not isinstance(sql_query, str) else sql_query
        session_id = str(session_id) if not isinstance(session_id, str) else session_id
        user_question = str(user_question) if not isinstance(user_question, str) else user_question

        logger.info(
            f"[{session_id}] Starting SQL execution for question: '{user_question[:50]}...'"
        )
        logger.debug(f"[{session_id}] SQL query: {sql_query}")

        try:
            logger.info(f"[{session_id}] Executing SQL: {sql_query[:100]}...")

            start_time = time.time()

            # Execute with timeout
            logger.debug(
                f"[{session_id}] Sending query to database with timeout {settings.QUERY_TIMEOUT}s"
            )
            result = await asyncio.wait_for(
                self._execute_with_connection(sql_query), timeout=settings.QUERY_TIMEOUT
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log successful execution
            logger.debug(f"[{session_id}] Logging successful execution to database")
            await self._log_execution(
                session_id=session_id,
                user_question=user_question,
                sql_query=sql_query,
                status="success",
                result_rows=len(result),
                execution_time_ms=execution_time_ms,
                retry_count=0,
            )

            logger.info(
                f"[{session_id}] Query executed successfully in {execution_time_ms}ms, returned {len(result)} rows"
            )

            return QueryExecutionResult(
                success=True,
                data=result,
                execution_time_ms=execution_time_ms,
                rows_affected=len(result),
            )

        except asyncio.TimeoutError:
            error_msg = f"Query timeout after {settings.QUERY_TIMEOUT}s"
            logger.warning(f"[{session_id}] {error_msg}")

            # Log timeout failure to database
            await self._log_execution(
                session_id=session_id,
                user_question=user_question,
                sql_query=sql_query,
                status="failed",
                error_message=error_msg,
                retry_count=0,
            )
            return QueryExecutionResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{session_id}] Query execution failed: {error_msg}")

            # Log execution failure to database
            await self._log_execution(
                session_id=session_id,
                user_question=user_question,
                sql_query=sql_query,
                status="failed",
                error_message=error_msg,
                retry_count=0,
            )
            return QueryExecutionResult(success=False, error=error_msg)

    async def _execute_with_connection(self, sql_query: str) -> List[Dict[str, Any]]:
        logger.debug(f"Executing SQL query: {sql_query[:100]}...")

        try:
            rows = await self.coffee_conn.fetch(sql_query)
            logger.debug(f"Query returned {len(rows)} rows")

            if not rows:
                logger.debug("Query returned no results")
                return []

            # Convert to list of dicts
            result = [dict(row) for row in rows]
            logger.debug(f"Converted {len(result)} rows to dict format")
            return result

        except Exception as e:
            logger.error(f"Database query execution failed: {e}", exc_info=True)
            raise

    async def _log_execution(
        self,
        session_id: str,
        user_question: str,
        sql_query: str,
        status: str,
        error_message: Optional[str] = None,
        result_rows: Optional[int] = None,
        execution_time_ms: Optional[int] = None,
        retry_count: int = 0,
    ) -> None:
        try:
            # Ensure sql_query is a string, not a dict
            if isinstance(sql_query, dict):
                logger.warning(f"Received dict instead of string for sql_query: {sql_query}")
                if hasattr(sql_query, "get") and callable(getattr(sql_query, "get")):
                    sql_query = sql_query.get("sql_query", str(sql_query))
                else:
                    sql_query = str(sql_query)
                logger.warning(f"Converted sql_query to: {sql_query}")

            logger.debug(
                f"Logging execution with sql_query type: {type(sql_query)}, value: {sql_query[:100] if isinstance(sql_query, str) else sql_query}"
            )

            await self.main_conn.execute(
                """
                INSERT INTO query_execution_log
                (session_id, user_question, generated_sql, execution_status,
                 error_message, result_rows, execution_time_ms, retry_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                session_id,
                user_question,
                sql_query,
                status,
                error_message,
                result_rows,
                execution_time_ms,
                retry_count,
            )
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
