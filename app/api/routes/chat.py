import asyncio
import logging
import uuid
from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import (
    get_clarification_service,
    get_coffee_db_connection,
    get_db_connection,
    get_llm_client,
    get_memory_manager,
    get_query_processor,
    get_retrieval_service,
    get_schema_extractor,
    get_sql_executor,
    get_vector_store,
)
from app.core.llm_client import LLMClient
from app.core.memory_manager import MemoryManager
from app.core.query_processor import QueryProcessor
from app.core.vector_store import VectorStoreManager
from app.models.schemas import ChatRequest, ChatResponse
from app.services.clarification_service import ClarificationService
from app.services.retrieval_service import RetrievalService
from app.services.schema_extractor import SchemaExtractor
from app.services.sql_executor import SQLExecutor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: asyncpg.Connection = Depends(get_db_connection),
    coffee_db: asyncpg.Connection = Depends(get_coffee_db_connection),
    llm_client: LLMClient = Depends(get_llm_client),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    query_processor: QueryProcessor = Depends(get_query_processor),
    sql_executor: SQLExecutor = Depends(get_sql_executor),
    schema_extractor: SchemaExtractor = Depends(get_schema_extractor),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    clarification_service: ClarificationService = Depends(get_clarification_service),
):
    # Generate session ID if not provided (outside try block for error handling)
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(f"[{session_id}] === CHAT REQUEST START ===")
        logger.info(f"[{session_id}] Processing query: '{request.message[:100]}...'")
        logger.debug(
            f"[{session_id}] Request metadata: session_id={request.session_id is not None}"
        )
        logger.info(f"[{session_id}] === CHAT REQUEST START ===")
        logger.info(f"[{session_id}] Processing query: '{request.message[:100]}...'")
        logger.debug(
            f"[{session_id}] Request metadata: session_id={request.session_id is not None}"
        )

        # Check if this is a clarification response
        # (Check if there's a recent clarification request in memory)
        logger.debug(f"[{session_id}] Checking if this is a clarification response")
        recent_messages = await memory_manager.get_short_term_memory(session_id)
        logger.debug(f"[{session_id}] Found {len(recent_messages)} recent messages")

        is_clarification_response = False
        original_query = None

        if recent_messages and len(recent_messages) >= 2:
            logger.debug(f"[{session_id}] Analyzing recent messages for clarification patterns")
            last_assistant_msg = None
            for msg in reversed(recent_messages):
                if msg.role.value == "assistant":
                    last_assistant_msg = msg
                    break

            if (
                last_assistant_msg
                and "need" in last_assistant_msg.content.lower()
                and (
                    "clarification" in last_assistant_msg.content.lower()
                    or "information" in last_assistant_msg.content.lower()
                )
            ):
                is_clarification_response = True
                logger.info(f"[{session_id}] Detected clarification response")
                # Find the original user query
                for msg in reversed(recent_messages):
                    if msg.role.value == "user" and msg != recent_messages[-1]:
                        original_query = msg.content
                        logger.debug(
                            f"[{session_id}] Found original query: '{original_query[:50]}...'"
                        )
                        break

        # If this is a clarification response, enhance the query
        if is_clarification_response and original_query:
            logger.info(f"[{session_id}] Processing clarification response")
            enhanced_query = await clarification_service.process_clarification_response(
                original_query=original_query,
                clarification_response=request.message,
                session_id=session_id,
            )
            logger.info(f"[{session_id}] Enhanced query: '{enhanced_query}'")
            # Use enhanced query for processing
            query_to_process = enhanced_query
        else:
            query_to_process = request.message
            logger.debug(f"[{session_id}] Using original query for processing")

        # Check if query needs clarification (only for new queries)
        if not is_clarification_response:
            logger.debug(f"[{session_id}] Checking if new query needs clarification")
            (
                needs_clarification,
                questions,
                metadata,
            ) = await clarification_service.needs_clarification(
                query=query_to_process, session_id=session_id
            )

            if needs_clarification and questions:
                logger.info(f"[{session_id}] Query needs clarification: {metadata}")

                clarification_message = clarification_service.format_clarification_response(
                    questions=questions, metadata=metadata
                )
                logger.debug(
                    f"[{session_id}] Formatted clarification message: '{clarification_message[:100]}...'"
                )

                # Save interaction
                logger.debug(f"[{session_id}] Saving clarification interaction to memory")
                await memory_manager.save_interaction(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_message=clarification_message,
                    metadata={"needs_clarification": True, **metadata},
                )

                logger.info(f"[{session_id}] Returning clarification response")
                return ChatResponse(
                    response=clarification_message,
                    session_id=session_id,
                    clarification_needed=True,
                    clarification_questions=questions,
                    metadata=metadata,
                )

        # Process query and get action plan
        logger.debug(f"[{session_id}] Getting action plan for query")
        plan = await query_processor.process_query(query_to_process, session_id)

        # Execute action sequence
        logger.debug(f"[{session_id}] Building context for action execution")
        context = await memory_manager.build_context(session_id, query_to_process)
        similar_queries = []
        relevant_schemas = []
        generated_sql = None
        execution_result = None

        logger.info(
            f"[{session_id}] Executing {len(plan.get('actions', []))} actions: {plan.get('actions', [])}"
        )
        for i, action in enumerate(plan.get("actions", []), 1):
            logger.info(f"[{session_id}] Action {i}/{len(plan.get('actions', []))}: {action}")

            if action == "RETRIEVE_SIMILAR":
                # Retrieve similar query examples
                logger.debug(f"[{session_id}] Retrieving similar query examples")
                similar_queries = await retrieval_service.get_similar_examples(
                    query_to_process, top_k=3
                )

            elif action == "EXTRACT_SCHEMA":
                # Get relevant schemas
                logger.debug(f"[{session_id}] Extracting relevant schemas")
                relevant_schemas = await schema_extractor.get_relevant_schemas(
                    query_to_process, top_k=3
                )

                # Ensure fact_sales is included for transaction-related queries
                query_lower = query_to_process.lower()
                transaction_keywords = [
                    "transaction",
                    "sales",
                    "quantity",
                    "purchase",
                    "order",
                    "receipt",
                ]
                if any(keyword in query_lower for keyword in transaction_keywords):
                    # Check if fact_sales is already in relevant schemas
                    fact_sales_included = any(
                        schema.get("table_name") == "fact_sales" for schema in relevant_schemas
                    )
                    if not fact_sales_included:
                        # Add fact_sales schema
                        fact_sales_schema = await schema_extractor.get_table_schema("fact_sales")
                        relevant_schemas.append(
                            {
                                "table_name": "fact_sales",
                                "schema_info": fact_sales_schema,
                                "row_count": fact_sales_schema.get("row_count", 0),
                                "similarity": 1.0,  # High relevance for transaction queries
                            }
                        )
                        logger.info(
                            f"[{session_id}] Added fact_sales schema for transaction-related query"
                        )

                # Ensure dim_store is included for city/location-related queries
                location_keywords = [
                    "city",
                    "location",
                    "store",
                    "outlet",
                    "address",
                    "state",
                    "province",
                ]
                if any(keyword in query_lower for keyword in location_keywords):
                    # Check if dim_store is already in relevant schemas
                    dim_store_included = any(
                        schema.get("table_name") == "dim_store" for schema in relevant_schemas
                    )
                    if not dim_store_included:
                        # Add dim_store schema from cache
                        dim_store_schema = await schema_extractor.get_table_schema("dim_store")
                        relevant_schemas.append(
                            {
                                "table_name": "dim_store",
                                "schema_info": dim_store_schema,
                                "row_count": dim_store_schema.get("row_count", 0),
                                "similarity": 1.0,  # High relevance for location queries
                            }
                        )
                        logger.info(
                            f"[{session_id}] Added dim_store schema for location-related query"
                        )

            elif action == "GENERATE_SQL":
                # Generate SQL query
                logger.debug(f"[{session_id}] Generating SQL query")
                schema_text = schema_extractor.format_schema_for_llm(relevant_schemas)
                generated_sql = await retrieval_service.generate_sql_with_examples(
                    user_query=query_to_process,
                    schema_info=schema_text,
                    similar_examples=similar_queries,
                    context=context,
                )

            elif action == "EXECUTE_SQL":
                # Execute SQL query with retry mechanism and query regeneration
                if generated_sql:
                    logger.debug(f"[{session_id}] Executing generated SQL with retry loop")

                    max_sql_retries = 3
                    error_history = []
                    execution_result = None
                    sql_attempt = 0

                    while sql_attempt < max_sql_retries:
                        logger.info(
                            f"[{session_id}] SQL attempt {sql_attempt + 1}/{max_sql_retries}"
                        )

                        # Add error history to context for regeneration (except for first attempt)
                        if sql_attempt > 0:
                            error_context = "\n\nPrevious Errors:\n"
                            for i, error in enumerate(error_history, 1):
                                error_context += f"Attempt {i}: Query failed with error: {error}\n"
                            error_context += "\nPlease fix the SQL query to address these errors."

                            # Regenerate SQL with error context
                            logger.debug(f"[{session_id}] Regenerating SQL with error context")
                            schema_text = schema_extractor.format_schema_for_llm(relevant_schemas)
                            generated_sql = await retrieval_service.generate_sql_with_examples(
                                user_query=query_to_process,
                                schema_info=schema_text,
                                similar_examples=similar_queries,
                                context=context + error_context,
                            )
                            logger.info(f"[{session_id}] Regenerated SQL: {generated_sql}")

                        # Execute the SQL query
                        execution_result = await sql_executor.execute_query(
                            sql_query=generated_sql,
                            session_id=session_id,
                            user_question=query_to_process,
                        )

                        if execution_result and execution_result.success:
                            logger.info(
                                f"[{session_id}] SQL execution successful on attempt {sql_attempt + 1}"
                            )
                            break
                        else:
                            error_msg = (
                                execution_result.error if execution_result else "Unknown error"
                            )
                            error_history.append(error_msg)
                            logger.warning(
                                f"[{session_id}] SQL execution failed on attempt {sql_attempt + 1}: {error_msg}"
                            )

                            # Wait before retry with exponential backoff
                            if sql_attempt < max_sql_retries - 1:
                                wait_time = 2**sql_attempt
                                logger.debug(f"[{session_id}] Waiting {wait_time}s before retry")
                                await asyncio.sleep(wait_time)

                        sql_attempt += 1

                    if not execution_result or not execution_result.success:
                        final_error = (
                            execution_result.error if execution_result else "Unknown error"
                        )
                        logger.error(
                            f"[{session_id}] All SQL retry attempts failed. Last error: {final_error}"
                        )

            elif action == "ANSWER":
                # Generate final answer
                logger.debug(f"[{session_id}] Generating final answer")
                if execution_result and execution_result.success:
                    answer = await retrieval_service.generate_answer_from_results(
                        user_query=query_to_process,
                        sql_query=generated_sql or "",
                        results=execution_result.data or [],
                        context=context,
                    )
                else:
                    error_detail = execution_result.error if execution_result else "Unknown error"
                    answer = f"I encountered an error while executing the query: {error_detail}. Please try rephrasing your question."

                # Save interaction to memory
                logger.debug(f"[{session_id}] Saving successful interaction to memory")
                await memory_manager.save_interaction(
                    session_id=session_id,
                    user_message=request.message,
                    assistant_message=answer,
                    metadata={
                        "sql_query": generated_sql,
                        "execution_status": "success"
                        if execution_result and execution_result.success
                        else "failed",
                        "was_clarification_response": is_clarification_response,
                    },
                )

                # If successful, update vector store success count
                if execution_result and execution_result.success and similar_queries:
                    logger.debug(f"[{session_id}] Updating vector store success counts")
                    for sq in similar_queries:
                        if sq.similarity > 0.8:  # High similarity threshold
                            await vector_store.update_query_success(sq.question)

                logger.info(f"[{session_id}] === CHAT REQUEST SUCCESS ===")
                return ChatResponse(
                    response=answer,
                    session_id=session_id,
                    sql_query=generated_sql,
                    clarification_needed=False,
                    metadata={
                        "execution_time_ms": execution_result.execution_time_ms
                        if execution_result
                        else None,
                        "rows_returned": execution_result.rows_affected
                        if execution_result
                        else None,
                        "similar_queries_used": len(similar_queries),
                        "reasoning": plan.get("reasoning", ""),
                        "complexity": plan.get("estimated_complexity", "medium"),
                    },
                )

        # Fallback response if no ANSWER action was executed
        logger.warning(f"[{session_id}] No ANSWER action executed, using fallback response")
        fallback_answer = "I'm processing your request. Could you provide more details?"
        await memory_manager.save_interaction(
            session_id=session_id,
            user_message=request.message,
            assistant_message=fallback_answer,
            metadata={"fallback": True},
        )

        logger.info(f"[{session_id}] === CHAT REQUEST FALLBACK ===")
        return ChatResponse(
            response=fallback_answer, session_id=session_id, clarification_needed=False
        )

    except Exception as e:
        logger.error(f"[{session_id}] === CHAT REQUEST ERROR ===")
        logger.error(f"[{session_id}] Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache-schemas")
async def cache_schemas(
    db: asyncpg.Connection = Depends(get_db_connection),
    schema_extractor: SchemaExtractor = Depends(get_schema_extractor),
):
    try:
        await schema_extractor.cache_all_schemas()

        return {"status": "success", "message": "All schemas cached successfully"}
    except Exception as e:
        logger.error(f"Schema caching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-query-example")
async def add_query_example(
    question: str,
    sql_query: str,
    db: asyncpg.Connection = Depends(get_db_connection),
    vector_store: VectorStoreManager = Depends(get_vector_store),
):
    try:
        await vector_store.store_query_example(question=question, sql_query=sql_query)

        return {"status": "success", "message": f"Query example added: {question[:50]}..."}
    except Exception as e:
        logger.error(f"Add query example error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: Optional[int] = 20,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    try:
        history = await memory_manager.get_long_term_memory(session_id, limit=limit)
        summary = await memory_manager.get_conversation_summary(session_id)

        return {
            "session_id": session_id,
            "summary": summary,
            "history": [msg.model_dump() for msg in history],
        }
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str, db: asyncpg.Connection = Depends(get_db_connection)):
    try:
        # Delete conversation history
        await db.execute("DELETE FROM conversation_history WHERE session_id = $1", session_id)

        # Delete summary
        await db.execute("DELETE FROM conversation_summary WHERE session_id = $1", session_id)

        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
