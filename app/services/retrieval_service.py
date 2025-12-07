import logging
import re
from typing import Dict, List, Optional

from app.core.llm_client import LLMClient
from app.core.vector_store import VectorStoreManager
from app.models.schemas import SimilarQuery

logger = logging.getLogger(__name__)


def extract_tables_and_columns_from_sql(sql_query: str, schema_info: str) -> Dict[str, set]:
    # Extract table names (after FROM, JOIN, INTO, UPDATE)
    table_pattern = r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b"
    tables = set(re.findall(table_pattern, sql_query, re.IGNORECASE))

    # Extract column names (more comprehensive)
    column_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.(?:[a-zA-Z_][a-zA-Z0-9_]*)\b|\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=,|\s+FROM|\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|\s+LIMIT|\s*\))"
    all_matches = re.findall(column_pattern, sql_query, re.IGNORECASE)
    columns = set()
    for match in all_matches:
        if match[0]:  # table.column format
            columns.add(match[0] + "." + match[1])
        elif match[1]:  # standalone column
            columns.add(match[1])

    return {"tables": tables, "columns": columns}


def validate_sql_against_schema(sql_query: str, schema_info: str):
    errors = []

    # Parse schema to get available tables and columns
    available_tables = {}
    current_table = None

    for line in schema_info.split("\n"):
        if line.startswith("Table:"):
            current_table = line.split(":")[1].strip()
            available_tables[current_table] = set()
        elif line.strip().startswith("- ") and current_table:
            column_name = line.strip()[2:].split(" ")[0]  # Get column name before type
            available_tables[current_table].add(column_name)

    # Extract tables and columns from SQL
    extracted = extract_tables_and_columns_from_sql(sql_query, schema_info)

    # Validate tables
    for table in extracted["tables"]:
        if table not in available_tables:
            errors.append(
                f"Table '{table}' does not exist in schema. Available tables: {list(available_tables.keys())}"
            )

    # Validate columns
    for col_ref in extracted["columns"]:
        if "." in col_ref:  # table.column format
            table, column = col_ref.split(".", 1)
            if table not in available_tables:
                errors.append(f"Table '{table}' does not exist in schema")
            elif column not in available_tables[table]:
                errors.append(
                    f"Column '{column}' does not exist in table '{table}'. Available columns in {table}: {list(available_tables[table])}"
                )
        else:  # standalone column
            # Check if column exists in any table
            found = False
            for table_name, columns in available_tables.items():
                if col_ref in columns:
                    found = True
                    break
            if not found:
                available_columns = []
                for table_name, columns in available_tables.items():
                    available_columns.extend([f"{table_name}.{col}" for col in columns])
                errors.append(
                    f"Column '{col_ref}' not found in any table. Available columns: {available_columns[:10]}..."
                )

    return {"valid": len(errors) == 0, "errors": errors}


class RetrievalService:
    def __init__(self, vector_store: VectorStoreManager, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    async def get_similar_examples(self, query: str, top_k: int = 3) -> List[SimilarQuery]:
        """Retrieve similar query examples from vector store."""
        logger.debug(f"Searching for similar examples for query: '{query[:50]}...', top_k={top_k}")

        try:
            similar = await self.vector_store.search_similar_queries(query, top_k=top_k)

            logger.info(f"Found {len(similar)} similar queries for: {query[:50]}...")
            for i, sq in enumerate(similar, 1):
                logger.info(f"  {i}. Similarity {sq.similarity:.3f}: {sq.question[:50]}")
                logger.debug(f"     SQL: {sq.sql_query[:100]}...")

            return similar
        except Exception as e:
            logger.error(f"Failed to retrieve similar examples: {e}", exc_info=True)
            raise

    async def generate_sql_with_examples(
        self,
        user_query: str,
        schema_info: str,
        similar_examples: List[SimilarQuery],
        context: Optional[str] = None,
    ) -> str:
        logger.debug(f"Generating SQL for query: '{user_query[:50]}...'")
        logger.debug(
            f"Using {len(similar_examples)} similar examples and {len(schema_info)} chars of schema info"
        )

        try:
            # Build examples section
            examples_text = ""
            if similar_examples:
                examples_text = "Similar Query Examples:\n\n"
                for i, example in enumerate(similar_examples, 1):
                    examples_text += f"Example {i} (similarity: {example.similarity:.2f}):\n"
                    examples_text += f"Question: {example.question}\n"
                    examples_text += f"SQL: {example.sql_query}\n\n"
                logger.debug(f"Built examples text with {len(similar_examples)} examples")
            else:
                logger.debug("No similar examples available")

        except Exception as e:
            logger.error(f"Failed to build SQL generation context: {e}", exc_info=True)
            raise

        system_prompt = """You are an expert SQL query generator for a coffee shop database.

CRITICAL RULES - MUST FOLLOW:
1. ONLY use table names and column names that are explicitly provided in the Database Schema above
2. NEVER invent or guess table names or column names
3. ALWAYS cross-reference every table and column name against the provided schema
4. If a column name doesn't exist in the schema, the query will fail

Generate PostgreSQL queries that are:
1. Syntactically correct
2. Optimized for performance
3. Use proper JOINs when needed
4. Handle NULL values appropriately
5. Use appropriate date/time functions

Important:
- Always use explicit table names in SELECT
- Use aliases for better readability
- Add LIMIT clause for large result sets
- Use appropriate aggregation functions
- Format dates properly
- Learn from previous errors and avoid repeating mistakes

If previous errors are provided in the context:
- Analyze what went wrong in previous attempts
- Fix the specific issues mentioned in the errors
- Ensure column names, table names, and syntax are correct
- Pay attention to data types and formatting
- DOUBLE-CHECK that every column name exists in the provided schema

SCHEMA VALIDATION CHECKLIST before generating SQL:
-  Every table name exists in the provided schema
-  Every column name exists in the provided schema for its table
- JOIN conditions use correct foreign key relationships from schema
- No invented table names or column names

Respond with ONLY the SQL query, no explanations."""

        user_prompt = f"""Database Schema:
{schema_info}

{examples_text}

{f"Context: {context}" if context else ""}

Generate a SQL query to answer this question:
{user_query}

SQL Query:"""

        logger.info(f"Full prompt being sent to LLM:\n{user_prompt}")

        max_attempts = 3
        current_attempt = 0
        sql_query = ""

        while current_attempt < max_attempts:
            try:
                logger.debug(
                    f"Sending request to LLM for SQL generation (attempt {current_attempt + 1}/{max_attempts})"
                )
                sql_query = await self.llm_client.generate(
                    prompt=user_prompt, system=system_prompt, max_tokens=1000, temperature=0.0
                )
                logger.debug(f"LLM SQL response: {sql_query[:200]}...")

                # Clean up response
                sql_query = sql_query.strip()

                # Remove markdown code blocks if present
                if sql_query.startswith("```sql"):
                    sql_query = sql_query[6:]
                if sql_query.startswith("```"):
                    sql_query = sql_query[3:]
                if sql_query.endswith("```"):
                    sql_query = sql_query[:-3]

                sql_query = sql_query.strip()

                # Ensure it ends with semicolon
                if not sql_query.endswith(";"):
                    sql_query += ";"

                # Validate SQL against schema
                validation_result = validate_sql_against_schema(sql_query, schema_info)

                if validation_result["valid"]:
                    logger.info(f"Generated SQL (attempt {current_attempt + 1}): {sql_query}")
                    return sql_query
                else:
                    logger.warning(
                        f"SQL validation failed (attempt {current_attempt + 1}): {validation_result['errors']}"
                    )
                    # Add validation errors to context for next attempt
                    if context:
                        context += (
                            f"\n\nValidation Errors (attempt {current_attempt + 1}):\n"
                            + "\n".join(validation_result["errors"])
                        )
                    else:
                        context = (
                            f"Validation Errors (attempt {current_attempt + 1}):\n"
                            + "\n".join(validation_result["errors"])
                        )

                    user_prompt = f"""Database Schema:
{schema_info}

{examples_text}

Context: {context}

Generate a SQL query to answer this question:
{user_query}

SQL Query:"""
                    current_attempt += 1
                    continue

            except Exception as e:
                logger.error(f"Failed to generate SQL: {e}", exc_info=True)
                raise

        # If all attempts failed, return the last generated SQL
        logger.error(
            f"All {max_attempts} validation attempts failed. Returning last generated SQL."
        )
        return sql_query

    async def generate_answer_from_results(
        self, user_query: str, sql_query: str, results: List[dict], context: Optional[str] = None
    ) -> str:
        """Generate natural language answer from SQL results."""
        logger.debug(f"Generating answer for query: '{user_query[:50]}...'")
        logger.debug(f"SQL query returned {len(results)} results")

        try:
            # Format results for LLM
            if not results:
                results_text = "No results found."
                logger.debug("No results found for query")
            elif len(results) == 1:
                results_text = f"Result: {results[0]}"
                logger.debug(f"Single result: {results[0]}")
            else:
                results_text = f"Results ({len(results)} rows):\n"
                # Show first 10 rows
                for i, row in enumerate(results[:10], 1):
                    results_text += f"{i}. {row}\n"
                if len(results) > 10:
                    results_text += f"... and {len(results) - 10} more rows"
                logger.debug(f"Formatted {len(results)} results, showing first 10")

        except Exception as e:
            logger.error(f"Failed to format results for answer generation: {e}", exc_info=True)
            raise

        system_prompt = """You are a helpful assistant that explains database query results in natural language.

Provide clear, concise answers that:
1. Directly answer the user's question
2. Include relevant numbers and data points
3. Are easy to understand for non-technical users
4. Highlight key insights or patterns
5. Are conversational and friendly"""

        user_prompt = f"""{f"Context: {context}" if context else ""}

User Question: {user_query}

SQL Query Used: {sql_query}

Query Results:
{results_text}

Provide a natural language answer to the user's question:"""

        try:
            logger.debug("Sending request to LLM for answer generation")
            answer = await self.llm_client.generate(
                prompt=user_prompt, system=system_prompt, max_tokens=500, temperature=0.3
            )
            logger.debug(f"LLM answer: {answer[:200]}...")

            return answer.strip()

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}", exc_info=True)
            # Fallback to simple response
            if not results:
                return "I couldn't find any data matching your query."
            return f"I found {len(results)} result(s). Here's what I found: {results_text[:200]}..."
