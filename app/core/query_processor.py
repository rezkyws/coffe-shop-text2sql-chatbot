import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.core.llm_client import LLMClient
from app.core.memory_manager import MemoryManager
from app.core.vector_store import VectorStoreManager

settings = get_settings()
logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    CLARIFY = "clarify"
    RETRIEVE_SIMILAR = "retrieve_similar"
    EXTRACT_SCHEMA = "extract_schema"
    GENERATE_SQL = "generate_sql"
    EXECUTE_SQL = "execute_sql"
    ANSWER = "answer"


class QueryProcessor:
    def __init__(
        self, llm_client: LLMClient, memory_manager: MemoryManager, vector_store: VectorStoreManager
    ):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.vector_store = vector_store

    async def process_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        logger.info(f"[{session_id}] Starting query processing for: '{user_query[:100]}...'")

        try:
            # Build context from memory
            logger.debug(f"[{session_id}] Building context from memory")
            context = await self.memory_manager.build_context(session_id, user_query)
            logger.debug(f"[{session_id}] Context built, length: {len(context)} chars")

            # Get conversation summary for better understanding
            logger.debug(f"[{session_id}] Retrieving conversation summary")
            summary = await self.memory_manager.get_conversation_summary(session_id)
            logger.debug(f"[{session_id}] Summary retrieved: {bool(summary)}")

            # Generate reasoning and action plan
            logger.debug(f"[{session_id}] Generating action plan")
            plan = await self._generate_action_plan(user_query, context, summary)

            logger.info(
                f"[{session_id}] Generated action plan: {plan['actions']}, complexity: {plan.get('estimated_complexity', 'unknown')}"
            )
            logger.debug(
                f"[{session_id}] Plan reasoning: {plan.get('reasoning', 'No reasoning provided')}"
            )

            return plan

        except Exception as e:
            logger.error(f"[{session_id}] Query processing failed: {e}", exc_info=True)
            raise

    async def _generate_action_plan(
        self, query: str, context: str, summary: Optional[str]
    ) -> Dict[str, Any]:
        logger.debug(f"Generating action plan for query: '{query[:50]}...'")
        logger.debug(f"Context length: {len(context)} chars, Summary: {bool(summary)}")

        system_prompt = """You are a query analysis expert for a coffee shop database system.

Your role is to analyze user queries and determine the best sequence of actions to answer them.

Available Actions:
1. CLARIFY - Ask for clarification if query is ambiguous (e.g., missing date range, store location)
2. RETRIEVE_SIMILAR - Search for similar queries in the knowledge base
3. EXTRACT_SCHEMA - Get relevant table schemas from the database
4. GENERATE_SQL - Generate SQL query based on context and examples
5. EXECUTE_SQL - Execute the generated SQL query
6. ANSWER - Formulate the final answer for the user

Guidelines:
- If query is ambiguous or incomplete, start with CLARIFY
- Always RETRIEVE_SIMILAR first to find example queries
- Use EXTRACT_SCHEMA to understand relevant tables
- Use examples from similar queries when generating SQL
- Answer should be natural language, not raw SQL results

Respond with JSON only:
{
  "needs_clarification": boolean,
  "reasoning": "Brief explanation of the plan",
  "clarification_questions": ["question1", "question2"],
  "actions": ["ACTION1", "ACTION2", ...],
  "estimated_complexity": "low|medium|high"
}"""

        user_prompt = f"""Context:
{context}

{f"Conversation Summary: {summary}" if summary else ""}

Analyze this query and create an action plan:
Query: {query}

Respond with JSON only:"""

        response = None
        response = None
        try:
            logger.debug("Sending request to LLM for action plan generation")
            response = await self.llm_client.generate(
                prompt=user_prompt, system=system_prompt, max_tokens=1000, temperature=0.0
            )
            logger.debug(f"LLM response received: {response[:200]}...")

            # Parse JSON response
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            logger.debug("Parsing JSON response")
            plan = json.loads(response.strip())

            # Validate and set defaults
            plan.setdefault("needs_clarification", False)
            plan.setdefault("clarification_questions", [])
            plan.setdefault(
                "actions",
                ["RETRIEVE_SIMILAR", "EXTRACT_SCHEMA", "GENERATE_SQL", "EXECUTE_SQL", "ANSWER"],
            )
            plan.setdefault("reasoning", "")
            plan.setdefault("estimated_complexity", "medium")

            logger.debug(f"Action plan parsed successfully: {len(plan['actions'])} actions")
            return plan

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for action plan: {e}")
            logger.error(f"Raw response: {response if response else 'No response received'}")
            # Return a safe default plan
            return {
                "needs_clarification": False,
                "clarification_questions": [],
                "actions": [
                    "RETRIEVE_SIMILAR",
                    "EXTRACT_SCHEMA",
                    "GENERATE_SQL",
                    "EXECUTE_SQL",
                    "ANSWER",
                ],
                "reasoning": "Using default action sequence due to JSON parsing error",
                "estimated_complexity": "medium",
            }
        except Exception as e:
            logger.error(f"Failed to generate action plan: {e}", exc_info=True)
            # Return a safe default plan
            return {
                "needs_clarification": False,
                "clarification_questions": [],
                "actions": [
                    "RETRIEVE_SIMILAR",
                    "EXTRACT_SCHEMA",
                    "GENERATE_SQL",
                    "EXECUTE_SQL",
                    "ANSWER",
                ],
                "reasoning": "Using default action sequence due to planning error",
                "estimated_complexity": "medium",
            }

    async def should_clarify(self, query: str, context: str) -> Tuple[bool, List[str]]:
        logger.debug(f"Checking if query needs clarification: '{query[:50]}...'")

        system_prompt = """You are a query clarification expert. Analyze if the user's query has all necessary information to generate a SQL query for a coffee shop database.

Common ambiguities:
- Missing time periods (today, last week, last month, specific dates)
- Missing store/location information
- Unclear aggregation (total, average, count)
- Ambiguous product references
- Asking outside business context

Respond with JSON only:
{
  "needs_clarification": boolean,
  "questions": ["specific question 1", "specific question 2"]
}"""

        user_prompt = f"""Context: {context}

Query: {query}

Does this query need clarification? If yes, what specific information is missing?"""

        response = None
        try:
            logger.debug("Sending request to LLM for clarification check")
            response = await self.llm_client.generate(
                prompt=user_prompt, system=system_prompt, max_tokens=500, temperature=0.0
            )
            logger.debug(f"Clarification response received: {response[:200]}...")

            # Parse response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]

            logger.debug("Parsing clarification JSON response")
            result = json.loads(response.strip())

            needs_clarification = result.get("needs_clarification", False)
            questions = result.get("questions", [])

            logger.debug(
                f"Clarification result: needs_clarification={needs_clarification}, questions={len(questions)}"
            )

            return (needs_clarification, questions)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for clarification: {e}")
            logger.error(
                f"Raw clarification response: {response if response else 'No response received'}"
            )
            return False, []
        except Exception as e:
            logger.error(f"Clarification check failed: {e}", exc_info=True)
            return False, []
