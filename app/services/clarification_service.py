import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.llm_client import LLMClient
from app.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ClarificationService:
    def __init__(self, llm_client: LLMClient, memory_manager: MemoryManager):
        self.llm_client = llm_client
        self.memory_manager = memory_manager

    async def needs_clarification(
        self, query: str, session_id: str
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        logger.info(f"[{session_id}] Checking if query needs clarification: '{query[:50]}...'")

        try:
            # Build context from conversation history
            logger.debug(f"[{session_id}] Building context for clarification check")
            context = await self.memory_manager.build_context(session_id, query)
            logger.debug(f"[{session_id}] Context built, length: {len(context)} chars")

        except Exception as e:
            logger.error(
                f"[{session_id}] Failed to build context for clarification: {e}", exc_info=True
            )
            return False, [], {"error": str(e)}

        system_prompt = """You are a query clarification expert for a coffee shop database system.

Your task is to identify if a user's query is ambiguous or missing critical information needed to generate an accurate SQL query.

EXTREMELY IMPORTANT: Be extremely conservative about asking for clarification. The default should always be to NOT ask for clarification unless the query is completely impossible to answer.

Assumptions you should make by default:
- Time period: If not specified, assume "all time" or "all available data"
- Location: If not specified, assume "all stores" or "all locations"
- Product scope: If not specified, assume "all products"
- Aggregation: If not specified, use the most common interpretation (e.g., "total" for revenue, "count" for items)

ONLY ask for clarification in these specific cases:
1. The query is completely ambiguous (e.g., "How did we do?" - compared to what?)
2. The query contains contradictory information
3. The query references entities that don't exist or are unclear

DO NOT ask for clarification for:
- Missing time periods (assume all time)
- Missing locations (assume all stores)
- Missing product categories (assume all products)
- General business questions that can be reasonably interpreted

Rules:
- When in doubt, DO NOT ask for clarification
- Maximum 1 question only
- If the query can be reasonably answered with default assumptions, let it proceed

Respond with JSON only:
{
  "needs_clarification": boolean,
  "reasons": ["reason1", "reason2"],
  "questions": ["specific question 1"],
  "missing_info": ["time_period", "location", "product_type"],
  "confidence": 0.0-1.0
}"""

        user_prompt = f"""Context from conversation:
{context}

Current Query: {query}

Analyze if this query needs clarification:"""

        response = None
        try:
            logger.debug(f"[{session_id}] Sending request to LLM for clarification analysis")
            response = await self.llm_client.generate(
                prompt=user_prompt, system=system_prompt, max_tokens=800, temperature=0.0
            )
            logger.debug(f"[{session_id}] LLM clarification response: {response[:200]}...")

            # Parse JSON response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            logger.debug(f"[{session_id}] Parsing clarification JSON response")
            result = json.loads(response.strip())

            needs_clarification = result.get("needs_clarification", False)
            questions = result.get("questions", [])
            confidence = result.get("confidence", 0.5)

            # Apply very strict confidence threshold, only ask for clarification if confidence is very high (>0.95)
            if needs_clarification and confidence < 0.95:
                logger.info(
                    f"[{session_id}] Low confidence ({confidence}) for clarification, allowing query to proceed"
                )
                needs_clarification = False
                questions = []

            metadata = {
                "reasons": result.get("reasons", []),
                "missing_info": result.get("missing_info", []),
                "confidence": confidence,
            }

            logger.info(
                f"[{session_id}] Clarification check for '{query[:50]}...': "
                f"needs_clarification={needs_clarification}, "
                f"questions={len(questions)}, "
                f"confidence={metadata['confidence']}"
            )

            return needs_clarification, questions, metadata

        except json.JSONDecodeError as e:
            logger.error(f"[{session_id}] JSON parsing failed for clarification: {e}")
            logger.error(
                f"[{session_id}] Raw clarification response: {response if response else 'No response received'}"
            )
            # Conservative default, don't ask for clarification on error
            return False, [], {"error": str(e)}
        except Exception as e:
            logger.error(f"[{session_id}] Clarification analysis failed: {e}", exc_info=True)
            # Conservative default, don't ask for clarification on error
            return False, [], {"error": str(e)}

    async def process_clarification_response(
        self, original_query: str, clarification_response: str, session_id: str
    ) -> str:
        logger.info(f"[{session_id}] Processing clarification response")
        logger.debug(f"[{session_id}] Original query: '{original_query}'")
        logger.debug(f"[{session_id}] Clarification response: '{clarification_response}'")

        try:
            # Get recent context
            logger.debug(f"[{session_id}] Getting recent messages for context")
            recent_messages = await self.memory_manager.get_short_term_memory(session_id)

            context_text = "\n".join(
                [
                    f"{'User' if msg.role.value == 'user' else 'Assistant'}: {msg.content}"
                    for msg in recent_messages[-4:]  # Last 2 exchanges
                ]
            )
            logger.debug(f"[{session_id}] Built context from {len(recent_messages)} messages")

            system_prompt = """You are a query enhancement expert.

Your task is to combine an original ambiguous query with the user's clarification response to create a complete, unambiguous query.

The enhanced query should:
1. Include all relevant information from both messages
2. Be clear and specific
3. Be phrased as a single, coherent question
4. Include time periods, locations, and other specifics provided

Example:
Original: "How many coffees sold?"
Clarification: "Last week at the downtown store"
Enhanced: "How many coffees were sold last week at the downtown store?"

Respond with ONLY the enhanced query, nothing else."""

            user_prompt = f"""Recent conversation:
{context_text}

Original Query: {original_query}
Clarification Response: {clarification_response}

Enhanced Query:"""

            logger.debug(f"[{session_id}] Sending request to LLM for query enhancement")
            enhanced_query = await self.llm_client.generate(
                prompt=user_prompt, system=system_prompt, max_tokens=200, temperature=0.0
            )

            enhanced_query = enhanced_query.strip().strip('"').strip("'")

            logger.info(f"[{session_id}] Enhanced query: '{original_query}' → '{enhanced_query}'")

            return enhanced_query

        except Exception as e:
            logger.error(f"[{session_id}] Query enhancement failed: {e}", exc_info=True)
            # Fallback, just append the clarification
            return f"{original_query} {clarification_response}"

    def format_clarification_response(
        self, questions: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not questions:
            return "I need some additional information to answer your question."

        if len(questions) == 1:
            intro = "I need to know:"
        else:
            intro = "I need some clarification on a few things:"

        formatted_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

        return f"{intro}\n\n{formatted_questions}"
