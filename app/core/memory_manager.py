import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg
from redis import asyncio as redis_async

from app.core.config import get_settings
from app.core.llm_client import LLMClient
from app.models.schemas import Message, MessageRole

settings = get_settings()
logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, conn: asyncpg.Connection, llm_client: LLMClient):
        self.conn = conn
        self.llm_client = llm_client
        try:
            self.redis_client = redis_async.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    async def get_short_term_memory(self, session_id: str) -> List[Message]:
        """Get recent K interactions from Redis cache."""
        if not self.redis_client:
            # Fallback to DB if Redis is not available
            return await self._get_recent_from_db(session_id, settings.SHORT_TERM_MEMORY_WINDOW)

        try:
            key = f"stm:{session_id}"
            messages = await self.redis_client.lrange(key, 0, settings.SHORT_TERM_MEMORY_WINDOW - 1)
            return [Message(**json.loads(msg)) for msg in messages]
        except Exception as e:
            logger.warning(f"Failed to get short-term memory from Redis: {e}")
            # Fallback to DB
            return await self._get_recent_from_db(session_id, settings.SHORT_TERM_MEMORY_WINDOW)

    async def get_long_term_memory(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Message]:
        query = """
            SELECT role, content, metadata, created_at
            FROM conversation_history
            WHERE session_id = $1
            ORDER BY created_at DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        rows = await self.conn.fetch(query, session_id)

        return [
            Message(
                role=MessageRole(row["role"]),
                content=row["content"],
                metadata=json.loads(row["metadata"])
                if row["metadata"] and isinstance(row["metadata"], str)
                else (row["metadata"] or {}),
            )
            for row in reversed(rows)
        ]

    async def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """Get compressed conversation summary."""
        row = await self.conn.fetchrow(
            "SELECT summary FROM conversation_summary WHERE session_id = $1", session_id
        )
        return row["summary"] if row else None

    async def save_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save a user-assistant interaction to both short-term and long-term memory."""
        metadata = metadata or {}

        # Save to database (long-term memory)
        await self.conn.execute(
            """
            INSERT INTO conversation_history (session_id, role, content, metadata)
            VALUES ($1, 'user', $2, $3)
            """,
            session_id,
            user_message,
            json.dumps(metadata) if metadata else "{}",
        )

        await self.conn.execute(
            """
            INSERT INTO conversation_history (session_id, role, content, metadata)
            VALUES ($1, 'assistant', $2, $3)
            """,
            session_id,
            assistant_message,
            json.dumps(metadata) if metadata else "{}",
        )

        # Save to Redis (short-term memory)
        if self.redis_client:
            try:
                key = f"stm:{session_id}"
                user_msg = Message(role=MessageRole.USER, content=user_message, metadata=metadata)
                asst_msg = Message(
                    role=MessageRole.ASSISTANT, content=assistant_message, metadata=metadata
                )

                await self.redis_client.lpush(key, json.dumps(user_msg.model_dump()))
                await self.redis_client.lpush(key, json.dumps(asst_msg.model_dump()))
                await self.redis_client.ltrim(key, 0, settings.SHORT_TERM_MEMORY_WINDOW * 2 - 1)
                await self.redis_client.expire(key, 86400)  # 24 hour TTL
            except Exception as e:
                logger.warning(f"Failed to save to Redis short-term memory: {e}")

        # Update summary if conversation is getting long
        await self._update_summary_if_needed(session_id)

    async def build_context(self, session_id: str, current_query: str) -> str:
        summary = await self.get_conversation_summary(session_id)
        recent_messages = await self.get_short_term_memory(session_id)

        context_parts = []

        if summary:
            context_parts.append(f"Conversation Summary:\n{summary}\n")

        if recent_messages:
            context_parts.append("Recent Conversation:")
            for msg in recent_messages:
                role_label = "User" if msg.role == MessageRole.USER else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")

        context_parts.append(f"\nCurrent Query: {current_query}")

        return "\n".join(context_parts)

    async def _update_summary_if_needed(self, session_id: str) -> None:
        # Get total message count
        total_count = await self.conn.fetchval(
            "SELECT COUNT(*) FROM conversation_history WHERE session_id = $1", session_id
        )

        # Update summary every MAX_CONVERSATION_HISTORY messages
        if total_count % settings.MAX_CONVERSATION_HISTORY == 0:
            await self._generate_summary(session_id)

    async def _generate_summary(self, session_id: str) -> None:
        """Generate a new conversation summary using LLM."""
        # Get existing summary
        existing_summary = await self.conn.fetchrow(
            "SELECT summary, last_updated FROM conversation_summary WHERE session_id = $1",
            session_id,
        )

        # Get messages to summarize
        if existing_summary:
            messages = await self._get_messages_since(session_id, existing_summary["last_updated"])
        else:
            messages = await self.get_long_term_memory(
                session_id, settings.MAX_CONVERSATION_HISTORY
            )

        if not messages:
            return

        # Build summary prompt
        conversation_text = "\n".join(
            [
                f"{'User' if m.role == MessageRole.USER else 'Assistant'}: {m.content}"
                for m in messages
            ]
        )

        summary_prompt = f"""Please provide a concise summary of the following conversation, focusing on:
1. Main topics discussed
2. Key questions asked
3. Important context or data mentioned
4. Any unresolved issues

Previous Summary: {existing_summary["summary"] if existing_summary else "None"}

Recent Conversation:
{conversation_text}

Provide a comprehensive but concise summary:"""

        try:
            new_summary = await self.llm_client.generate(summary_prompt, max_tokens=500)

            if existing_summary:
                await self.conn.execute(
                    """
                    UPDATE conversation_summary
                    SET summary = $2, total_interactions = $3, last_updated = NOW()
                    WHERE session_id = $1
                    """,
                    session_id,
                    new_summary,
                    len(messages),
                )
            else:
                await self.conn.execute(
                    """
                    INSERT INTO conversation_summary (session_id, summary, total_interactions)
                    VALUES ($1, $2, $3)
                    """,
                    session_id,
                    new_summary,
                    len(messages),
                )

            logger.info(f"Generated summary for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    async def _get_recent_from_db(self, session_id: str, limit: int) -> List[Message]:
        rows = await self.conn.fetch(
            """
            SELECT role, content, metadata, created_at
            FROM conversation_history
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )

        return [
            Message(
                role=MessageRole(row["role"]),
                content=row["content"],
                metadata=json.loads(row["metadata"])
                if row["metadata"] and isinstance(row["metadata"], str)
                else (row["metadata"] or {}),
            )
            for row in reversed(rows)
        ]

    async def _get_messages_since(self, session_id: str, since: datetime) -> List[Message]:
        rows = await self.conn.fetch(
            """
            SELECT role, content, metadata, created_at
            FROM conversation_history
            WHERE session_id = $1 AND created_at > $2
            ORDER BY created_at
            """,
            session_id,
            since,
        )

        return [
            Message(
                role=MessageRole(row["role"]),
                content=row["content"],
                metadata=json.loads(row["metadata"])
                if row["metadata"] and isinstance(row["metadata"], str)
                else (row["metadata"] or {}),
            )
            for row in reversed(rows)
        ]
