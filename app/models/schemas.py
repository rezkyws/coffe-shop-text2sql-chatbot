from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question or query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How many coffees were sold last week?",
                "session_id": "user-123",
            }
        }


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sql_query: Optional[str] = None
    clarification_needed: bool = False
    clarification_questions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = {}


class QueryExecutionResult(BaseModel):
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    rows_affected: Optional[int] = None


class SimilarQuery(BaseModel):
    question: str
    sql_query: str
    similarity: float


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
