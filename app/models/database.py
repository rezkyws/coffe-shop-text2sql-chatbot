import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ConversationHistory:
    id: Optional[uuid.UUID] = None
    session_id: str = ""
    role: str = ""  # 'user' or 'assistant'
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ConversationSummary:
    id: Optional[uuid.UUID] = None
    session_id: str = ""
    summary: str = ""
    total_interactions: int = 0
    last_updated: Optional[datetime] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class QueryVectorStore:
    id: Optional[uuid.UUID] = None
    question: str = ""
    sql_query: str = ""
    embedding: Optional[list] = None
    success_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class SchemaCache:
    id: Optional[uuid.UUID] = None
    table_name: str = ""
    schema_info: Optional[Dict[str, Any]] = None
    embedding: Optional[list] = None
    row_count: Optional[int] = None
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


@dataclass
class QueryExecutionLog:
    id: Optional[uuid.UUID] = None
    session_id: str = ""
    user_question: str = ""
    generated_sql: str = ""
    execution_status: str = ""
    error_message: Optional[str] = None
    result_rows: Optional[int] = None
    execution_time_ms: Optional[int] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4()
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
