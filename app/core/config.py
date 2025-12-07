from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://username:password@localhost:5432/user_coffe_shop"
    COFFEE_DB_URL: str = "postgresql://username:password@localhost:5432/coffe_shop"
    REDIS_URL: str = "redis://localhost:6379"

    # Legacy DB config (for backward compatibility)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "coffe_shop"
    DB_NAME_USER: str = "user_coffe_shop"
    DB_USER: str = "username"
    DB_PASSWORD: str = "password"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # LLM
    LLM_API_KEY: str = "dummy-key"
    LLM_MODEL: str = "deepseek-ai/DeepSeek-R1-0528"
    BASE_URL_LLM: str = "https://api.openai.com/v1"

    # Embedding
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIMENSION: int = 1024

    # Memory
    MAX_CONVERSATION_HISTORY: int = 20
    SUMMARY_TOKEN_LIMIT: int = 1000
    SHORT_TERM_MEMORY_WINDOW: int = 5

    # Vector Search
    VECTOR_SEARCH_TOP_K: int = 3
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 64

    # Query Execution
    MAX_RETRY_ATTEMPTS: int = 3
    QUERY_TIMEOUT: int = 30

    # Timeouts
    LLM_TIMEOUT: int = 60
    REQUEST_TIMEOUT: int = 120

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
