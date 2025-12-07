# Text-to-SQL Chatbot

## Features

- **Natural Language to SQL**: Converts user questions into SQL queries using LLM reasoning
- **Vector Similarity Search**: Finds similar past questions using pgvector with HNSW indexing
- **Memory Management**: Maintains conversation context with long-term and short-term memory
- **Intelligent Clarification**: Asks for clarification when queries are ambiguous
- **Coffee Shop Dataset**: Pre-loaded with coffee shop sales data for demonstration

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- uv package manager

### Install uv package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies
```bash
uv sync
```

### Environment Setup
1. Copy `.env.example` to `.env`
2. Configure your database connection and OpenAI API key

### Database Setup
```bash
# Create database and enable pgvector extension
psql -U your_username -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create tables
psql -U your_username -d your_database -f schema.sql
```

### Load Sample Data
```bash
uv run etl_coffe_shop.py
```

### Run the Application
```bash
uv run server.py
```

## Project Structure

```
text-to-sql-chatbot/
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore file
├── pyproject.toml           # Python project configuration
├── README.md                # This file
├── schema.sql               # Database schema
├── server.py                # FastAPI application entry point
├── etl_coffe_shop.py        # ETL script for sample data
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app configuration
│   ├── api/
│   │   ├── deps.py          # FastAPI dependencies
│   │   └── routes/          # API endpoints
│   │       ├── __init__.py
│   │       ├── chat.py      # Chat endpoint
│   │       └── health.py    # Health check endpoint
│   ├── core/
│   │   ├── config.py        # Application configuration
│   │   ├── database.py      # Database connection
│   │   ├── llm_client.py    # OpenAI API client
│   │   ├── memory_manager.py # Conversation memory management
│   │   ├── query_processor.py # LLM reasoning engine
│   │   └── vector_store.py  # Vector database operations
│   ├── models/
│   │   ├── database.py      # Database models
│   │   └── schemas.py       # Pydantic schemas
│   └── services/
│       ├── __init__.py
│       ├── clarification_service.py # Query clarification
│       ├── retrieval_service.py    # Vector similarity search
│       ├── schema_extractor.py     # Database schema extraction
│       └── sql_executor.py         # SQL query execution
└── data/                    # Sample CSV data files
    ├── customer.csv
    ├── product.csv
    ├── sales_outlet.csv
    └── ...
```

## Key Components

### Core Modules
- **`memory_manager.py`**: Handles conversation memory with summarization and context management
- **`query_processor.py`**: LLM reasoning engine that determines action sequences
- **`vector_store.py`**: Manages pgvector operations with HNSW indexing for similarity search
- **`llm_client.py`**: OpenAI API client wrapper with error handling

### Services
- **`sql_executor.py`**: Executes SQL queries with retry mechanisms and safety checks
- **`schema_extractor.py`**: Extracts and formats database schema information
- **`retrieval_service.py`**: Searches for similar questions in the vector store
- **`clarification_service.py`**: Handles ambiguous queries and requests user clarification

### API Endpoints
- **`/chat`**: Main chat endpoint for natural language queries
- **`/health`**: Health check endpoint

## Architecture

The chatbot follows a multi-step reasoning process:

1. **Query Analysis**: Analyzes the user's natural language query
2. **Similarity Search**: Searches vector store for similar past questions
3. **Schema Retrieval**: Extracts relevant database schema information
4. **SQL Generation**: Uses LLM to generate appropriate SQL queries
5. **Execution**: Safely executes SQL with retry mechanisms
6. **Response Formatting**: Formats results for user-friendly display

## Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

### Type Checking
```bash
uv run mypy .
```
