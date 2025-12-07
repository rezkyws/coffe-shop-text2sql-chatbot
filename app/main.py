import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health
from app.core.config import get_settings
from app.core.database import close_pools, create_pools, init_db

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set specific log levels for different components
logging.getLogger("app").setLevel(logging.DEBUG)
logging.getLogger("app.services").setLevel(logging.DEBUG)
logging.getLogger("app.core").setLevel(logging.DEBUG)
logging.getLogger("app.api").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def cleanup_resources():
    try:
        import os

        # Clean up any lingering multiprocessing resources
        os.system('pkill -f "loky|multiprocessing|sentence-transformers" 2>/dev/null || true')
    except Exception:
        pass


# Register cleanup function
import atexit

atexit.register(cleanup_resources)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Text-to-SQL Chatbot API...")
    try:
        logger.info("Creating pools...")
        await create_pools()
        logger.info("Pools created, initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        await asyncio.wait_for(close_pools(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out")


app = FastAPI(
    title="Text-to-SQL Chatbot API",
    description="AI powered chatbot for querying coffee shop database with natural language",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Text-to-SQL Chatbot API", "version": "0.1.0", "docs": "/docs"}
