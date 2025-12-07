import asyncpg
from fastapi import APIRouter, Depends

from app.core.database import get_coffee_db_dependency, get_db_dependency

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@router.get("/health/db")
async def database_health(
    db: asyncpg.Connection = Depends(get_db_dependency),
    coffee_db: asyncpg.Connection = Depends(get_coffee_db_dependency),
):
    """Check database connectivity."""
    try:
        # Test main database
        await db.fetchval("SELECT 1")

        # Test coffee database
        await coffee_db.fetchval("SELECT 1")

        return {"status": "healthy", "main_db": "connected", "coffee_db": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
