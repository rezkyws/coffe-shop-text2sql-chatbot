import uvicorn

from app.core.config import get_settings


def main():
    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info",
    )


if __name__ == "__main__":
    main()
