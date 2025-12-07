import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        logger.info("Initializing LLM client...")
        try:
            self.client = AsyncOpenAI(
                api_key=settings.LLM_API_KEY, base_url=getattr(settings, "BASE_URL_LLM", None)
            )
            self.model = settings.LLM_MODEL
            logger.info("LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        timeout: int = 60,
        **kwargs,
    ) -> str:
        try:
            messages = []

            if system:
                messages.append({"role": "system", "content": system})

            messages.append({"role": "user", "content": prompt})

            params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
                "timeout": timeout,
                **kwargs,
            }

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**params), timeout=timeout
            )

            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {timeout}s")
            raise Exception(f"LLM request timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
