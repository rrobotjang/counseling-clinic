import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL")


class RedisCache:
    def __init__(self):
        self._client = None
        self._available = False

    async def connect(self):
        if not REDIS_URL:
            logger.info("Redis not configured, using local cache only")
            return
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self._client.ping()
            self._available = True
            logger.info(f"Redis connected: {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")

    async def close(self):
        if self._client:
            await self._client.close()

    async def get(self, key: str) -> Optional[str]:
        if not self._available:
            return None
        try:
            return await self._client.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = 300):
        if not self._available:
            return
        try:
            await self._client.setex(key, ttl, value)
        except Exception:
            pass

    async def get_inference(self, agent_id: str, prompt: str) -> Optional[str]:
        key = f"inference:{agent_id}:{hash(prompt) % 100000}"
        return await self.get(key)

    async def set_inference(self, agent_id: str, prompt: str, response: str, ttl: int = 600):
        key = f"inference:{agent_id}:{hash(prompt) % 100000}"
        await self.set(key, response, ttl)

    @property
    def available(self) -> bool:
        return self._available


redis_cache = RedisCache()
