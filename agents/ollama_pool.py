import asyncio
import time
import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    SYSTEM = 0
    COUNSELOR = 10
    OBSERVER = 20


@dataclass(order=True)
class InferenceRequest:
    priority: RequestPriority
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    agent_id: str = field(compare=False)
    future: asyncio.Future = field(compare=False, default=None)
    created_at: float = field(compare=False, default_factory=time.monotonic)


class InferencePool:
    """Single-worker priority queue with cloud fallback for ollama CPU bottleneck."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        cloud_url: Optional[str] = None,
        cloud_key: Optional[str] = None,
        max_queue: int = 200,
        ollama_timeout: float = 30.0,
        cloud_timeout: float = 15.0,
    ):
        self.ollama_url = ollama_url
        self.cloud_url = cloud_url or os.environ.get("CLOVA_API_URL")
        self.cloud_key = cloud_key or os.environ.get("CLOVA_API_KEY")
        self.max_queue = max_queue
        self.ollama_timeout = ollama_timeout
        self.cloud_timeout = cloud_timeout
        self._queue: asyncio.PriorityQueue = None
        self._worker: asyncio.Task = None
        self._running = False
        self._request_counter = 0
        self._semaphore: asyncio.Semaphore = None
        self._stats = {
            "total": 0,
            "completed_ollama": 0,
            "completed_cloud": 0,
            "dropped": 0,
            "avg_latency_ms": 0,
            "queue_size": 0,
            "max_queue": max_queue,
        }

    async def start(self):
        self._queue = asyncio.PriorityQueue(maxsize=self.max_queue)
        self._semaphore = asyncio.Semaphore(1)
        self._running = True
        self._worker = asyncio.create_task(self._worker_loop())
        logger.info(f"InferencePool started: ollama={self.ollama_url}, cloud={'enabled' if self.cloud_url else 'disabled'}")

    async def stop(self):
        self._running = False
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        logger.info("InferencePool stopped")

    async def submit(self, prompt: str, agent_id: str, priority: RequestPriority = RequestPriority.COUNSELOR, timeout: float = 60.0) -> str:
        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        future = asyncio.get_event_loop().create_future()
        req = InferenceRequest(
            priority=priority,
            request_id=request_id,
            prompt=prompt,
            agent_id=agent_id,
            future=future,
        )
        try:
            self._queue.put_nowait(req)
            self._stats["total"] += 1
            self._stats["queue_size"] = self._queue.qsize()
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            raise Exception(f"Queue full ({self.max_queue}), request dropped")

        return await asyncio.wait_for(future, timeout=timeout)

    async def _worker_loop(self):
        import httpx

        async with httpx.AsyncClient(timeout=self.ollama_timeout) as ollama_client:
            cloud_client_ctx = None
            cloud_client = None
            if self.cloud_url:
                cloud_client_ctx = httpx.AsyncClient(
                    base_url=self.cloud_url,
                    headers={"X-NCP-CLOVASTUDIO-API-KEY": self.cloud_key or ""} if self.cloud_key else {},
                    timeout=self.cloud_timeout,
                )
                cloud_client = await cloud_client_ctx.__aenter__()

            try:
                while self._running:
                    try:
                        req = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    self._stats["queue_size"] = self._queue.qsize()

                    t0 = time.monotonic()
                    result = None

                    try:
                        resp = await ollama_client.post(f"{self.ollama_url}/api/generate", json={
                            "model": "qwen2.5:3b",
                            "prompt": req.prompt,
                            "stream": False,
                            "options": {"temperature": 0.7, "num_predict": 200},
                        })
                        resp.raise_for_status()
                        result = resp.json().get("response", "")
                        self._stats["completed_ollama"] += 1

                    except Exception as ollama_err:
                        logger.warning(f"[pool] ollama failed ({ollama_err}), trying cloud fallback")
                        if cloud_client and self.cloud_url:
                            try:
                                resp = await cloud_client.post("/testapp/v3/chat-completions/HCX-005", json={
                                    "messages": [{"role": "user", "content": req.prompt}],
                                    "topP": 0.8, "topK": 0, "temperature": 0.7,
                                    "maxTokens": 200, "repeatPenalty": 1.1,
                                })
                                resp.raise_for_status()
                                data = resp.json()
                                result = data["result"]["message"]["content"]
                                self._stats["completed_cloud"] += 1
                            except Exception as cloud_err:
                                logger.error(f"[pool] cloud fallback also failed: {cloud_err}")
                                req.future.set_exception(Exception(f"All backends failed: ollama={ollama_err}, cloud={cloud_err}"))
                                continue
                        else:
                            req.future.set_exception(ollama_err)
                            continue

                    elapsed_ms = (time.monotonic() - t0) * 1000
                    n = self._stats["completed_ollama"] + self._stats["completed_cloud"]
                    self._stats["avg_latency_ms"] = (
                        self._stats["avg_latency_ms"] * (n - 1) + elapsed_ms
                    ) / n if n > 0 else elapsed_ms

                    req.future.set_result(result)
                    logger.debug(f"[pool] {req.request_id} done in {elapsed_ms:.0f}ms")

            finally:
                if cloud_client_ctx:
                    await cloud_client_ctx.__aexit__(None, None, None)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "queue_size": self._queue.qsize() if self._queue else 0,
        }


_pool: Optional[InferencePool] = None


def get_pool() -> InferencePool:
    global _pool
    if _pool is None:
        _pool = InferencePool()
    return _pool


async def init_pool():
    pool = get_pool()
    await pool.start()
    return pool


async def shutdown_pool():
    global _pool
    if _pool:
        await _pool.stop()
        _pool = None
