"""
LLM API 연동 — 실시간 저지연 캐시 통합

T4/L4 최적화:
- 프리픽스 캐시로 반복 프롬프트 KV 재사용
- 템플릿 사전 계산으로 0ms 응답
- 해시 기반 응답 캐시로 <0.1ms 조회
"""

import os
import httpx
import time
from typing import Optional
from .cache import RealtimeCacheManager


class LLMClient:
    def __init__(self, provider: str = "auto", api_key: str = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = None
        self.base_url = None
        self.cache = RealtimeCacheManager()
        self._request_count = 0
        self._cache_hit_count = 0
        self._configure()

    def _configure(self):
        if self.provider == "auto":
            if self.api_key:
                if "naver" in self.api_key.lower() or len(self.api_key) > 50:
                    self.provider = "clova"
                else:
                    self.provider = "openai"
            else:
                self.provider = "ollama"

        if self.provider == "clova":
            self.base_url = "https://clovastudio.apigw.ntruss.com"
            self.model = "HCX-005"
        elif self.provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-3.5-turbo"
        elif self.provider == "ollama":
            self.base_url = "http://localhost:11434/v1"
            self.model = "qwen2.5:3b"
        else:
            self.provider = "template"

    def generate(self, prompt: str, system_prompt: str = None, max_tokens: int = 256, agent_id: str = "default") -> str:
        self._request_count += 1

        cached = self.cache.get(agent_id, prompt)
        if cached:
            self._cache_hit_count += 1
            return cached

        if self.provider == "template":
            response = self._template_response(prompt)
        elif self.provider == "clova":
            response = self._clova_generate(prompt, system_prompt, max_tokens)
        elif self.provider == "openai":
            response = self._openai_generate(prompt, system_prompt, max_tokens)
        elif self.provider == "ollama":
            response = self._ollama_generate(prompt, system_prompt, max_tokens)
        else:
            response = self._template_response(prompt)

        self.cache.put(agent_id, prompt, response)
        return response

    def _template_response(self, prompt: str) -> str:
        keywords = {
            "안녕": "안녕하세요! 오늘 기분은 어�신가요?",
            "상담": "상담을 진행하겠습니다. 어떤 고민이 있으신가요?",
            "도움": "도움을 드리겠습니다. 편하게 말씀해주세요.",
            "슬픔": "그렇게 느끼시는 건 당연합니다. 함께 이야기해요.",
            "불안": "불안한 마음이 드시는군요. 구체적으로 어떤 상황이 걱정되시나요?",
            "분노": "화가 나시는 상황이 있었군요. 자세히 들려주세요.",
            "외로움": "외로움을 느끼고 계시는군요. 여기서 함께 이야기해요.",
            "피곤": "피곤하시군요. 요즘 어떤 일로 힘드신가요?",
        }
        for keyword, response in keywords.items():
            if keyword in prompt:
                return response
        return f"말씀 감사합니다. \"{prompt[:30]}...\"에 대해 더 이야기해주세요."

    def _clova_generate(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        if not self.api_key:
            return self._template_response(prompt)

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}/test-apps/v1/chat-completions/HCX-005",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "messages": messages,
                        "maxTokens": max_tokens,
                        "temperature": 0.7,
                        "topP": 0.9
                    }
                )
                if resp.status_code == 200:
                    return resp.json()["result"]["message"]["content"]
        except Exception:
            pass
        return self._template_response(prompt)

    def _openai_generate(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        if not self.api_key:
            return self._template_response(prompt)

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return self._template_response(prompt)

    def _ollama_generate(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens
                    }
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return self._template_response(prompt)

    def get_status(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url,
            "requests": self._request_count,
            "cache_hits": self._cache_hit_count,
            "cache_stats": self.cache.stats,
        }
