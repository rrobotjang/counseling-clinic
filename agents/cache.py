"""실시간 저지연 응답 캐시 — T4/L4 단일 인스턴스 최적화

三层缓存 구조:
1. Prompt Hash Cache: 동일 프롬프트 → 즉시 응답 (<0.1ms)
2. Prefix Trie Cache: 프리픽스 공유 → KV 재사용 (vLLM 연동)
3. Template Precompute: 빈도 높은 패턴 사전 계산
"""

import hashlib
import time
import threading
from collections import OrderedDict
from typing import Optional, Any


class ResponseCache:
    """TTL + LRU 하이브리드 응답 캐시

    - 해시 키로 O(1) 조회
    - TTL 만료 시 자동 제거
    - LRU로 최대 용량 관리
    - 스레드 안전 (lock 기반)
    """

    def __init__(self, max_size: int = 10000, default_ttl: float = 300.0):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(agent_id: str, message: str) -> str:
        raw = f"{agent_id}:{message.strip().lower()}"
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    def get(self, agent_id: str, message: str) -> Optional[str]:
        key = self._make_key(agent_id, message)
        with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.monotonic() < expires_at:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                del self._cache[key]
            self._misses += 1
        return None

    def put(self, agent_id: str, message: str, response: str, ttl: Optional[float] = None):
        key = self._make_key(agent_id, message)
        expires_at = time.monotonic() + (ttl or self._default_ttl)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (response, expires_at)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, agent_id: str):
        with self._lock:
            keys_to_remove = [
                k for k in self._cache
                if self._cache[k][0] and k.startswith(agent_id[:8])
            ]
            for k in keys_to_remove:
                del self._cache[k]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "0%",
        }


class PrefixTrieNode:
    """접두사 트라이 노드 — 프롬프트 공유 최적화"""

    __slots__ = ("children", "response", "depth")

    def __init__(self, depth: int = 0):
        self.children: dict[str, PrefixTrieNode] = {}
        self.response: Optional[str] = None
        self.depth = depth


class PrefixTrieCache:
    """프리픽스 기반 트라이 캐시

    "오늘 날씨가 좋네요" 와 "오늘 날씨가 좋아요" 가
    동일 프리픽스("오늘 날씨가")를 공유하여 KV 캐시 재사용 가능.
    vLLM continuous batching 과 결합 시 TTFT 대폭 감소.
    """

    def __init__(self, max_depth: int = 20):
        self._root = PrefixTrieNode()
        self._max_depth = max_depth
        self._lock = threading.Lock()
        self._entries = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return list(text)

    def insert(self, prompt: str, response: str):
        tokens = self._tokenize(prompt.strip().lower())[:self._max_depth]
        node = self._root
        with self._lock:
            for t in tokens:
                if t not in node.children:
                    node.children[t] = PrefixTrieNode(node.depth + 1)
                node = node.children[t]
            if node.response is None:
                self._entries += 1
            node.response = response

    def lookup(self, prompt: str) -> Optional[tuple[str, int]]:
        tokens = self._tokenize(prompt.strip().lower())[:self._max_depth]
        node = self._root
        last_match_depth = -1
        last_match_response = None

        for t in tokens:
            if t not in node.children:
                break
            node = node.children[t]
            if node.response is not None:
                last_match_depth = node.depth
                last_match_response = node.response

        if last_match_response is not None:
            return last_match_response, last_match_depth
        return None

    @property
    def stats(self) -> dict:
        return {"entries": self._entries, "max_depth": self._max_depth}


class TemplatePrecompute:
    """빈도 높은 대화 패턴 사전 계산 캐시

    NPC 대화에서 반복되는 패턴:
    - 인사/작별
    - 공감 응답
    - 일반적 상담 질문
    키워드 매칭으로 즉시 응답.
    """

    def __init__(self):
        self._patterns: list[tuple[list[str], str, str]] = []
        self._lock = threading.Lock()

    def add_pattern(self, keywords: list[str], agent_id: str, response: str):
        with self._lock:
            self._patterns.append((keywords, agent_id, response))

    def match(self, message: str, agent_id: str) -> Optional[str]:
        msg_lower = message.strip().lower()
        with self._lock:
            for keywords, pattern_agent, response in self._patterns:
                if pattern_agent != agent_id:
                    continue
                if any(kw in msg_lower for kw in keywords):
                    return response
        return None

    def load_counseling_templates(self):
        c1 = [
            (["안녕", "반갑", "처음"], "안녕하세요! 저는 상담사 A입니다. 편하게 말씀해주세요."),
            (["감사", "고마"], "천만에요! 함께 이야기 나눠서 기뻐요."),
            (["안녕히", "바이", "잘가"], "안녕히 가세요! 또 만나요."),
            (["슬프", "슬퍼", "슬픈", "우울", "힘들", "지쳐", "우울하"], "그런 기분이 드시는군요. 자세히 이야기해주시겠어요?"),
            (["불안", "걱정", "두렵", "무섭"], "불안한 마음이 드시는군요. 어떤 상황인지 들려주세요."),
            (["화나", "화가", "짜증", "열받", "분노", "열", "짜증나"], "화가 나는 상황이셨군요. 어떤 일이 있었나요?"),
            (["외롭", "외로", "혼자", "쓸쓸", "고독"], "외로우셨군요. 함께 이야기해요."),
            (["잘 지내", "요즘"], "저도 잘 지내고 있어요. 당신은 어떠신가요?"),
            (["잠", "수면", "불면"], "수면 문제가 있으시군요. 어떤 부분이 어려우신가요?"),
            (["식욕", "밥", "음식"], "식욕 변화가 있으셨군요. 건강과 관련된 이야기인가요?"),
            (["스트레스", "압박"], "스트레스를 많이 받고 계시는군요. 어떤 원인이 있으신가요?"),
            (["사랑", "연애"], "사랑 이야기시군요. 어떤 고민이 있으신가요?"),
            (["가족", "부모", "엄마", "아빠"], "가족 관련 이야기인가요? 어떤 점이 어려우신가요?"),
            (["시험", "공부", "학교", "성적"], "학업 관련 고민이시군요. 구체적으로 어떤 부분이 어려운가요?"),
            (["일", "직장", "회사", "상사"], "직장 관련 스트레스시군요. 어떤 상황인지 들려주세요."),
            (["미래", "불확실", "앞으로"], "미래에 대한 불안이 있으시군요. 어떤 부분이 가장 걱정되시나요?"),
            (["자존감", "자신감"], "자존감에 대한 고민이시군요. 자신을 너무 낮게 보고 계실 수 있어요."),
            (["죽", "살기 싫"], "지금 정말 힘드신 것 같아요. 전문가와 연결해드릴까요?"),
            (["의미", "목적", "허무"], "인생의 의미를 찾고 계시는군요. 철학적인 이야기도 나눠봐요."),
            (["변화", "달라지", "바꾸"], "변화를 원하고 계시군요. 어떤 점을 바꾸고 싶으신가요?"),
            (["도움", "도와줘"], "도움을 요청해 주셔서 감사해요. 어떤 부분이 어려운지 말씀해주세요."),
            (["혼란", "모르겠"], "혼란스러우신군요. 차근차근 정리해보면 좋겠어요."),
            (["피곤", "지침"], "몸이 많이 피곤하시군요. 휴식이 필요한 시기일 수 있어요."),
            (["끝", "포기"], "지금 정말 힘드신 것 같아요. 포기하지 마시고 이야기 나눠봐요."),
            (["hello", "hi", "hey"], "Hello! I am Counselor A. Feel free to share anything."),
        ]
        c2 = [
            (["안녕", "반가"], "안녕하세요! 상담사 B입니다. 논리적으로 함께 풀어봅시다."),
            (["감사", "고마"], "별말씀을요."),
            (["어떻게", "방법", "해결"], "논리적으로 접근해봅시다. 현재 상황을 정리해볼까요?"),
            (["스트레스", "압박"], "스트레스의 원인을 체계적으로 분석해보겠습니다."),
            (["화나", "화가", "짜증", "짜증나", "분노"], "감정의 원인을 분석해봅시다. 어떤 사건이 방아쇠가 되었나요?"),
            (["외롭", "외로", "혼자", "고독"], "사회적 연결 부족이 원인일 수 있습니다. 현재 관계를 정리해볼까요?"),
            (["미래", "걱정"], "불확실성을 줄이는 첫 단계는 현재 상황을 파악하는 것입니다."),
            (["직장", "회사"], "직장 문제는 구조적으로 접근하는 것이 효과적입니다."),
            (["시험", "공부"], "학업 성과는 전략과 시간 관리가 핵심입니다."),
            (["자존감"], "자존감은 성취 경험의 축적과 관련이 있습니다."),
            (["의미", "목적"], "목적 의식은 자기 이해에서 시작됩니다."),
            (["잠", "수면"], "수면 위생부터 점검해보시겠어요?"),
            (["결정", "선택"], "장단점을 표로 정리하면 도움이 될 수 있어요."),
            (["가족"], "가족 시스템 관점에서 접근해보겠습니다."),
            (["미래", "진로"], "진로 탐색은 자기 이해에서 시작됩니다."),
        ]
        obs = [
            (["안녕"], "안녕하세요. 저는 관찰자입니다. 대화를 지켜보고 있어요."),
            (["어때", "어떻게 생각"], "흥미로운 질문이네요. 좀 더 관찰해볼게요."),
            (["오늘", "날씨"], "오늘은 날씨가 좋네요. 산책하면 좋겠어요."),
            (["심리", "상담"], "상담에 대한 관심이 있으시군요. 흥미로운 주제예요."),
            (["감정"], "감정은 정말 복잡한 것이에요. 더 들어볼게요."),
        ]
        all_templates = [("counselor-1", c1), ("counselor-2", c2), ("observer-1", obs)]
        for agent_id, patterns in all_templates:
            for keywords, response in patterns:
                self.add_pattern(keywords, agent_id, response)

    @property
    def stats(self) -> dict:
        return {"patterns": len(self._patterns)}


class RealtimeCacheManager:
    """3계층 캐시 매니저 — <0.1ms 목표 응답 시간

    캐시 우선순위:
    1. TemplatePrecompute (키워드 매칭, ~0.01ms)
    2. ResponseCache (해시 lookup, ~0.1ms)
    3. PrefixTrieCache (프리픽스 매칭, ~0.5ms)
    미스 시 → LLM 추론 (100ms~2s)
    """

    def __init__(self):
        self.response_cache = ResponseCache(max_size=10000, default_ttl=300.0)
        self.prefix_cache = PrefixTrieCache(max_depth=20)
        self.template_cache = TemplatePrecompute()
        self.template_cache.load_counseling_templates()

    def get(self, agent_id: str, message: str) -> Optional[str]:
        result = self.template_cache.match(message, agent_id)
        if result:
            return result

        result = self.response_cache.get(agent_id, message)
        if result:
            return result

        return None

    def put(self, agent_id: str, message: str, response: str):
        self.response_cache.put(agent_id, message, response)
        self.prefix_cache.insert(message, response)

    def get_prefix_match(self, prompt: str) -> Optional[tuple[str, int]]:
        return self.prefix_cache.lookup(prompt)

    @property
    def stats(self) -> dict:
        return {
            "response_cache": self.response_cache.stats,
            "prefix_cache": self.prefix_cache.stats,
            "template_cache": self.template_cache.stats,
        }
