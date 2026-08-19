"""
记忆 시스템 - 에이전트의 기억을 관리하는 모듈

기억의 종류:
1. Working Memory (작업 기억): 현재 컨텍스트, 즉시 처리 중인 정보
2. Episodic Memory (에피소드 기억): 특정 시간/장소에서의 경험
3. Semantic Memory (의미 기억): 추상화된 지식, 개념
4. Social Memory (사회 기억): 다른 에이전트와의 관계/상호작용
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import deque
import json
import math


@dataclass
class MemoryEvent:
    """개별 기억 이벤트"""
    timestamp: float          # 시뮬레이션 시간
    event_type: str           # 'conversation', 'observation', 'reflection', 'plan'
    content: str              # 기억 내용
    participants: List[str]   # 관련 에이전트들
    importance: float = 0.5   # 중요도 (0-1)
    emotion: str = 'neutral'  # 감정 상태
    context: Dict = field(default_factory=dict)  # 추가 컨텍스트
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'content': self.content,
            'participants': self.participants,
            'importance': self.importance,
            'emotion': self.emotion,
            'context': self.context
        }


@dataclass
class Relation:
    """에이전트 간 관계"""
    target: str               # 대상 에이전트 이름
    trust: float = 0.5        # 신뢰도 (0-1)
    familiarity: float = 0.0  # 친밀도 (0-1)
    last_interaction: float = 0.0  # 마지막 상호작용 시간
    interaction_count: int = 0     # 상호작용 횟수
    shared_memories: List[str] = field(default_factory=list)  # 공유된 기억


class MemorySystem:
    """
    에이전트의 기억 시스템
    
    시간에 따라:
    - 새 기억은 working memory에 저장
    - 중요도가 높은 기억은 long-term memory로 승격
    - 오래된 기억은 점차 희미해짐 (consolidation)
    - 특정 키워드/상황으로 관련 기억을 회상 (retrieval)
    """
    
    def __init__(self, agent_name: str, working_memory_size: int = 10, 
                 long_term_memory_size: int = 100):
        self.agent_name = agent_name
        self.working_memory: deque = deque(maxlen=working_memory_size)
        self.long_term_memory: List[MemoryEvent] = []
        self.long_term_capacity = long_term_memory_size
        self.semantic_memory: Dict[str, Any] = {}  # 추상화된 지식
        self.social_memory: Dict[str, Relation] = {}  # 관계 기억
        self.current_context: Dict[str, Any] = {}  # 현재 컨텍스트
        
    def store_event(self, event: MemoryEvent):
        """새 이벤트를 기억에 저장"""
        # working memory에 즉시 저장
        self.working_memory.append(event)
        
        # 중요도가 높으면 long-term memory로
        if event.importance > 0.7:
            self._consolidate_to_long_term(event)
        
        # 관계 업데이트
        for participant in event.participants:
            if participant != self.agent_name:
                self._update_relation(participant, event)
    
    def _consolidate_to_long_term(self, event: MemoryEvent):
        """중요한 기억을 장기 기억으로 이동"""
        if len(self.long_term_memory) >= self.long_term_capacity:
            # 가장 덜 중요한 기억 제거
            self.long_term_memory.sort(key=lambda e: e.importance)
            self.long_term_memory.pop(0)
        self.long_term_memory.append(event)
    
    def _update_relation(self, other_agent: str, event: MemoryEvent):
        """관계 기억 업데이트"""
        if other_agent not in self.social_memory:
            self.social_memory[other_agent] = Relation(target=other_agent)
        
        relation = self.social_memory[other_agent]
        relation.last_interaction = event.timestamp
        relation.interaction_count += 1
        
        # 상호작용할수록 친밀도 증가
        relation.familiarity = min(1.0, relation.familiarity + 0.05)
        
        # 긍정적 감정이면 신뢰도 증가
        if event.emotion in ['positive', 'happy', 'grateful']:
            relation.trust = min(1.0, relation.trust + 0.03)
        elif event.emotion in ['negative', 'angry', 'sad']:
            relation.trust = max(0.0, relation.trust - 0.05)
    
    def recall(self, query: str, context: Dict = None, 
               top_k: int = 5) -> List[MemoryEvent]:
        """
        관련 기억을 회상
        
        회상 알고리즘:
        1. 키워드 매칭 (의미 유사도)
        2. 최근성 (recency) - 최근 기억 우선
        3. 중요도 (importance) - 중요한 기억 우선
        4. 감정적 강도 (emotional intensity)
        """
        all_memories = list(self.working_memory) + self.long_term_memory
        
        if not all_memories:
            return []
        
        scored_memories = []
        for memory in all_memories:
            score = self._calculate_relevance_score(memory, query, context)
            scored_memories.append((score, memory))
        
        # 점수 기반 정렬
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in scored_memories[:top_k]]
    
    def _calculate_relevance_score(self, memory: MemoryEvent, 
                                    query: str, context: Dict = None) -> float:
        """기억 관련성 점수 계산"""
        score = 0.0
        
        # 1. 키워드 매칭 (단순 문자열 포함)
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())
        overlap = len(query_words & memory_words)
        if query_words:
            score += (overlap / len(query_words)) * 0.3
        
        # 2. 참여자 매칭
        if context and 'speaker' in context:
            if context['speaker'] in memory.participants:
                score += 0.2
        
        # 3. 최근성 (최근일수록 높은 점수)
        if context and 'current_time' in context:
            time_diff = context['current_time'] - memory.timestamp
            recency_score = math.exp(-time_diff / 100)  # 감쇠 함수
            score += recency_score * 0.3
        
        # 4. 중요도
        score += memory.importance * 0.2
        
        return score
    
    def get_relationship_summary(self, other_agent: str) -> str:
        """특정 에이전트와의 관계 요약"""
        if other_agent not in self.social_memory:
            return f"{other_agent}와의 관계: 알지 못함"
        
        rel = self.social_memory[other_agent]
        
        trust_level = "높음" if rel.trust > 0.7 else "보통" if rel.trust > 0.3 else "낮음"
        familiarity_level = "친밀함" if rel.familiarity > 0.7 else "알고 지냄" if rel.familiarity > 0.3 else "낯섬"
        
        return (f"{other_agent}: 신뢰 {trust_level}({rel.trust:.2f}), "
                f"친밀도 {familiarity_level}({rel.familiarity:.2f}), "
                f"만남 {rel.interaction_count}회")
    
    def get_context_summary(self) -> str:
        """현재 컨텍스트 요약"""
        summary_parts = []
        
        # 최근 대화
        recent_conversations = [
            m for m in self.working_memory 
            if m.event_type == 'conversation'
        ][-3:]
        
        if recent_conversations:
            summary_parts.append("최근 대화:")
            for conv in recent_conversations:
                summary_parts.append(f"  - {conv.content[:50]}...")
        
        # 관계 요약
        if self.social_memory:
            summary_parts.append("\n알고 지내는 사람들:")
            for name, rel in self.social_memory.items():
                if rel.interaction_count > 0:
                    summary_parts.append(f"  - {self.get_relationship_summary(name)}")
        
        return "\n".join(summary_parts) if summary_parts else "아직 기억이 없습니다."
    
    def to_dict(self) -> Dict:
        """직렬화"""
        return {
            'agent_name': self.agent_name,
            'working_memory': [m.to_dict() for m in self.working_memory],
            'long_term_memory': [m.to_dict() for m in self.long_term_memory],
            'semantic_memory': self.semantic_memory,
            'social_memory': {
                name: {
                    'target': rel.target,
                    'trust': rel.trust,
                    'familiarity': rel.familiarity,
                    'last_interaction': rel.last_interaction,
                    'interaction_count': rel.interaction_count
                }
                for name, rel in self.social_memory.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemorySystem':
        """역직렬화"""
        memory = cls(data['agent_name'])
        
        for m_data in data.get('working_memory', []):
            memory.working_memory.append(MemoryEvent(**m_data))
        
        for m_data in data.get('long_term_memory', []):
            memory.long_term_memory.append(MemoryEvent(**m_data))
        
        memory.semantic_memory = data.get('semantic_memory', {})
        
        for name, rel_data in data.get('social_memory', {}).items():
            memory.social_memory[name] = Relation(**rel_data)
        
        return memory
