"""
내담자 에이전트 - 감정 상태와 기억을 가진 AI 내담자

심리상담 클리닉에서 실제 내담자처럼 행동:
1. 감정 상태 변화
2. 기억 축적
3. 상담사에 대한 신뢰 형성
4. 자기표현 수준 변화
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class EmotionalState(Enum):
    """감정 상태"""
    ANXIOUS = "불안"
    SAD = "우울"
    ANGRY = "분노"
    HOPEFUL = "희망적"
    TRUSTING = "신뢰"
    NEUTRAL = "무감정"


@dataclass
class Memory:
    """상담 기억"""
    session_id: int
    time: int
    content: str
    emotion_at_time: Dict[str, float]
    importance: float


class ClientAgent:
    """
    내담자 에이전트
    
    역할:
    1. 상담사의 질문에 자연스럽게 반응
    2. 감정 상태가 시간에 따라 변화
    3. 기억을 바탕으로 과거 경험 공유
    4. 신뢰 수준에 따라 자기표현 깊이 변화
    """
    
    def __init__(self, 
                 name: str,
                 age: int,
                 gender: str,
                 issues: List[str],
                 background: str):
        
        self.name = name
        self.age = age
        self.gender = gender
        self.issues = issues
        self.background = background
        
        self.emotional_state: Dict[str, float] = {
            'anxiety': 0.5,
            'sadness': 0.3,
            'anger': 0.2,
            'hope': 0.4,
            'trust': 0.2
        }
        
        self.energy: float = 0.6
        self.openness: float = 0.3
        self.session_count: int = 0
        self.memory: List[Memory] = []
        
        self.response_templates = self._build_response_templates()
    
    def _build_response_templates(self) -> Dict[str, List[str]]:
        """반응 템플릿 구축"""
        templates = {
            "anxiety_high": [
                "متأسفم... Really nervous.",
                "이게 맞는지 모르겠어요.",
                "조금 불안하네요.",
                "걱정이 많이 되요."
            ],
            "anxiety_low": [
                "좀 나아진 것 같아요.",
                "그래도 괜찮은 것 같아요.",
                "아까보다는 덜 불안해요."
            ],
            "sadness_high": [
                " really sad... 힘들어요.",
                "why this happen...",
                "我真的 힘들어요.",
                "为什么这样...",
                "我为什么这么难过..."
            ],
            "sadness_low": [
                "조금 좋아진 것 같아요.",
                "울_city가 좀 나아졌어요.",
                "아까보다는 괜찮아요."
            ],
            "hope_high": [
                "해볼 수 있을 것 같아요!",
                "좋은 생각이에요!",
                "시도해 보고 싶어요!",
                "hope has_been"
            ],
            "hope_low": [
                "hard to think...",
                "don't know...",
                "어렵네요..."
            ],
            "trust_high": [
                "你说得对...",
                "really helpful",
                "감사합니다.",
                "도움이 돼요."
            ],
            "trust_low": [
                "글쎄요...",
                "잘 모르겠어요.",
                "Maybe..."
            ],
            "general": [
                "네...",
                "글쎄요...",
                "그런가 봐요.",
                "hm...",
                "네, 맞아요.",
                "좀 그렇네요.",
                "그럴 수도 있겠네요."
            ]
        }
        return templates
    
    def respond(self, counselor_message: str, counselor_category: str) -> Dict:
        """
        상담사 메시지에 대한 반응 생성
        
        Returns:
            {
                'text': str,
                'emotion_change': Dict[str, float],
                'openness_change': float,
                'memory_add': bool
            }
        """
        emotion_change = {}
        openness_change = 0.0
        memory_add = False
        
        if counselor_category == "공감":
            self.emotional_state['trust'] = min(1.0, self.emotional_state['trust'] + 0.08)
            self.emotional_state['sadness'] = max(0.0, self.emotional_state['sadness'] - 0.05)
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.03)
            openness_change = 0.05
            memory_add = True
            
        elif counselor_category == "질문":
            self.emotional_state['anxiety'] = min(1.0, self.emotional_state['anxiety'] + 0.03)
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.02)
            openness_change = 0.03
            memory_add = True
            
        elif counselor_category == "반영":
            self.emotional_state['trust'] = min(1.0, self.emotional_state['trust'] + 0.1)
            self.emotional_state['sadness'] = max(0.0, self.emotional_state['sadness'] - 0.03)
            openness_change = 0.08
            memory_add = True
            
        elif counselor_category == "해석":
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.05)
            self.emotional_state['anxiety'] = max(0.0, self.emotional_state['anxiety'] - 0.03)
            openness_change = 0.04
            memory_add = True
            
        elif counselor_category == "지시":
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.04)
            self.energy = min(1.0, self.energy + 0.02)
            
        elif counselor_category == "정보제공":
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.03)
            self.emotional_state['anxiety'] = max(0.0, self.emotional_state['anxiety'] - 0.02)
        
        self.openness = min(1.0, max(0.0, self.openness + openness_change))
        self.energy = max(0.1, min(1.0, self.energy - 0.02))
        
        response_text = self._generate_response(counselor_category)
        
        return {
            'text': response_text,
            'emotion_change': emotion_change,
            'openness_change': openness_change,
            'memory_add': memory_add
        }
    
    def _generate_response(self, category: str) -> str:
        """반응 텍스트 생성"""
        dominant_emotion = max(self.emotional_state, key=self.emotional_state.get)
        emotion_value = self.emotional_state[dominant_emotion]
        
        if dominant_emotion == 'anxiety' and emotion_value > 0.6:
            pool = self.response_templates['anxiety_high']
        elif dominant_emotion == 'sadness' and emotion_value > 0.6:
            pool = self.response_templates['sadness_high']
        elif dominant_emotion == 'hope' and emotion_value > 0.6:
            pool = self.response_templates['hope_high']
        elif dominant_emotion == 'trust' and emotion_value > 0.5:
            pool = self.response_templates['trust_high']
        else:
            pool = self.response_templates['general']
        
        return random.choice(pool)
    
    def add_memory(self, session_id: int, time: int, content: str, importance: float = 0.5):
        """기억 추가"""
        memory = Memory(
            session_id=session_id,
            time=time,
            content=content,
            emotion_at_time=self.emotional_state.copy(),
            importance=importance
        )
        self.memory.append(memory)
        
        if len(self.memory) > 50:
            self.memory.sort(key=lambda x: x.importance, reverse=True)
            self.memory = self.memory[:30]
    
    def get_state_vector(self) -> list:
        """상태 벡터 반환"""
        return [
            self.emotional_state['anxiety'],
            self.emotional_state['sadness'],
            self.emotional_state['anger'],
            self.emotional_state['hope'],
            self.emotional_state['trust'],
            self.energy,
            self.openness,
            min(self.session_count / 10, 1.0),
            min(len(self.memory) / 20, 1.0)
        ]
    
    def get_summary(self) -> Dict:
        """내담자 요약"""
        return {
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'issues': self.issues,
            'background': self.background,
            'emotional_state': self.emotional_state.copy(),
            'energy': self.energy,
            'openness': self.openness,
            'session_count': self.session_count,
            'memory_count': len(self.memory)
        }
    
    def start_new_session(self):
        """새 세션 시작"""
        self.session_count += 1
        self.energy = min(1.0, self.energy + 0.1)


# 기본 내담자 프로필
DEFAULT_CLIENTS = [
    {
        'name': '김민지',
        'age': 28,
        'gender': '여',
        'issues': ['불안', '직장 스트레스'],
        'background': '대기업 근무, 업무 스트레스로 불안 증상 호소'
    },
    {
        'name': '박준호',
        'age': 35,
        'gender': '남',
        'issues': ['우울', '대인관계'],
        'background': '이혼 후 우울감, 대인관계 어려움'
    },
    {
        'name': '이서연',
        'age': 22,
        'gender': '여',
        'issues': ['자존감', '불안'],
        'background': '대학생, 자기CONFIDENCE 부족, 취업 불안'
    },
    {
        'name': '최현우',
        'age': 42,
        'gender': '남',
        'issues': ['스트레스', '수면장애'],
        'background': '자영업자, 경제적 어려움, 수면 문제'
    },
    {
        'name': '정수빈',
        'age': 19,
        'gender': '여',
        'issues': ['트라우마', '우울'],
        'background': '학교 폭력 경험, 트라우마'
    }
]


def create_default_clients() -> Dict[str, ClientAgent]:
    """기본 내담자 세트 생성"""
    clients = {}
    for profile in DEFAULT_CLIENTS:
        client = ClientAgent(
            name=profile['name'],
            age=profile['age'],
            gender=profile['gender'],
            issues=profile['issues'],
            background=profile['background']
        )
        clients[client.name] = client
    return clients
