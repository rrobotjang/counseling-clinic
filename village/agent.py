"""
에이전트 - 기억과 추론이 가능한 자율 에이전트

각 에이전트는:
1. 고유한 개성 (personality)을 가짐
2. 기억 시스템을 통해 과거를 기억함
3. 컨텍스트를 기반으로 추론함
4. 미래를 위한 플랜을 세울 수 있음
5. 다른 에이전트와 상호작용함
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import random
import json

from .memory import MemorySystem, MemoryEvent


class Emotion(Enum):
    """감정 상태"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    CURIOUS = "curious"
    ANXIOUS = "anxious"


class Personality:
    """
    에이전트의 개성
    
    특성값은 0-1 범위:
    - openness: 개방성 (새로운 경험 수용도)
    - conscientiousness: 성실성 (계획 실행력)
    - extraversion: 외향성 (사회적 상호작용 선호)
    - agreeableness: 친화성 (타인 협력도)
    - neuroticism: 신경성 (불안정성)
    """
    
    def __init__(self, 
                 openness: float = 0.5,
                 conscientiousness: float = 0.5,
                 extraversion: float = 0.5,
                 agreeableness: float = 0.5,
                 neuroticism: float = 0.5,
                 traits: Dict[str, float] = None):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        self.traits = traits or {}  # 사용자 정의 특성
    
    def get_social_tendency(self) -> float:
        """사회적 상호작용 성향"""
        return (self.extraversion * 0.6 + self.agreeableness * 0.4)
    
    def get_planning_tendency(self) -> float:
        """계획 성향"""
        return (self.conscientiousness * 0.7 + (1 - self.neuroticism) * 0.3)
    
    def to_dict(self) -> Dict:
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
            'traits': self.traits
        }


@dataclass
class Plan:
    """에이전트의 계획"""
    goal: str                    # 목표
    steps: List[str]             # 실행 단계들
    current_step: int = 0        # 현재 단계
    priority: float = 0.5        # 우선순위
    status: str = 'pending'      # pending, active, completed, failed
    
    def advance(self) -> bool:
        """다음 단계로 진행"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        return False
    
    def complete(self):
        """계획 완료"""
        self.status = 'completed'
        self.current_step = len(self.steps) - 1
    
    def to_dict(self) -> Dict:
        return {
            'goal': self.goal,
            'steps': self.steps,
            'current_step': self.current_step,
            'priority': self.priority,
            'status': self.status
        }


class Agent:
    """
    자율 에이전트
    
    능력:
    1. 기억: 과거 상호작용을 기억하고 회상
    2. 추론: 현재 컨텍스트를 기반으로 판단
    3. 계획: 미래 행동을 위한 플랜 수립
    4. 학습: 경험으로부터 개성/지식 업데이트
    5. 소통: 다른 에이전트와 대화
    """
    
    def __init__(self, 
                 name: str,
                 personality: Personality,
                 chatbot_model=None,
                 tokenizer=None,
                 system_prompt: str = ""):
        self.name = name
        self.personality = personality
        self.memory = MemorySystem(agent_name=name)
        
        # 챗봇 모델 (선택적)
        self.chatbot_model = chatbot_model
        self.tokenizer = tokenizer
        
        # 현재 상태
        self.current_emotion = Emotion.NEUTRAL
        self.current_activity = "idle"
        self.energy = 1.0  # 에너지 레벨
        
        # 플랜
        self.plans: List[Plan] = []
        self.completed_plans: List[Plan] = []
        
        # 지식
        self.knowledge: Dict[str, Any] = {}
        
        # 시스템 프롬프트
        self.system_prompt = system_prompt or self._generate_default_prompt()
    
    def _generate_default_prompt(self) -> str:
        """기본 시스템 프롬프트 생성"""
        return f"""당신은 '{self.name}'이라는 이름을 가진 에이전트입니다.
당신의 성격 특성:
- 개방성: {self.personality.openness:.2f}
- 성실성: {self.personality.conscientiousness:.2f}
- 외향성: {self.personality.extraversion:.2f}
- 친화성: {self.personality.agreeableness:.2f}

자연스럽고 대화하듯이 한국어로 응답하세요.
간결하게 1-2문장으로 답변하세요."""
    
    def think(self, context: Dict[str, Any]) -> str:
        """
        추론 과정
        
        1. 현재 컨텍스트 분석
        2. 관련 기억 회상
        3. 감정 상태 업데이트
        4. 판단/결정
        """
        # 1. 컨텍스트 분석
        current_situation = context.get('situation', '')
        speaker = context.get('speaker', '')
        current_time = context.get('current_time', 0)
        
        # 2. 관련 기억 회상
        relevant_memories = self.memory.recall(
            current_situation, 
            context={'speaker': speaker, 'current_time': current_time},
            top_k=3
        )
        
        # 3. 감정 상태 업데이트
        self._update_emotion(context, relevant_memories)
        
        # 4. 관계 정보
        relationship_info = ""
        if speaker and speaker in self.memory.social_memory:
            relationship_info = self.memory.get_relationship_summary(speaker)
        
        # 5. 생각 구성
        thought_parts = [f"상황: {current_situation}"]
        
        if relevant_memories:
            thought_parts.append("기억나는 것:")
            for mem in relevant_memories[:2]:
                thought_parts.append(f"  - {mem.content[:50]}")
        
        if relationship_info:
            thought_parts.append(f"관계: {relationship_info}")
        
        thought_parts.append(f"감정: {self.current_emotion.value}")
        
        return "\n".join(thought_parts)
    
    def _update_emotion(self, context: Dict, memories: List[MemoryEvent]):
        """감정 상태 업데이트"""
        # 기본적으로 중립 유지
        base_emotion = Emotion.NEUTRAL
        
        # 최근 기억이 긍정적이면 행복
        if memories and any(m.emotion in ['positive', 'happy'] for m in memories):
            base_emotion = Emotion.HAPPY
        
        # 에너지가 낮으면 피곤
        if self.energy < 0.3:
            base_emotion = Emotion.SAD
        
        # 신경성 높으면 불안
        if self.personality.neuroticism > 0.7 and random.random() < 0.3:
            base_emotion = Emotion.ANXIOUS
        
        self.current_emotion = base_emotion
    
    def plan_next_action(self, available_agents: List[str]) -> Optional[Plan]:
        """
        다음 행동 계획
        
        고려 요소:
        - 현재 감정/에너지
        - 성격 특성
        - 과거 상호작용
        - 목표/지식
        """
        # 에너지가 낮으면 휴식 계획
        if self.energy < 0.2:
            return Plan(
                goal="휴식하기",
                steps=["자리에 앉기", "쉬기", "에너지 회복하기"],
                priority=0.9
            )
        
        # 사회적 성향이 높으면 대화 시도
        social_tendency = self.personality.get_social_tendency()
        if social_tendency > 0.6 and random.random() < social_tendency:
            if available_agents:
                target = random.choice(available_agents)
                return Plan(
                    goal=f"{target}와 대화하기",
                    steps=[
                        f"{target} 찾기",
                        f"인사하기",
                        f"대화 시작하기",
                        f"이야기 나누기"
                    ],
                    priority=0.7
                )
        
        # 계획 성향이 높으면 무언가 시작
        if self.personality.conscientiousness > 0.6:
            activities = ["산책하기", "책 읽기", "주변 둘러보기", "생각 정리하기"]
            return Plan(
                goal=random.choice(activities),
                steps=[random.choice(activities)],
                priority=0.5
            )
        
        return None
    
    def interact_with(self, other_agent: 'Agent', message: str, 
                      current_time: float) -> str:
        """
        다른 에이전트와 상호작용
        
        과정:
        1. 관련 기억 확인
        2. 메시지 처리
        3. 응답 생성
        4. 기억 저장
        """
        # 1. 컨텍스트 구성
        context = {
            'situation': message,
            'speaker': other_agent.name,
            'current_time': current_time,
            'relationship': self.memory.get_relationship_summary(other_agent.name)
        }
        
        # 2. 생각하기
        thought = self.think(context)
        
        # 3. 응답 생성
        response = self._generate_response(message, other_agent, thought)
        
        # 4. 기억 저장
        event = MemoryEvent(
            timestamp=current_time,
            event_type='conversation',
            content=f"{other_agent.name}에게 '{response}'라고 응답",
            participants=[self.name, other_agent.name],
            importance=0.6,
            emotion=self.current_emotion.value,
            context={'message': message, 'thought': thought}
        )
        self.memory.store_event(event)
        
        # 5. 에너지 소모
        self.energy = max(0.1, self.energy - 0.05)
        
        return response
    
    def _generate_response(self, message: str, other_agent: 'Agent', 
                          thought: str) -> str:
        """
        응답 생성
        
        모델이 있으면 모델 사용, 없으면 규칙 기반
        """
        if self.chatbot_model and self.tokenizer:
            # 모델 기반 응답
            prompt = f"{self.system_prompt}\n\n{thought}\n\n상대방: {message}\n응답:"
            return self._call_chatbot(prompt)
        
        # 규칙 기반 응답 (모델 없을 때)
        return self._rule_based_response(message, other_agent)
    
    def _call_chatbot(self, prompt: str) -> str:
        """챗봇 모델 호출"""
        import torch
        
        device = next(self.chatbot_model.parameters()).device
        
        # 전처리 및 인코딩
        from app import preprocess_sentence, decoder_inference
        
        response_ids = decoder_inference(
            self.chatbot_model, 
            prompt, 
            self.tokenizer, 
            device
        )
        
        response = self.tokenizer.decode(
            [t for t in response_ids 
             if t != self.tokenizer.bos_id() 
             and t != self.tokenizer.eos_id() 
             and t != self.tokenizer.pad_id()]
        )
        
        return response
    
    def _rule_based_response(self, message: str, other_agent: 'Agent') -> str:
        """규칙 기반 응답 (모델 대체용)"""
        # 관계 확인
        relation = self.memory.social_memory.get(other_agent.name)
        
        # 인사 패턴
        greetings = ["안녕", "반가워", "좋은 하루"]
        farewells = ["잘 가", "안녕히", "다음에 또 봐"]
        
        # 감정 기반 응답
        emotion_responses = {
            Emotion.HAPPY: ["좋은 하루네!", "기분이 좋아!", "반가워!"],
            Emotion.SAD: ["...", "음...", "그래..."],
            Emotion.CURIOUS: ["흥미롭군", "더 알려줘", "그게 뭐야?"],
            Emotion.NEUTRAL: ["알겠어", "그래", "응"],
        }
        
        # 메시지 분석
        message_lower = message.lower()
        
        if any(g in message_lower for g in greetings):
            if relation and relation.familiarity > 0.5:
                return f"오 {other_agent.name}! 반가워! 잘 지냈어?"
            else:
                return f"안녕하세요, {other_agent.name}님."
        
        if any(f in message_lower for f in farewells):
            return "잘 가, 또 보자!"
        
        # 감정 기반 응답
        responses = emotion_responses.get(self.current_emotion, ["응"])
        
        # 성격 반영
        if self.personality.extraversion > 0.7:
            responses = [f"오! {r}" for r in responses]
        elif self.personality.agreeableness > 0.7:
            responses = [f"{r} 좋지!" for r in responses]
        
        return random.choice(responses)
    
    def observe(self, event_description: str, participants: List[str], 
                current_time: float, importance: float = 0.5):
        """환경 관찰 및 기억"""
        event = MemoryEvent(
            timestamp=current_time,
            event_type='observation',
            content=event_description,
            participants=participants,
            importance=importance,
            emotion=self.current_emotion.value
        )
        self.memory.store_event(event)
    
    def reflect(self, current_time: float):
        """
        성찰 - 과거 경험을 돌아보고 지식 추출
        
        주기적으로 호출되어:
        1. 최근 경험 정리
        2. 패턴 발견
        3. 지식 업데이트
        4. 플랜 조정
        """
        # 최근 기억 회상
        recent_memories = list(self.memory.working_memory)[-5:]
        
        if not recent_memories:
            return
        
        # 대화 패턴 분석
        conversations = [m for m in recent_memories if m.event_type == 'conversation']
        
        if conversations:
            # 누구와 대화했는지
            participants = set()
            for conv in conversations:
                participants.update(conv.participants)
            participants.discard(self.name)
            
            if participants:
                self.knowledge['recent_conversations_with'] = list(participants)
        
        # 미완료 플랜 확인
        for plan in self.plans:
            if plan.status == 'active':
                # 플랜 진행 상황 점검
                if random.random() < self.personality.conscientiousness:
                    plan.advance()
                    if plan.current_step >= len(plan.steps) - 1:
                        plan.complete()
                        self.completed_plans.append(plan)
    
    def get_status(self) -> Dict:
        """현재 상태 요약"""
        active_plans = [p for p in self.plans if p.status == 'active']
        
        return {
            'name': self.name,
            'emotion': self.current_emotion.value,
            'activity': self.current_activity,
            'energy': self.energy,
            'relationships': len(self.memory.social_memory),
            'memories': len(self.memory.working_memory) + len(self.memory.long_term_memory),
            'active_plans': len(active_plans),
            'completed_plans': len(self.completed_plans)
        }
    
    def to_dict(self) -> Dict:
        """직렬화"""
        return {
            'name': self.name,
            'personality': self.personality.to_dict(),
            'memory': self.memory.to_dict(),
            'current_emotion': self.current_emotion.value,
            'current_activity': self.current_activity,
            'energy': self.energy,
            'plans': [p.to_dict() for p in self.plans],
            'completed_plans': [p.to_dict() for p in self.completed_plans],
            'knowledge': self.knowledge,
            'system_prompt': self.system_prompt
        }
    
    @classmethod
    def from_dict(cls, data: Dict, chatbot_model=None, tokenizer=None) -> 'Agent':
        """역직렬화"""
        personality_data = data['personality']
        personality = Personality(**personality_data)
        
        agent = cls(
            name=data['name'],
            personality=personality,
            chatbot_model=chatbot_model,
            tokenizer=tokenizer,
            system_prompt=data.get('system_prompt', '')
        )
        
        agent.memory = MemorySystem.from_dict(data['memory'])
        agent.current_emotion = Emotion(data['current_emotion'])
        agent.current_activity = data['current_activity']
        agent.energy = data['energy']
        agent.plans = [Plan(**p) for p in data.get('plans', [])]
        agent.completed_plans = [Plan(**p) for p in data.get('completed_plans', [])]
        agent.knowledge = data.get('knowledge', {})
        
        return agent


def create_sample_agents() -> List[Agent]:
    """샘플 에이전트들 생성"""
    
    agents = [
        Agent(
            name="민수",
            personality=Personality(
                openness=0.8,
                conscientiousness=0.6,
                extraversion=0.7,
                agreeableness=0.8,
                neuroticism=0.3
            )
        ),
        Agent(
            name="지현",
            personality=Personality(
                openness=0.6,
                conscientiousness=0.9,
                extraversion=0.4,
                agreeableness=0.7,
                neuroticism=0.4
            )
        ),
        Agent(
            name="태웅",
            personality=Personality(
                openness=0.5,
                conscientiousness=0.7,
                extraversion=0.8,
                agreeableness=0.6,
                neuroticism=0.2
            )
        ),
        Agent(
            name="수진",
            personality=Personality(
                openness=0.9,
                conscientiousness=0.5,
                extraversion=0.6,
                agreeableness=0.9,
                neuroticism=0.5
            )
        ),
        Agent(
            name="현우",
            personality=Personality(
                openness=0.4,
                conscientiousness=0.8,
                extraversion=0.3,
                agreeableness=0.5,
                neuroticism=0.6
            )
        ),
    ]
    
    return agents
