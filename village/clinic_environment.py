"""
심리상담 클리닉 환경

내담자와 상담사가 세션을 가지는 환경:
1. 내담자: 다양한 심리 문제를 가진 캐릭터
2. 상담사: RL로 학습하는 상담 전략
3. 세션: 상담 과정 시뮬레이션
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class CounselingPhase(Enum):
    """상담 단계"""
    GREETING = "인사"
    PROBLEM_IDENTIFICATION = "문제 파악"
    EXPLORATION = "탐색"
    INTERVENTION = "개입"
    CLOSING = "마무리"


class PsychologicalIssue(Enum):
    """심리 문제 유형"""
    ANXIETY = "불안"
    DEPRESSION = "우울"
    RELATIONSHIP = "대인관계"
    STRESS = "스트레스"
    SELF_ESTEEM = "자존감"
    TRAUMA = "트라우마"
    SLEEP = "수면장애"
    WORK = "직장 스트레스"


@dataclass
class ClientState:
    """내담자 상태"""
    name: str
    age: int
    gender: str
    issues: List[PsychologicalIssue]
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'anxiety': 0.5,
        'sadness': 0.3,
        'anger': 0.2,
        'hope': 0.5,
        'trust': 0.3
    })
    energy: float = 0.7
    openness: float = 0.5  # 자기표현 성향
    memory: List[Dict] = field(default_factory=list)
    session_count: int = 0
    total_improvement: float = 0.0


@dataclass
class CounselingSession:
    """상담 세션"""
    session_id: int
    client: ClientState
    phase: CounselingPhase
    dialogue_history: List[Dict] = field(default_factory=list)
    interventions_used: List[str] = field(default_factory=list)
    improvement_score: float = 0.0
    completed: bool = False


class CounselingClinicEnvironment:
    """
    심리상담 클리닉 환경
    
    상태 공간:
    - 내담자 감정 상태
    - 상담 단계
    - 대화 이력
    
    행동 공간:
    - 공감, 질문, 반영, 해석, 지시, 정보제공 등
    """
    
    def __init__(self):
        self.clients: Dict[str, ClientState] = {}
        self.current_session: Optional[CounselingSession] = None
        self.session_history: List[CounselingSession] = []
        self.current_time: int = 0
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """내담자 데이터 초기화"""
        client_profiles = [
            {
                'name': '김민지',
                'age': 28,
                'gender': '여',
                'issues': [PsychologicalIssue.ANXIETY, PsychologicalIssue.WORK],
                'background': '대기업 근무, 업무 스트레스로 불안 증상',
                'emotional_state': {'anxiety': 0.8, 'sadness': 0.4, 'anger': 0.2, 'hope': 0.4, 'trust': 0.3}
            },
            {
                'name': '박준호',
                'age': 35,
                'gender': '남',
                'issues': [PsychologicalIssue.DEPRESSION, PsychologicalIssue.RELATIONSHIP],
                'background': '이혼 후 우울감, 대인관계 어려움',
                'emotional_state': {'anxiety': 0.3, 'sadness': 0.8, 'anger': 0.3, 'hope': 0.3, 'trust': 0.2}
            },
            {
                'name': '이서연',
                'age': 22,
                'gender': '여',
                'issues': [PsychologicalIssue.SELF_ESTEEM, PsychologicalIssue.ANXIETY],
                'background': '대학생, 자기CONFIDENCE 부족, 취업 불안',
                'emotional_state': {'anxiety': 0.7, 'sadness': 0.5, 'anger': 0.1, 'hope': 0.5, 'trust': 0.4}
            },
            {
                'name': '최현우',
                'age': 42,
                'gender': '남',
                'issues': [PsychologicalIssue.STRESS, PsychologicalIssue.SLEEP],
                'background': '자영업자, 경제적 어려움, 수면 문제',
                'emotional_state': {'anxiety': 0.6, 'sadness': 0.4, 'anger': 0.4, 'hope': 0.3, 'trust': 0.3}
            },
            {
                'name': '정수빈',
                'age': 19,
                'gender': '여',
                'issues': [PsychologicalIssue.TRAUMA, PsychologicalIssue.DEPRESSION],
                'background': '학교 폭력 경험, 트라우마',
                'emotional_state': {'anxiety': 0.6, 'sadness': 0.7, 'anger': 0.5, 'hope': 0.4, 'trust': 0.1}
            }
        ]
        
        for profile in client_profiles:
            client = ClientState(
                name=profile['name'],
                age=profile['age'],
                gender=profile['gender'],
                issues=profile['issues'],
                emotional_state=profile['emotional_state']
            )
            self.clients[client.name] = client
    
    def reset(self, client_name: str = None) -> Dict:
        """세션 초기화"""
        if client_name is None:
            client_name = random.choice(list(self.clients.keys()))
        
        client = self.clients[client_name]
        client.session_count += 1
        
        self.current_session = CounselingSession(
            session_id=client.session_count,
            client=client,
            phase=CounselingPhase.GREETING
        )
        
        self.current_time = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict:
        """현재 관찰 반환"""
        if self.current_session is None:
            return {}
        
        client = self.current_session.client
        
        return {
            'client_name': client.name,
            'client_age': client.age,
            'client_issues': [issue.value for issue in client.issues],
            'emotional_state': client.emotional_state.copy(),
            'energy': client.energy,
            'openness': client.openness,
            'phase': self.current_session.phase.value,
            'session_number': self.current_session.session_id,
            'dialogue_count': len(self.current_session.dialogue_history),
            'available_phases': [p.value for p in CounselingPhase]
        }
    
    def step(self, counselor_action: str, action_category: str) -> Tuple[Dict, float, bool, Dict]:
        """
        상담 스텝 실행
        
        Args:
            counselor_action: 상담사의 자연어 응답
            action_category: 행동 카테고리 (공감, 질문, 반영, 해석, 지시, 정보제공)
        
        Returns:
            observation, reward, done, info
        """
        self.current_time += 1
        session = self.current_session
        client = session.client
        
        # 상담사 응답 기록
        session.dialogue_history.append({
            'role': 'counselor',
            'content': counselor_action,
            'category': action_category,
            'time': self.current_time
        })
        
        # 내담자 반응 시뮬레이션
        client_response = self._simulate_client_response(client, counselor_action, action_category)
        
        session.dialogue_history.append({
            'role': 'client',
            'content': client_response['text'],
            'emotion_change': client_response['emotion_change'],
            'time': self.current_time
        })
        
        # 보상 계산
        reward = self._calculate_reward(client, action_category, client_response)
        
        # 상태 업데이트
        self._update_client_state(client, client_response)
        
        # 상담 단계 업데이트
        self._update_phase(session)
        
        # 종료 조건 체크
        done = self._check_completion(session)
        
        info = {
            'client_response': client_response,
            'phase': session.phase.value,
            'improvement': client_response.get('improvement', 0)
        }
        
        return self._get_observation(), reward, done, info
    
    def _simulate_client_response(self, client: ClientState, 
                                   counselor_action: str, 
                                   action_category: str) -> Dict:
        """내담자 반응 시뮬레이션"""
        
        response_templates = {
            "공감": [
                "네, 맞아요. 정말 그런 느낌이에요.",
                "이해해 주시니까 조금 편해지네요.",
                "그 말씀 듣고 보니 그렇게 느낄 수 있었겠네요."
            ],
            "질문": [
                "글쎄요, 잘 모르겠어요.",
                "그런 생각은 해본 적 없네요.",
                "음... 한번 생각해 볼게요.",
                "솔직히 말하면, 그때 Really 힘들었어요."
            ],
            "반영": [
                "제가 그렇게 말했었나요?",
                "᩠ 맞다, 그렇게 느꼈었어요.",
                "제 마음을 정확히 말씀해 주시네요."
            ],
            "해석": [
                "그럴 수도 있겠네요.",
                "_state of_umbrella_interest_한번 더 생각해 볼게요.",
                "그런 관계가 있었군요."
            ],
            "지시": [
                "알겠어요, 한번 해볼게요.",
                "그렇게 해보겠습니다.",
                "좋은 방법 같아요."
            ],
            "정보제공": [
                "아, 그런 거군요.",
                "몰랐는데 알게 되네요.",
                "도움이 되는 말씀이네요."
            ]
        }
        
        templates = response_templates.get(action_category, response_templates["공감"])
        response_text = random.choice(templates)
        
        # 감정 변화 계산
        emotion_change = self._calculate_emotion_change(client, action_category)
        
        # 개선도 계산
        improvement = 0.0
        if action_category in ["공감", "반영"]:
            improvement = 0.1
            client.emotional_state['trust'] = min(1.0, client.emotional_state['trust'] + 0.05)
        elif action_category == "질문":
            improvement = 0.05
            client.openness = min(1.0, client.openness + 0.03)
        
        return {
            'text': response_text,
            'emotion_change': emotion_change,
            'improvement': improvement
        }
    
    def _calculate_emotion_change(self, client: ClientState, action_category: str) -> Dict[str, float]:
        """감정 변화 계산"""
        changes = {}
        
        if action_category == "공감":
            changes['sadness'] = -0.05
            changes['trust'] = 0.05
            changes['hope'] = 0.03
        elif action_category == "질문":
            changes['anxiety'] = 0.02  # 질문은 약간의 불안 유발
            changes['hope'] = 0.02
        elif action_category == "반영":
            changes['trust'] = 0.08
            changes['sadness'] = -0.03
        elif action_category == "해석":
            changes['hope'] = 0.05
            changes['anxiety'] = -0.03
        elif action_category == "지시":
            changes['hope'] = 0.04
            changes['anxiety'] = -0.02
        elif action_category == "정보제공":
            changes['hope'] = 0.03
            changes['anxiety'] = -0.02
        
        return changes
    
    def _update_client_state(self, client: ClientState, response: Dict):
        """내담자 상태 업데이트"""
        for emotion, change in response['emotion_change'].items():
            if emotion in client.emotional_state:
                client.emotional_state[emotion] = max(0, min(1, 
                    client.emotional_state[emotion] + change))
        
        client.energy = max(0.1, min(1.0, client.energy - 0.02))
        
        client.memory.append({
            'time': self.current_time,
            'content': response['text'],
            'improvement': response.get('improvement', 0)
        })
        
        client.total_improvement += response.get('improvement', 0)
    
    def _update_phase(self, session: CounselingSession):
        """상담 단계 업데이트"""
        dialogue_count = len(session.dialogue_history)
        
        if dialogue_count <= 2:
            session.phase = CounselingPhase.GREETING
        elif dialogue_count <= 5:
            session.phase = CounselingPhase.PROBLEM_IDENTIFICATION
        elif dialogue_count <= 10:
            session.phase = CounselingPhase.EXPLORATION
        elif dialogue_count <= 15:
            session.phase = CounselingPhase.INTERVENTION
        else:
            session.phase = CounselingPhase.CLOSING
    
    def _calculate_reward(self, client: ClientState, action_category: str, 
                          response: Dict) -> float:
        """보상 계산"""
        reward = 0.0
        
        category_rewards = {
            "공감": 0.4,
            "질문": 0.3,
            "반영": 0.5,
            "해석": 0.3,
            "지시": 0.2,
            "정보제공": 0.2
        }
        
        reward += category_rewards.get(action_category, 0.1)
        
        reward += response.get('improvement', 0)
        
        if client.emotional_state['trust'] > 0.5:
            reward += 0.1
        
        if client.emotional_state['hope'] > 0.5:
            reward += 0.1
        
        return reward
    
    def _check_completion(self, session: CounselingSession) -> bool:
        """세션 완료 체크"""
        if len(session.dialogue_history) >= 20:
            return True
        
        if session.phase == CounselingPhase.CLOSING and len(session.dialogue_history) >= 12:
            return True
        
        return False
    
    def get_session_summary(self) -> Dict:
        """세션 요약"""
        if self.current_session is None:
            return {}
        
        session = self.current_session
        client = session.client
        
        initial_state = session.dialogue_history[0] if session.dialogue_history else None
        
        return {
            'session_id': session.session_id,
            'client': client.name,
            'issues': [issue.value for issue in client.issues],
            'total_dialogues': len(session.dialogue_history),
            'final_phase': session.phase.value,
            'improvement': client.total_improvement,
            'final_emotions': client.emotional_state.copy(),
            'counselor_styles': session.interventions_used
        }
    
    def get_all_clients_info(self) -> List[Dict]:
        """모든 내담자 정보"""
        info = []
        for name, client in self.clients.items():
            info.append({
                'name': client.name,
                'age': client.age,
                'gender': client.gender,
                'issues': [issue.value for issue in client.issues],
                'sessions': client.session_count,
                'total_improvement': client.total_improvement
            })
        return info
