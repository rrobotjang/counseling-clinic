"""
강화학습 환경 - 빌리지 샌드박스

에이전트가 상호작용하는 환경:
1. 상태 관찰
2. 행동 실행
3. 보상 계산
4. 다음 상태 생성
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class Observation:
    """환경 관찰"""
    speaker: str
    listener: str
    message: str
    context: Dict
    relationships: Dict[str, float]
    

class VillageEnvironment:
    """
    빌리지 환경
    
    강화학습 에이전트가 학습할 수 있는 샌드박스
    
    상태 공간:
    - 감정 상태
    - 에너지 레벨
    - 관계 정보
    - 최근 대화
    
    행동 공간:
    - 자연어 템플릿 (21개)
    
    보상 함수:
    - 대화 성공도
    - 관계 개선
    - 다양한 상호작용
    """
    
    def __init__(self, agent_names: List[str] = None, num_agents: int = 5):
        if agent_names:
            self.agent_names = agent_names
            self.num_agents = len(agent_names)
        else:
            self.agent_names = [f"agent_{i}" for i in range(num_agents)]
            self.num_agents = num_agents
        
        self.agent_states = {
            name: {
                'emotion': 'neutral',
                'energy': 1.0,
                'relationships': {},
                'conversation_count': 0,
                'satisfaction': 0.5
            }
            for name in self.agent_names
        }
        
        self.history: List[Dict] = []
        self.current_time: float = 0.0
    
    def reset(self) -> Dict:
        """환경 초기화"""
        for agent_id in self.agent_states:
            self.agent_states[agent_id] = {
                'emotion': 'neutral',
                'energy': 1.0,
                'relationships': {},
                'conversation_count': 0,
                'satisfaction': 0.5
            }
        
        self.history = []
        self.current_time = 0.0
        
        return self.get_initial_observation()
    
    def get_initial_observation(self) -> Dict:
        """초기 관찰"""
        return {
            'time': 0,
            'available_agents': list(self.agent_states.keys()),
            'agent_states': self.agent_states.copy()
        }
    
    def step(self, agent_id: str, action_text: str, 
             target_id: str = None) -> Tuple[Dict, float, bool, Dict]:
        """
        환경 스텝
        
        Args:
            agent_id: 행동하는 에이전트
            action_text: 자연어 행동
            target_id: 상대방 (선택적)
            
        Returns:
            observation: 다음 관찰
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        self.current_time += 1
        
        # 대상 선택 (없으면 랜덤)
        if target_id is None:
            available = [a for a in self.agent_states.keys() if a != agent_id]
            target_id = random.choice(available) if available else agent_id
        
        # 보상 계산
        reward = self._calculate_reward(agent_id, action_text, target_id)
        
        # 상태 업데이트
        self._update_states(agent_id, target_id, action_text, reward)
        
        # 이력 저장
        self.history.append({
            'time': self.current_time,
            'agent': agent_id,
            'target': target_id,
            'action': action_text,
            'reward': reward
        })
        
        # 다음 관찰
        observation = {
            'time': self.current_time,
            'last_action': action_text,
            'last_reward': reward,
            'agent_states': self.agent_states.copy(),
            'available_agents': list(self.agent_states.keys())
        }
        
        # 종료 조건 (100 스텝 후)
        done = self.current_time >= 100
        
        info = {
            'target': target_id,
            'conversation_happened': agent_id != target_id
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, agent_id: str, action_text: str, 
                          target_id: str) -> float:
        """
        보상 계산
        
        보상 요소:
        1. 기본 보상 (액션 유형)
        2. 관계 보상 (친밀도, 신뢰)
        3. 다양성 보상 (새로운 상호작용)
        4. 감정 보상 (긍정적 감정 유도)
        """
        reward = 0.0
        
        # 1. 액션 유형별 기본 보상
        action_rewards = {
            "greeting": 0.3,
            "question": 0.4,
            "empathy": 0.5,
            "suggestion": 0.4,
            "observation": 0.2,
            "emotion": 0.3,
            "other": 0.1
        }
        
        # 액션 카테고리 추정 (간단한 키워드 매칭)
        for category, base_reward in action_rewards.items():
            if any(kw in action_text for kw in self._get_category_keywords(category)):
                reward += base_reward
                break
        
        # 2. 관계 보상
        if target_id in self.agent_states[agent_id]['relationships']:
            familiarity = self.agent_states[agent_id]['relationships'].get(target_id, 0)
            reward += familiarity * 0.2  # 친한 사람과의 대화 보너스
        
        # 3. 다양성 보상 (지난 대화와 다른 행동)
        recent_actions = [h['action'] for h in self.history[-5:]]
        if action_text not in recent_actions:
            reward += 0.2  # 새로운 행동 보너스
        
        # 4. 감정 보상
        current_emotion = self.agent_states[agent_id]['emotion']
        if current_emotion in ['happy', 'excited', 'calm']:
            reward += 0.1  # 긍정적 감정 보너스
        
        return reward
    
    def _get_category_keywords(self, category: str) -> List[str]:
        """카테고리별 키워드"""
        keywords = {
            "greeting": ["안녕", "반가워", "좋은"],
            "question": ["어때", "무슨", "어떻게"],
            "empathy": ["이해", "그렇구나", "할 수 있지"],
            "suggestion": ["같이", "할래", "하러"],
            "observation": ["오늘", "예쁜", "좋네"],
            "emotion": ["기분", "행복", "신나"],
            "other": ["그래", "응", "그건"]
        }
        return keywords.get(category, [])
    
    def _update_states(self, agent_id: str, target_id: str, 
                       action_text: str, reward: float):
        """에이전트 상태 업데이트"""
        # 에너지 소모
        self.agent_states[agent_id]['energy'] = max(
            0.1, 
            self.agent_states[agent_id]['energy'] - 0.05
        )
        
        # 대화 횟수 증가
        self.agent_states[agent_id]['conversation_count'] += 1
        
        # 관계 업데이트
        if agent_id != target_id:
            # 관계가 없으면 생성
            if target_id not in self.agent_states[agent_id]['relationships']:
                self.agent_states[agent_id]['relationships'][target_id] = 0.0
            
            # 상호작용으로 친밀도 증가
            self.agent_states[agent_id]['relationships'][target_id] = min(
                1.0,
                self.agent_states[agent_id]['relationships'][target_id] + 0.05
            )
        
        # 만족도 업데이트
        self.agent_states[agent_id]['satisfaction'] = min(
            1.0,
            self.agent_states[agent_id]['satisfaction'] + reward * 0.1
        )
        
        # 감정 업데이트 (간단한 규칙)
        if reward > 0.5:
            self.agent_states[agent_id]['emotion'] = 'happy'
        elif reward > 0.3:
            self.agent_states[agent_id]['emotion'] = 'calm'
        elif reward < 0.1:
            self.agent_states[agent_id]['emotion'] = 'neutral'
    
    def get_agent_state(self, agent_id: str) -> Dict:
        """특정 에이전트 상태"""
        return self.agent_states.get(agent_id, {})
    
    def get_relationship_matrix(self) -> np.ndarray:
        """관계 행렬 반환"""
        agent_ids = sorted(self.agent_states.keys())
        n = len(agent_ids)
        
        matrix = np.zeros((n, n))
        
        for i, agent_id in enumerate(agent_ids):
            for j, target_id in enumerate(agent_ids):
                if agent_id != target_id:
                    rel = self.agent_states[agent_id]['relationships'].get(target_id, 0)
                    matrix[i, j] = rel
        
        return matrix
    
    def get_statistics(self) -> Dict:
        """환경 통계"""
        total_conversations = sum(
            s['conversation_count'] for s in self.agent_states.values()
        )
        
        avg_satisfaction = np.mean(
            [s['satisfaction'] for s in self.agent_states.values()]
        )
        
        return {
            'time': self.current_time,
            'total_conversations': total_conversations,
            'avg_satisfaction': avg_satisfaction,
            'num_agents': len(self.agent_states)
        }


# ============================================================
# 보상 함수들
# ============================================================

class RewardFunction:
    """
    다양한 보상 함수
    
    사용자가 선택할 수 있는 보상 전략
    """
    
    @staticmethod
    def simple_reward(action_category: str) -> float:
        """단순 카테고리별 보상"""
        rewards = {
            "greeting": 0.3,
            "question": 0.4,
            "empathy": 0.5,
            "suggestion": 0.4,
            "observation": 0.2,
            "emotion": 0.3,
            "other": 0.1
        }
        return rewards.get(action_category, 0.1)
    
    @staticmethod
    def relationship_reward(familiarity: float) -> float:
        """관계 기반 보상"""
        return familiarity * 0.5
    
    @staticmethod
    def novelty_reward(action: str, history: List[str]) -> float:
        """새로움 보상"""
        if action not in history:
            return 0.3
        return 0.0
    
    @staticmethod
    def emotion_reward(emotion: str) -> float:
        """감정 보상"""
        rewards = {
            'happy': 0.4,
            'excited': 0.3,
            'calm': 0.2,
            'neutral': 0.1,
            'sad': -0.1,
            'angry': -0.2
        }
        return rewards.get(emotion, 0.0)
    
    @staticmethod
    def composite_reward(action_category: str, familiarity: float,
                         emotion: str, history: List[str]) -> float:
        """복합 보상"""
        r1 = RewardFunction.simple_reward(action_category)
        r2 = RewardFunction.relationship_reward(familiarity)
        r3 = RewardFunction.novelty_reward(action_category, history)
        r4 = RewardFunction.emotion_reward(emotion)
        
        return r1 + r2 + r3 + r4
