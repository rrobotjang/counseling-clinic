"""
강화학습 에이전트 - 자연어로 행동하고 학습하는 에이전트

학습 가능 요소:
1. 어떤 말을 할지 (정책)
2. 누구와 대화할지 (선택)
3. 언제 성찰할지 (타이밍)

자연어 액션:
- 인사, 질문, 공감, 제안, 관찰 등
- 템플릿 + 자유 형식
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random
import json


# ============================================================
# 자연어 액션 정의
# ============================================================

@dataclass
class ActionTemplate:
    """자연어 액션 템플릿"""
    category: str      # 인사, 질문, 공감, 제안, 관찰, 기타
    template: str      # 템플릿 문장
    weight: float = 1.0  # 선택 가중치
    
    def to_string(self, **kwargs) -> str:
        """템플릿을 실제 문장으로 변환"""
        try:
            return self.template.format(**kwargs)
        except KeyError:
            return self.template


# 기본 액션 템플릿들
DEFAULT_ACTIONS = [
    # 인사
    ActionTemplate("greeting", "안녕!"),
    ActionTemplate("greeting", "반가워!"),
    ActionTemplate("greeting", "좋은 하루야!"),
    
    # 질문
    ActionTemplate("question", "어떻게 지냈어?"),
    ActionTemplate("question", "무슨 일 있어?"),
    ActionTemplate("question", "기분이 어때?"),
    
    # 공감
    ActionTemplate("empathy", "그렇구나."),
    ActionTemplate("empathy", "이해해."),
    ActionTemplate("empathy", "그럴 수 있지."),
    
    # 제안
    ActionTemplate("suggestion", "같이 산책할까?"),
    ActionTemplate("suggestion", "봉사활동 할래?"),
    ActionTemplate("suggestion", "이야기 나눌까?"),
    
    # 관찰
    ActionTemplate("observation", "오늘 날씨가 좋네."),
    ActionTemplate("observation", "예쁜 꽃이 폈어."),
    ActionTemplate("observation", "노래가 듣고 싶어."),
    
    # 감정 표현
    ActionTemplate("emotion", "기분이 좋아!"),
    ActionTemplate("emotion", "행복해!"),
    ActionTemplate("emotion", "신나!"),
    
    # 기타
    ActionTemplate("other", "그래."),
    ActionTemplate("other", "응."),
    ActionTemplate("other", "그건가."),
]

# 액션 카테고리별 보상 기본값
CATEGORY_REWARDS = {
    "greeting": 0.3,
    "question": 0.4,
    "empathy": 0.5,
    "suggestion": 0.4,
    "observation": 0.2,
    "emotion": 0.3,
    "other": 0.1
}


# ============================================================
# 정책 네트워크
# ============================================================

class PolicyNetwork(nn.Module):
    """
    정책 네트워크
    
    입력: 에이전트 상태 (감정, 에너지, 관계 등)
    출력: 액션 확률 분포
    
    사용법:
    - Forward pass로 액션 확률 계산
    - 확률 분포에서 샘플링으로 행동 선택
    - 보상 기반으로 가중치 업데이트
    """
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.log_probs = []
        self.values = []
        self.rewards = []
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """상태를 액션 확률과 상태 가치로 변환"""
        logits = self.network(state)
        values = self.value_head(state)
        
        return logits, values
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        상태에서 행동 선택
        
        Returns:
            action_idx: 선택된 액션 인덱스
            log_prob: 로그 확률 (학습용)
            value: 상태 가치 추정
        """
        logits, value = self.forward(state)
        
        # 확률 분포 생성
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # 샘플링
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value
    
    def update(self, gamma: float = 0.99):
        """
        정책 gradient 업데이트 (REINFORCE)
        """
        if len(self.rewards) == 0:
            return
        
        # 보상 계산
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 정책 손실 계산
        policy_loss = []
        value_loss = []
        
        for log_prob, value, R in zip(self.log_probs, self.values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.mse_loss(value.squeeze(), R.detach().clone()))
        
        # 역전파
        loss = torch.stack(policy_loss).sum() + 0.5 * torch.stack(value_loss).sum()
        
        # 메모리 초기화
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return loss
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))


# ============================================================
# 강화학습 에이전트
# ============================================================

class RLAgent:
    """
    강화학습이 가능한 에이전트
    
    능력:
    1. 자연어로 행동 (액션 생성)
    2. 환경과 상호작용
    3. 보상을 통해 학습
    4. 정책 업데이트
    5. 기억 축적 및 활용
    """
    
    def __init__(self, 
                 name: str,
                 action_templates: List[ActionTemplate] = None,
                 state_dim: int = 10,
                 learning_rate: float = 0.001):
        
        self.name = name
        
        # 액션 정의
        self.action_templates = action_templates or DEFAULT_ACTIONS
        self.num_actions = len(self.action_templates)
        
        # 상태 차원
        self.state_dim = state_dim
        
        # 정책 네트워크
        self.policy = PolicyNetwork(state_dim, self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 경험 메모리
        self.experience_buffer: List[Dict] = []
        
        # 누적 보상
        self.total_reward: float = 0.0
        self.episode_rewards: List[float] = []
        
        # 현재 상태
        self.current_state: Optional[torch.Tensor] = None
        self.last_action_idx: Optional[int] = None
        
        # 학습 통계
        self.episodes_done: int = 0
        self.avg_reward_history: List[float] = []
    
    def get_state_vector(self, 
                         emotion: str = "neutral",
                         energy: float = 1.0,
                         relationships: int = 0,
                         recent_conversations: int = 0,
                         time_of_day: float = 0.5) -> torch.Tensor:
        """
        현재 상태를 벡터로 변환
        
        요소:
        1. 감정 원-핫 (8 차원)
        2. 에너지 (1 차원)
        3. 관계 수 (정규화, 1 차원)
        """
        # 감정 원-핫
        emotions = ["neutral", "happy", "sad", "angry", "excited", "calm", "curious", "anxious"]
        emotion_vec = [1.0 if e == emotion else 0.0 for e in emotions]
        
        # 상태 벡터
        state = emotion_vec + [energy, min(relationships / 10, 1.0)]
        
        return torch.tensor(state, dtype=torch.float32)
    
    def select_action(self, state: torch.Tensor) -> Tuple[str, int]:
        """
        현재 상태에서 행동 선택
        
        Returns:
            action_text: 생성된 자연어 행동
            action_idx: 액션 인덱스
        """
        action_idx, log_prob, value = self.policy.get_action(state)
        
        # 상태와 행동 저장
        self.current_state = state
        self.last_action_idx = action_idx
        self.policy.log_probs.append(log_prob)
        self.policy.values.append(value)
        
        # 액션을 자연어로 변환
        action_template = self.action_templates[action_idx]
        action_text = action_template.to_string(name=self.name)
        
        return action_text, action_idx
    
    def receive_reward(self, reward: float, next_state: torch.Tensor = None):
        """
        보상 수신 및 학습 데이터 저장
        
        보상 소스:
        1. 상대방의 긍정적 반응
        2. 관계 개선
        3. 목표 달성
        4. 환경 적응
        """
        self.policy.rewards.append(reward)
        self.total_reward += reward
        
        # 경험 버퍼에 저장
        if self.current_state is not None:
            experience = {
                'state': self.current_state,
                'action': self.last_action_idx,
                'reward': reward,
                'next_state': next_state
            }
            self.experience_buffer.append(experience)
    
    def learn(self, gamma: float = 0.99):
        """
        경험으로부터 학습
        
        학습 알고리즘:
        1. REINFORCE (기본 정책 그래디언트)
        2. 선택적: 경험 리플레이
        """
        loss = self.policy.update(gamma)
        
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.episodes_done += 1
    
    def get_action_probabilities(self, state: torch.Tensor) -> Dict[str, float]:
        """
        상태에서 각 액션의 확률 반환
        
        해석 가능성을 위해 사용
        """
        with torch.no_grad():
            logits, _ = self.policy(state)
            probs = F.softmax(logits, dim=-1)
        
        action_probs = {}
        for i, template in enumerate(self.action_templates):
            action_probs[template.template] = probs[i].item()
        
        return action_probs
    
    def plan_action(self, context: Dict) -> str:
        """
        컨텍스트 기반으로 다음 행동 계획
        
        계획 과정:
        1. 상태 분석
        2. 관련 기억 회상
        3. 최적 행동 선택
        """
        # 상태 벡터 생성
        state = self.get_state_vector(
            emotion=context.get('emotion', 'neutral'),
            energy=context.get('energy', 1.0),
            relationships=context.get('relationships', 0)
        )
        
        # 행동 선택
        action_text, _ = self.select_action(state)
        
        return action_text
    
    def reflect_and_learn(self):
        """
        성찰 및 학습
        
        주기적으로 호출하여:
        1. 최근 경험 분석
        2. 패턴 발견
        3. 정책 업데이트
        """
        if len(self.experience_buffer) < 10:
            return
        
        # 학습
        self.learn()
        
        # 평균 보상 계산
        recent_rewards = [e['reward'] for e in self.experience_buffer[-10:]]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        self.avg_reward_history.append(avg_reward)
    
    def get_statistics(self) -> Dict:
        """학습 통계"""
        return {
            'name': self.name,
            'episodes': self.episodes_done,
            'total_reward': self.total_reward,
            'avg_reward': np.mean(self.avg_reward_history[-10:]) if self.avg_reward_history else 0,
            'buffer_size': len(self.experience_buffer)
        }
    
    def save(self, filepath: str):
        """에이전트 상태 저장"""
        state = {
            'name': self.name,
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'episodes_done': self.episodes_done,
            'total_reward': self.total_reward,
            'avg_reward_history': self.avg_reward_history,
            'policy_state': self.policy.state_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
        
        # 가중치 별도 저장
        self.policy.save(filepath.replace('.json', '_weights.pth'))
    
    def load(self, filepath: str):
        """에이전트 상태 로드"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.name = state['name']
        self.episodes_done = state['episodes_done']
        self.total_reward = state['total_reward']
        self.avg_reward_history = state['avg_reward_history']
        
        # 가중치 로드
        self.policy.load(filepath.replace('.json', '_weights.pth'))


# ============================================================
# 여러 에이전트 관리
# ============================================================

class MultiAgentSystem:
    """
    여러 강화학습 에이전트를 관리하는 시스템
    """
    
    def __init__(self):
        self.agents: Dict[str, RLAgent] = {}
    
    def add_agent(self, agent: RLAgent):
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[RLAgent]:
        return self.agents.get(name)
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """모든 에이전트 통계"""
        stats = {}
        for name, agent in self.agents.items():
            stats[name] = agent.get_statistics()
        return stats
    
    def save_all(self, directory: str):
        """모든 에이전트 저장"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, agent in self.agents.items():
            filepath = os.path.join(directory, f"{name}_agent.json")
            agent.save(filepath)
    
    def load_all(self, directory: str):
        """모든 에이전트 로드"""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith('_agent.json'):
                filepath = os.path.join(directory, filename)
                agent_name = filename.replace('_agent.json', '')
                agent = RLAgent(name=agent_name)
                agent.load(filepath)
                self.agents[agent_name] = agent
