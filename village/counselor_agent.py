"""
상담사 에이전트 - RL + 언어모델 통합

강화학습으로 상담 전략을 학습하고,
학습된 한국어 모델로 실제 응답을 생성합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from village.counseling_data import CounselingResponseSystem, CounselingSession


@dataclass
class CounselingAction:
    """상담 행동"""
    category: str
    description: str
    templates: List[str]


COUNSELING_ACTIONS = [
    CounselingAction(
        "공감",
        "내담자의 감정을 공감적으로 수용",
        [
            "그感じ 참 힘들었겠어요.",
            "많이 힘드셨군요.",
            "정말 어렵고 힘든 상황이었네요.",
            "그렇게 느끼는 것이 당연한 것 같아요.",
            "많이 속상하셨겠어요."
        ]
    ),
    CounselingAction(
        "질문",
        "개방형 질문으로 탐색",
        [
            "그때 어떤 생각이 드셨나요?",
            "어떻게 느끼셨어요?",
            "좀 더 자세히 말씀해 주실 수 있을까요?",
            "그 상황에서 가장 힘들었던 부분은 무엇이었나요?",
            "지금 가장 하고 싶은 말씀이 있으신가요?"
        ]
    ),
    CounselingAction(
        "반영",
        "내담자의 말을 되돌려줌",
        [
            "지금 말씀하신 것은 정말 중요하다고 느끼시는군요.",
            "그렇게 말씀하시니 저도 그 마음이 느껴집니다.",
            "그 말씀은 정말 의미가 있는 것 같아요.",
            "혹시 그 말씀의 의미를 좀 더 설명해 주실 수 있을까요?",
            "네, 그 말씀이 정말 와닿네요."
        ]
    ),
    CounselingAction(
        "해석",
        "내담자의 경험을 새롭게 해석",
        [
            "그렇다면 혹시 그것이 ваш_문제와 관련이 있을 수도 있겠네요.",
            "그 말씀을 듣고 보니, votre_다른 면도 있을 것 같아요.",
            "그렇게 생각하면, vous의_상황이 좀 더 이해가 되네요.",
            "그 말씀은 정말vous的重要 insight인 것 같아요.",
            "그렇다면, vous_다른 가능성도 있을 것 같아요."
        ]
    ),
    CounselingAction(
        "지시",
        "행동이나 변화를 위한 지시",
        [
            "오늘은 просто 쉬어가시는 것도 좋을 것 같아요.",
            "가벼운 산책을 해보시는 건 어떨까요?",
            "일기를 써보시는 것도 도움이 될 수 있어요.",
            "몸을 편안히 쉬는 연습을 해보시는 건 어떨까요?",
            "작은 목표를 하나 정해보시는 건 어떨까요?"
        ]
    ),
    CounselingAction(
        "정보제공",
        "유용한 정보나 관점 제공",
        [
            "그런 경우에는 그럴 수도 있다는 말도 있더라고요.",
            "정말 diverse_경험이 많으시네요.",
            "사실 그런 경우에는 diverse_사람도 많다고 하더라고요.",
            "그렇게 생각하시는 분도 많으시더라고요.",
            "그 말씀 듣고 보니, diverse_정말 중요하다는 생각이 드네요."
        ]
    )
]


class PolicyNetwork(nn.Module):
    """상담 전략 정책 네트워크"""
    
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
        logits = self.network(state)
        values = self.value_head(state)
        return logits, values
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
    
    def update(self, gamma: float = 0.99):
        if len(self.rewards) == 0:
            return None
        
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        value_loss = []
        
        for log_prob, value, R in zip(self.log_probs, self.values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.mse_loss(value.squeeze(), R.detach().clone()))
        
        loss = torch.stack(policy_loss).sum() + 0.5 * torch.stack(value_loss).sum()
        
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return loss


class CounselorAgent:
    """
    상담사 에이전트
    
    통합 구조:
    1. RL 정책: 어떤 상담 전략(카테고리)을 사용할지 결정
    2. 언어 모델: 선택된 전략에 따른 실제 응답 생성
    
    학습 가능 요소:
    - 어떤 전략을 언제 사용할지 (정책)
    - 어떤 템플릿을 선택할지 (미세 조정)
    - 보상 기반으로 전략 개선
    """
    
    def __init__(self, 
                 name: str = "상담사",
                 state_dim: int = 15,
                 learning_rate: float = 0.001,
                 counseling_data_path: str = None):
        
        self.name = name
        self.state_dim = state_dim
        self.num_actions = len(COUNSELING_ACTIONS)
        
        self.policy = PolicyNetwork(state_dim, self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.counseling_actions = COUNSELING_ACTIONS
        
        self.response_system = CounselingResponseSystem(counseling_data_path)
        
        self.experience_buffer: List[Dict] = []
        self.total_reward: float = 0.0
        self.episodes_done: int = 0
        
        self.current_state: Optional[torch.Tensor] = None
        self.last_action_idx: Optional[int] = None
    
    def get_state_vector(self, 
                        emotional_state: Dict[str, float],
                        energy: float,
                        openness: float,
                        phase: str,
                        dialogue_count: int) -> torch.Tensor:
        """현재 상태를 벡터로 변환"""
        phase_encoding = {
            "인사": [1, 0, 0, 0, 0],
            "문제 파악": [0, 1, 0, 0, 0],
            "탐색": [0, 0, 1, 0, 0],
            "개입": [0, 0, 0, 1, 0],
            "마무리": [0, 0, 0, 0, 1]
        }
        
        phase_vec = phase_encoding.get(phase, [0, 0, 0, 0, 0])
        
        emotion_values = list(emotional_state.values())
        
        state = emotion_values + [energy, openness] + phase_vec + [min(dialogue_count / 20, 1.0)]
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        if state_tensor.shape[0] < self.state_dim:
            padding = torch.zeros(self.state_dim - state_tensor.shape[0])
            state_tensor = torch.cat([state_tensor, padding])
        elif state_tensor.shape[0] > self.state_dim:
            state_tensor = state_tensor[:self.state_dim]
        
        return state_tensor
    
    def select_action(self, state: torch.Tensor) -> Tuple[str, str, int]:
        """상담 행동 선택"""
        action_idx, log_prob, value = self.policy.get_action(state)
        
        self.current_state = state
        self.last_action_idx = action_idx
        self.policy.log_probs.append(log_prob)
        self.policy.values.append(value)
        
        action = self.counseling_actions[action_idx]
        
        response = random.choice(action.templates)
        
        return response, action.category, action_idx
    
    def receive_reward(self, reward: float):
        """보상 수신"""
        self.policy.rewards.append(reward)
        self.total_reward += reward
    
    def learn(self, gamma: float = 0.99):
        """학습"""
        loss = self.policy.update(gamma)
        
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.episodes_done += 1
    
    def get_action_probabilities(self, state: torch.Tensor) -> Dict[str, float]:
        """각 행동의 확률 반환"""
        with torch.no_grad():
            logits, _ = self.policy(state)
            probs = F.softmax(logits, dim=-1)
        
        action_probs = {}
        for i, action in enumerate(self.counseling_actions):
            action_probs[action.category] = probs[i].item()
        
        return action_probs
    
    def plan_response(self, client_info: Dict) -> Tuple[str, str]:
        """응답 계획"""
        state = self.get_state_vector(
            emotional_state=client_info.get('emotional_state', {}),
            energy=client_info.get('energy', 0.7),
            openness=client_info.get('openness', 0.5),
            phase=client_info.get('phase', '탐색'),
            dialogue_count=client_info.get('dialogue_count', 0)
        )
        
        action_idx, log_prob, value = self.policy.get_action(state)
        
        self.current_state = state
        self.last_action_idx = action_idx
        self.policy.log_probs.append(log_prob)
        self.policy.values.append(value)
        
        action = self.counseling_actions[action_idx]
        category = action.category
        
        last_user_msg = client_info.get('last_user_message', '')
        
        if last_user_msg:
            response = self.response_system.generate_response(last_user_msg, category)
        else:
            response = random.choice(action.templates)
        
        return response, category
    
    def save(self, filepath: str):
        """모델 저장"""
        torch.save({
            'policy_state': self.policy.state_dict(),
            'episodes_done': self.episodes_done,
            'total_reward': self.total_reward
        }, filepath)
    
    def load(self, filepath: str):
        """모델 로드"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location='cpu')
            self.policy.load_state_dict(checkpoint['policy_state'])
            self.episodes_done = checkpoint['episodes_done']
            self.total_reward = checkpoint['total_reward']
    
    def get_statistics(self) -> Dict:
        """통계"""
        return {
            'name': self.name,
            'episodes': self.episodes_done,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.episodes_done)
        }
