"""
학습 트레이너 - 에이전트 학습 관리

에피소드 기반 학습:
1. 환경 초기화
2. 에이전트 행동
3. 보상 수신
4. 정책 업데이트
5. 반복
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
import json
import os
from datetime import datetime

from village.rl_agent import RLAgent, MultiAgentSystem
from village.environment import VillageEnvironment


class Trainer:
    """
    강화학습 트레이너
    
    학습 과정:
    1. 에이전트들을 환경에 배치
    2. 에피소드 실행
    3. 경험 축적
    4. 정책 업데이트
    5. 성능 평가
    """
    
    def __init__(self, 
                 environment: VillageEnvironment,
                 agents: MultiAgentSystem):
        self.env = environment
        self.agents = agents
        
        self.training_history: List[Dict] = []
        self.episode_rewards: List[Dict] = []
        
        self.on_episode_end: Optional[Callable] = None
        self.on_agent_update: Optional[Callable] = None
    
    def train_episode(self, max_steps: int = 50) -> Dict:
        """
        하나의 에피소드 학습
        
        Returns:
            에피소드 결과
        """
        # 환경 초기화
        obs = self.env.reset()
        
        episode_reward = {name: 0.0 for name in self.agents.agents}
        episode_steps = 0
        
        for step in range(max_steps):
            # 랜덤으로 에이전트 선택
            agent_names = list(self.agents.agents.keys())
            current_agent_name = np.random.choice(agent_names)
            current_agent = self.agents.agents[current_agent_name]
            
            # 에이전트 상태 벡터 생성
            agent_state = self.env.get_agent_state(current_agent_name)
            state_vector = current_agent.get_state_vector(
                emotion=agent_state.get('emotion', 'neutral'),
                energy=agent_state.get('energy', 1.0),
                relationships=len(agent_state.get('relationships', {}))
            )
            
            action_text, action_idx = current_agent.select_action(state_vector)
            
            next_obs, reward, done, info = self.env.step(
                current_agent_name, 
                action_text
            )
            
            # 보상 수신
            current_agent.receive_reward(reward)
            episode_reward[current_agent_name] += reward
            
            episode_steps += 1
            
            if done:
                break
        
        # 에이전트 학습
        for agent in self.agents.agents.values():
            agent.reflect_and_learn()
        
        # 결과
        episode_result = {
            'episode': len(self.episode_rewards) + 1,
            'steps': episode_steps,
            'rewards': episode_reward,
            'env_stats': self.env.get_statistics(),
            'agent_stats': self.agents.get_all_statistics()
        }
        
        self.episode_rewards.append(episode_result)
        
        # 콜백
        if self.on_episode_end:
            self.on_episode_end(episode_result)
        
        return episode_result
    
    def train(self, num_episodes: int = 100, 
              steps_per_episode: int = 50,
              verbose: bool = True) -> List[Dict]:
        """
        여러 에피소드 학습
        
        Args:
            num_episodes: 학습 에피소드 수
            steps_per_episode: 에피소드당 스텝 수
            verbose: 출력 여부
            
        Returns:
            학습 이력
        """
        results = []
        
        for episode in range(num_episodes):
            result = self.train_episode(steps_per_episode)
            results.append(result)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean([r['rewards'].get('agent_0', 0) for r in results[-10:]])
                print(f"에피소드 {episode+1}/{num_episodes}")
                print(f"  평균 보상: {avg_reward:.3f}")
                print(f"  총 대화: {result['env_stats']['total_conversations']}")
        
        self.training_history = results
        return results
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        학습된 에이전트 평가
        
        학습 없이 성능 측정
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            
            for _ in range(50):
                agent_names = list(self.agents.agents.keys())
                agent_name = np.random.choice(agent_names)
                agent = self.agents.agents[agent_name]
                
                agent_state = self.env.get_agent_state(agent_name)
                state = agent.get_state_vector(
                    emotion=agent_state.get('emotion', 'neutral'),
                    energy=agent_state.get('energy', 1.0),
                    relationships=len(agent_state.get('relationships', {}))
                )
                
                action_text, _ = agent.select_action(state)
                next_obs, reward, done, info = self.env.step(agent_name, action_text)
                episode_reward += reward
                
                agent.receive_reward(reward)
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }
    
    def get_training_summary(self) -> str:
        """학습 요약"""
        if not self.episode_rewards:
            return "학습된 에피소드가 없습니다."
        
        recent = self.episode_rewards[-10:]
        
        avg_rewards = [r['rewards'].get('agent_0', 0) for r in recent]
        
        summary = [
            "=== 학습 요약 ===",
            f"총 에피소드: {len(self.episode_rewards)}",
            f"최근 10 에피소드 평균 보상: {np.mean(avg_rewards):.3f}",
            f"최근 10 에피소드 최대 보상: {np.max(avg_rewards):.3f}",
            "",
            "--- 에이전트별 통계 ---"
        ]
        
        for name, stats in self.agents.get_all_statistics().items():
            summary.append(
                f"{name}: "
                f"에피소드 {stats['episodes']}, "
                f"총 보상 {stats['total_reward']:.2f}"
            )
        
        return "\n".join(summary)
    
    def save_checkpoint(self, filepath: str):
        """체크포인트 저장"""
        checkpoint = {
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'timestamp': datetime.now().isoformat()
        }
        
        # 학습 기록 저장
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # 에이전트 가중치 저장
        agent_dir = filepath.replace('.json', '_agents')
        self.agents.save_all(agent_dir)
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.training_history = checkpoint['training_history']
        self.episode_rewards = checkpoint['episode_rewards']
        
        # 에이전트 가중치 로드
        agent_dir = filepath.replace('.json', '_agents')
        if os.path.exists(agent_dir):
            self.agents.load_all(agent_dir)


# ============================================================
# 학습 시각화
# ============================================================

def plot_training_results(results: List[Dict]):
    """학습 결과 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        episodes = [r['episode'] for r in results]
        rewards = [np.mean(list(r['rewards'].values())) for r in results]
        
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards)
        plt.xlabel('에피소드')
        plt.ylabel('평균 보상')
        plt.title('강화학습 학습 곡선')
        plt.grid(True)
        plt.savefig('training_curve.png')
        plt.close()
        
        return "학습 곡선이 training_curve.png에 저장되었습니다."
    except ImportError:
        return "matplotlib이 설치되어 있지 않습니다."
