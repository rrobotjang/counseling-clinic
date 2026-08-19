"""
빌리지 시뮬레이션 - 에이전트들의 마을

시간의 흐름에 따라:
1. 에이전트들이 상호작용
2. 기억이 축적됨
3. 관계가 형성됨
4. 집단적 지식이 생성됨
5. 에이전트들이 성장함
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import random
import json
from datetime import datetime

from village.agent import Agent, Plan, Personality, Emotion
from village.memory import MemoryEvent


class EventType(Enum):
    """이벤트 유형"""
    CONVERSATION = "conversation"
    OBSERVATION = "observation"
    GROUP_MEETING = "group_meeting"
    RANDOM_EVENT = "random_event"


@dataclass
class VillageEvent:
    """빌리지 이벤트"""
    timestamp: float
    event_type: EventType
    description: str
    participants: List[str]
    location: str = "마을 광장"
    importance: float = 0.5


class Village:
    """
    에이전트들의 마을 시뮬레이션
    
    특징:
    1. 시간 기반 시뮬레이션
    2. 자동 상호작용 생성
    3. 이벤트 시스템
    4. 글로벌 상태 관리
    5. 시각화 데이터 제공
    """
    
    def __init__(self, name: str = "행복 마을"):
        self.name = name
        self.agents: Dict[str, Agent] = {}
        self.current_time: float = 0.0
        self.time_step: float = 1.0  # 기본 시간 단위
        
        # 이력
        self.events: List[VillageEvent] = []
        self.conversation_log: List[Dict] = []
        
        # 콜백
        self.on_event: Optional[Callable] = None
        
        # 통계
        self.total_conversations: int = 0
        self.total_events: int = 0
    
    def add_agent(self, agent: Agent):
        """에이전트 추가"""
        self.agents[agent.name] = agent
        self._log_event(
            EventType.OBSERVATION,
            f"{agent.name}이(가) 마을에 도착했습니다.",
            [agent.name]
        )
    
    def remove_agent(self, agent_name: str):
        """에이전트 제거"""
        if agent_name in self.agents:
            del self.agents[agent_name]
    
    def get_agent_names(self) -> List[str]:
        """에이전트 이름 목록"""
        return list(self.agents.keys())
    
    def step(self) -> Dict:
        """
        시뮬레이션 한 스텝 진행
        
        과정:
        1. 시간 진행
        2. 에이전트들의 계획 실행
        3. 자동 상호작용 생성
        4. 랜덤 이벤트 발생
        5. 에이전트 성찰
        6. 상태 업데이트
        """
        # 1. 시간 진행
        self.current_time += self.time_step
        
        step_summary = {
            'time': self.current_time,
            'conversations': [],
            'events': [],
            'agent_changes': {}
        }
        
        # 2. 에이전트들의 플랜 실행
        for agent in self.agents.values():
            self._execute_agent_plans(agent)
        
        # 3. 자동 상호작용 (랜덤 페어링)
        if len(self.agents) >= 2 and random.random() < 0.7:
            interaction = self._create_random_interaction()
            if interaction:
                step_summary['conversations'].append(interaction)
        
        # 4. 랜덤 이벤트
        if random.random() < 0.3:
            event = self._generate_random_event()
            if event:
                step_summary['events'].append(event)
        
        # 5. 에이전트 성찰
        for agent in self.agents.values():
            old_status = agent.get_status()
            agent.reflect(self.current_time)
            new_status = agent.get_status()
            
            # 변화 기록
            changes = {}
            if old_status['emotion'] != new_status['emotion']:
                changes['emotion'] = new_status['emotion']
            if old_status['energy'] != new_status['energy']:
                changes['energy'] = new_status['energy']
            
            if changes:
                step_summary['agent_changes'][agent.name] = changes
        
        # 6. 에너지 회복 (시간에 따라)
        for agent in self.agents.values():
            agent.energy = min(1.0, agent.energy + 0.02)
        
        return step_summary
    
    def _execute_agent_plans(self, agent: Agent):
        """에이전트의 계획 실행"""
        # 활성 플랜이 없으면 새 플랜 생성
        active_plans = [p for p in agent.plans if p.status == 'active']
        
        if not active_plans:
            available_agents = [n for n in self.agents.keys() if n != agent.name]
            new_plan = agent.plan_next_action(available_agents)
            
            if new_plan:
                agent.plans.append(new_plan)
                new_plan.status = 'active'
    
    def _create_random_interaction(self) -> Optional[Dict]:
        """랜덤 에이전트 간 상호작용 생성"""
        agent_names = list(self.agents.keys())
        
        if len(agent_names) < 2:
            return None
        
        # 두 에이전트 선택
        agent1_name, agent2_name = random.sample(agent_names, 2)
        agent1 = self.agents[agent1_name]
        agent2 = self.agents[agent2_name]
        
        # 대화 주제 선택
        topics = [
            "오늘 날씨가 좋네요",
            "잘 지냈어요?",
            "새로운 소식 있어요?",
            "oplan이 있으세요?",
            "같이 산책할까요?",
            "이야기 나눌까요?",
            "obook 읽었어요?",
            "좋은 하루네요",
        ]
        
        topic = random.choice(topics)
        
        # 대화 실행
        response = agent1.interact_with(agent2, topic, self.current_time)
        
        # 로그 저장
        conversation = {
            'time': self.current_time,
            'from': agent1_name,
            'to': agent2_name,
            'message': topic,
            'response': response
        }
        self.conversation_log.append(conversation)
        self.total_conversations += 1
        
        # 이벤트 기록
        self._log_event(
            EventType.CONVERSATION,
            f"{agent1_name}이(가) {agent2_name}에게 '{topic}'라고 말함",
            [agent1_name, agent2_name]
        )
        
        return conversation
    
    def _generate_random_event(self) -> Optional[Dict]:
        """랜덤 이벤트 생성"""
        events = [
            ("비가 내리기 시작했습니다", ["all"]),
            ("무지개가 떴습니다", ["all"]),
            ("새가 노래를 부릅니다", ["all"]),
            ("바람이 불어옵니다", ["all"]),
            ("꽃이 피었습니다", ["all"]),
            ("해가 쨍쨍합니다", ["all"]),
        ]
        
        description, targets = random.choice(events)
        
        # 모든 에이전트에게 관찰
        for agent in self.agents.values():
            agent.observe(
                description,
                list(self.agents.keys()),
                self.current_time,
                importance=0.3
            )
        
        self._log_event(
            EventType.RANDOM_EVENT,
            description,
            list(self.agents.keys())
        )
        
        return {
            'time': self.current_time,
            'description': description,
            'targets': targets
        }
    
    def _log_event(self, event_type: EventType, description: str, 
                   participants: List[str]):
        """이벤트 로깅"""
        event = VillageEvent(
            timestamp=self.current_time,
            event_type=event_type,
            description=description,
            participants=participants
        )
        self.events.append(event)
        self.total_events += 1
        
        # 콜백 호출
        if self.on_event:
            self.on_event(event)
    
    def simulate(self, num_steps: int, verbose: bool = True) -> List[Dict]:
        """여러 스텝 시뮬레이션"""
        results = []
        
        for i in range(num_steps):
            step_result = self.step()
            results.append(step_result)
            
            if verbose:
                print(f"\n=== 시간 {step_result['time']:.1f} ===")
                for conv in step_result['conversations']:
                    print(f"  💬 {conv['from']} → {conv['to']}: {conv['message']}")
                    print(f"     ↳ {conv['response']}")
                for event in step_result['events']:
                    print(f"  🌟 {event['description']}")
        
        return results
    
    def get_village_status(self) -> Dict:
        """마을 전체 상태"""
        agent_statuses = {}
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_status()
        
        # 관계 네트워크
        relationships = []
        for agent in self.agents.values():
            for other_name, relation in agent.memory.social_memory.items():
                if other_name in self.agents:
                    relationships.append({
                        'from': agent.name,
                        'to': other_name,
                        'trust': relation.trust,
                        'familiarity': relation.familiarity,
                        'interactions': relation.interaction_count
                    })
        
        return {
            'name': self.name,
            'time': self.current_time,
            'num_agents': len(self.agents),
            'agents': agent_statuses,
            'relationships': relationships,
            'total_conversations': self.total_conversations,
            'total_events': self.total_events
        }
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict]:
        """대화 이력"""
        return self.conversation_log[-limit:]
    
    def get_agent_memory_summary(self, agent_name: str) -> str:
        """특정 에이전트의 기억 요약"""
        if agent_name not in self.agents:
            return "에이전트를 찾을 수 없습니다."
        
        agent = self.agents[agent_name]
        
        summary_parts = [
            f"=== {agent_name}의 기억 ===",
            f"감정: {agent.current_emotion.value}",
            f"에너지: {agent.energy:.2f}",
            f"\n--- 작업 기억 ---"
        ]
        
        for mem in agent.memory.working_memory:
            summary_parts.append(f"  [{mem.event_type}] {mem.content[:60]}")
        
        summary_parts.append(f"\n--- 관계 ---")
        for other, relation in agent.memory.social_memory.items():
            summary_parts.append(f"  {other}: 신뢰 {relation.trust:.2f}, 친밀도 {relation.familiarity:.2f}")
        
        return "\n".join(summary_parts)
    
    def save_state(self, filepath: str):
        """상태 저장"""
        state = {
            'name': self.name,
            'current_time': self.current_time,
            'agents': {name: agent.to_dict() for name, agent in self.agents.items()},
            'conversation_log': self.conversation_log,
            'total_conversations': self.total_conversations,
            'total_events': self.total_events
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load_state(self, filepath: str):
        """상태 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.name = state['name']
        self.current_time = state['current_time']
        self.conversation_log = state.get('conversation_log', [])
        self.total_conversations = state.get('total_conversations', 0)
        self.total_events = state.get('total_events', 0)
        
        # 에이전트 복원 (모델 없이)
        self.agents = {}
        for name, agent_data in state['agents'].items():
            self.agents[name] = Agent.from_dict(agent_data)


def create_sample_village() -> Village:
    """샘플 빌리지 생성"""
    from agent import create_sample_agents
    
    village = Village(name="꿈의 마을")
    
    for agent in create_sample_agents():
        village.add_agent(agent)
    
    return village
