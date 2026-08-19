"""
학습 가능한 AI 빌리지 - Gradio 앱

강화학습 에이전트들이 자연어로 대화하며 학습하는 시스템

기능:
1. 빌리지 시뮬레이션 실행
2. 에이전트 학습 (강화학습)
3. 실시간 학습 곡선
4. 에이전트와 대화
5. 관계 시각화
"""

import gradio as gr
import numpy as np
from typing import List, Dict
import json
import os
import sys

# 모듈 경로 설정
sys.path.insert(0, os.path.dirname(__file__))

from village.rl_agent import RLAgent, MultiAgentSystem, ActionTemplate
from village.environment import VillageEnvironment
from village.trainer import Trainer


# ============================================================
# 전역 변수
# ============================================================

# 에이전트 시스템
agents_system = MultiAgentSystem()

# 환경
environment = VillageEnvironment(num_agents=5)

# 트레이너
trainer = Trainer(environment, agents_system)

# 학습 이력
training_history = []


# ============================================================
# 초기화 함수
# ============================================================

def initialize_village():
    """빌리지 초기화"""
    global agents_system, environment, trainer
    
    # 기존 에이전트 시스템 사용
    agents_system = MultiAgentSystem()
    
    # 기본 에이전트 생성
    default_agents = [
        ("민수", ["empathy", "question"]),
        ("지현", ["greeting", "emotion"]),
        ("태웅", ["suggestion", "observation"]),
        ("수진", ["question", "empathy"]),
        ("현우", ["greeting", "suggestion"]),
    ]
    
    for name, preferred_categories in default_agents:
        # 액션 템플릿 필터링
        from village.rl_agent import DEFAULT_ACTIONS
        templates = DEFAULT_ACTIONS  # 모든 템플릿 사용
        
        agent = RLAgent(
            name=name,
            action_templates=templates,
            state_dim=10,
            learning_rate=0.001
        )
        agents_system.add_agent(agent)
    
    agent_names = [name for name, _ in default_agents]
    environment = VillageEnvironment(agent_names=agent_names)
    trainer = Trainer(environment, agents_system)
    
    return f"빌리지 생성 완료! 에이전트: {list(agents_system.agents.keys())}"


# ============================================================
# 학습 함수
# ============================================================

def train_agents(num_episodes: int = 10):
    """에이전트 학습"""
    global trainer, training_history
    
    results = trainer.train(
        num_episodes=num_episodes,
        steps_per_episode=30,
        verbose=False
    )
    
    training_history.extend(results)
    
    # 학습 요약
    summary = trainer.get_training_summary()
    
    return summary


def get_training_curve():
    """학습 곡선 데이터"""
    if not training_history:
        return None
    
    episodes = [r['episode'] for r in training_history]
    rewards = [np.mean(list(r['rewards'].values())) for r in training_history]
    
    # Gradio 차트용 데이터
    data = {
        'episode': episodes,
        'avg_reward': rewards
    }
    
    return data


# ============================================================
# 시뮬레이션 함수
# ============================================================

def run_simulation(num_steps: int = 5):
    """빌리지 시뮬레이션"""
    global environment, agents_system
    
    history = []
    
    for _ in range(num_steps):
        agent_names = list(agents_system.agents.keys())
        
        # 랜덤으로 에이전트 선택
        current_agent_name = np.random.choice(agent_names)
        current_agent = agents_system.agents[current_agent_name]
        
        # 에이전트 상태
        agent_state = environment.get_agent_state(current_agent_name)
        
        # 상태 벡터 생성
        state = current_agent.get_state_vector(
            emotion=agent_state.get('emotion', 'neutral'),
            energy=agent_state.get('energy', 1.0),
            relationships=len(agent_state.get('relationships', {}))
        )
        
        # 행동 선택
        action_text, action_idx = current_agent.select_action(state)
        
        # 환경 스텝
        obs, reward, done, info = environment.step(
            current_agent_name,
            action_text,
            info.get('target') if info else None
        )
        
        # 보상 수신
        current_agent.receive_reward(reward)
        
        history.append({
            'agent': current_agent_name,
            'action': action_text,
            'reward': reward
        })
    
    # 학습
    for agent in agents_system.agents.values():
        agent.reflect_and_learn()
    
    # 결과 포맷
    result_lines = [f"=== 시뮬레이션 결과 ({num_steps} 스텝) ==="]
    for i, h in enumerate(history, 1):
        result_lines.append(f"{i}. {h['agent']}: {h['action']} (보상: {h['reward']:.2f})")
    
    return "\n".join(result_lines)


# ============================================================
# 에이전트 상태
# ============================================================

def get_agent_status():
    """에이전트 상태 표시"""
    status_lines = ["=== 에이전트 상태 ===\n"]
    
    for name, agent in agents_system.agents.items():
        stats = agent.get_statistics()
        agent_state = environment.get_agent_state(name)
        
        relationships = agent_state.get('relationships', {})
        rel_str = ", ".join([f"{k}: {v:.2f}" for k, v in relationships.items()])
        
        status_lines.extend([
            f"🤖 {name}",
            f"   감정: {agent_state.get('emotion', 'neutral')}",
            f"   에너지: {agent_state.get('energy', 1.0):.2f}",
            f"   만족도: {agent_state.get('satisfaction', 0.5):.2f}",
            f"   관계: {rel_str if relationships else '없음'}",
            f"   총 보상: {stats['total_reward']:.2f}",
            f"   학습 횟수: {stats['episodes']}",
            ""
        ])
    
    return "\n".join(status_lines)


# ============================================================
# 대화 함수
# ============================================================

def chat_with_agent(agent_name: str, user_message: str, history: list):
    """에이전트와 대화"""
    if agent_name not in agents_system.agents:
        return history, f"{agent_name} 에이전트를 찾을 수 없습니다."
    
    agent = agents_system.agents[agent_name]
    agent_state = environment.get_agent_state(agent_name)
    
    # 상태 벡터 생성
    state = agent.get_state_vector(
        emotion=agent_state.get('emotion', 'neutral'),
        energy=agent_state.get('energy', 1.0),
        relationships=len(agent_state.get('relationships', {}))
    )
    
    # 행동 선택
    action_text, action_idx = agent.select_action(state)
    
    # 환경에 기록
    obs, reward, done, info = environment.step(
        agent_name,
        user_message,  # 사용자 메시지를 환경에 기록
        agent_name
    )
    
    # 보상 수신
    agent.receive_reward(reward)
    
    # 대화 기록 업데이트
    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": action_text})
    
    return history, f"🤖 {agent_name}: {action_text}\n(보상: {reward:.2f})"


# ============================================================
# 액션 확률 시각화
# ============================================================

def show_action_probabilities(agent_name: str):
    """에이전트의 액션 확률 표시"""
    if agent_name not in agents_system.agents:
        return "에이전트를 찾을 수 없습니다."
    
    agent = agents_system.agents[agent_name]
    agent_state = environment.get_agent_state(agent_name)
    
    state = agent.get_state_vector(
        emotion=agent_state.get('emotion', 'neutral'),
        energy=agent_state.get('energy', 1.0),
        relationships=len(agent_state.get('relationships', {}))
    )
    
    probs = agent.get_action_probabilities(state)
    
    # 확률 정렬
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    lines = [f"=== {agent_name} 액션 확률 ===\n"]
    for action, prob in sorted_probs:
        bar = "█" * int(prob * 50)
        lines.append(f"{action}: {prob:.3f} {bar}")
    
    return "\n".join(lines)


# ============================================================
# 관계 시각화
# ============================================================

def show_relationships():
    """에이전트 간 관계 시각화"""
    matrix = environment.get_relationship_matrix()
    agent_names = sorted(environment.agent_states.keys())
    
    lines = ["=== 관계 행렬 ===\n"]
    
    # 헤더
    header = "     " + "  ".join([f"{n[:2]:>6}" for n in agent_names])
    lines.append(header)
    lines.append("-" * len(header))
    
    # 행렬
    for i, name in enumerate(agent_names):
        row = f"{name[:2]:>4} "
        for j in range(len(agent_names)):
            val = matrix[i, j]
            if i == j:
                row += "  ----  "
            else:
                row += f"  {val:.2f}  "
        lines.append(row)
    
    return "\n".join(lines)


# ============================================================
# 메인 앱
# ============================================================

def create_app():
    """Gradio 앱 생성"""
    
    with gr.Blocks(
        title="🧠 학습 가능한 AI 빌리지",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🧠 학습 가능한 AI 빌리지
        
        강화학습 에이전트들이 자연어로 대화하며 학습하는 시스템
        
        **핵심 기능:**
        - 🎯 자연어로 행동 (인사, 질문, 공감, 제안 등)
        - 📈 강화학습으로 점진적 개선
        - 🏘️ 빌리지 환경에서 상호작용
        - 💬 에이전트와 직접 대화
        """)
        
        with gr.Tabs():
            
            # 탭 1: 학습
            with gr.Tab("📚 학습"):
                gr.Markdown("### 에이전트 학습")
                gr.Markdown("강화학습으로 에이전트들이 행동 전략을 학습합니다.")
                
                with gr.Row():
                    init_btn = gr.Button("🔄 빌리지 초기화", variant="primary")
                    train_btn = gr.Button("🎓 학습 시작", variant="primary")
                
                episodes_input = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1,
                    label="학습 에피소드 수"
                )
                
                training_output = gr.Textbox(
                    label="학습 결과",
                    lines=15,
                    interactive=False
                )
                
                init_btn.click(
                    fn=initialize_village,
                    outputs=training_output
                )
                
                train_btn.click(
                    fn=train_agents,
                    inputs=episodes_input,
                    outputs=training_output
                )
            
            # 탭 2: 시뮬레이션
            with gr.Tab("🎮 시뮬레이션"):
                gr.Markdown("### 빌리지 시뮬레이션")
                gr.Markdown("에이전트들이 자율적으로 상호작용합니다.")
                
                steps_input = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="시뮬레이션 스텝 수"
                )
                
                sim_btn = gr.Button("▶️ 시뮬레이션 실행", variant="primary")
                sim_output = gr.Textbox(
                    label="시뮬레이션 결과",
                    lines=15,
                    interactive=False
                )
                
                sim_btn.click(
                    fn=run_simulation,
                    inputs=steps_input,
                    outputs=sim_output
                )
            
            # 탭 3: 에이전트 상태
            with gr.Tab("📊 에이전트 상태"):
                gr.Markdown("### 에이전트 상태 및 확률")
                
                refresh_btn = gr.Button("🔄 새로고침")
                status_output = gr.Textbox(
                    label="에이전트 상태",
                    lines=20,
                    interactive=False
                )
                
                agent_selector = gr.Dropdown(
                    choices=list(agents_system.agents.keys()),
                    label="에이전트 선택",
                    value="민수"
                )
                
                prob_btn = gr.Button("📊 액션 확률 보기")
                prob_output = gr.Textbox(
                    label="액션 확률",
                    lines=10,
                    interactive=False
                )
                
                refresh_btn.click(
                    fn=get_agent_status,
                    outputs=status_output
                )
                
                prob_btn.click(
                    fn=show_action_probabilities,
                    inputs=agent_selector,
                    outputs=prob_output
                )
            
            # 탭 4: 대화
            with gr.Tab("💬 대화"):
                gr.Markdown("### 에이전트와 대화")
                gr.Markdown("학습된 에이전트와 직접 대화해보세요.")
                
                chat_agent_selector = gr.Dropdown(
                    choices=list(agents_system.agents.keys()),
                    label="대화할 에이전트",
                    value="민수"
                )
                
                chatbot = gr.Chatbot(label="대화 내용")
                msg_input = gr.Textbox(
                    label="메시지 입력",
                    placeholder="메시지를 입력하세요..."
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 전송", variant="primary")
                    clear_btn = gr.Button("🗑️ 대화 지우기")
                
                chat_history = gr.State([])
                
                def respond(message, history, agent_name):
                    new_history, response = chat_with_agent(
                        agent_name, message, history
                    )
                    return new_history, new_history, response
                
                send_btn.click(
                    fn=respond,
                    inputs=[msg_input, chat_history, chat_agent_selector],
                    outputs=[chatbot, chat_history, msg_input]
                )
                
                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, msg_input]
                )
            
            # 탭 5: 관계
            with gr.Tab("👥 관계"):
                gr.Markdown("### 에이전트 간 관계")
                gr.Markdown("상호작용을 통해 형성된 관계를 시각화합니다.")
                
                rel_btn = gr.Button("📊 관계 새로고침")
                rel_output = gr.Textbox(
                    label="관계 행렬",
                    lines=15,
                    interactive=False
                )
                
                rel_btn.click(
                    fn=show_relationships,
                    outputs=rel_output
                )
            
            # 탭 6: 학습 곡선
            with gr.Tab("📈 학습 곡선"):
                gr.Markdown("### 학습 진행 상황")
                gr.Markdown("에피소드별 평균 보상 변화를 추적합니다.")
                
                curve_btn = gr.Button("📈 차트 업데이트")
                curve_output = gr.Dataframe(
                    headers=["에피소드", "평균 보상"],
                    label="학습 이력"
                )
                
                def update_curve():
                    data = get_training_curve()
                    if data is None:
                        return []
                    
                    rows = []
                    for ep, rew in zip(data['episode'], data['avg_reward']):
                        rows.append([ep, round(rew, 3)])
                    return rows
                
                curve_btn.click(
                    fn=update_curve,
                    outputs=curve_output
                )
        
        # 하단 정보
        gr.Markdown("""
        ---
        ### 📖 사용법
        
        1. **학습 탭**: 빌리지 초기화 → 학습 에피소드 설정 → 학습 시작
        2. **시뮬레이션 탭**: 학습된 에이전트들의 자율적 상호작용 관찰
        3. **에이전트 상태 탭**: 각 에이전트의 상태와 행동 확률 확인
        4. **대화 탭**: 에이전트와 직접 대화하며 학습 정도 체험
        5. **관계 탭**: 에이전트 간 형성된 관계 시각화
        6. **학습 곡선 탭**: 학습 진행 상황 모니터링
        
        ---
        **강화학습 원리:**
        - 각 에이전트는 여러 행동 템플릿 중 선택
        - 행동 결과 보상 수신
        - 높은 보상을 얻은 행동은 더 자주 선택되도록 학습
        - 반복을 통해 최적 전략 발견
        """)
    
    return app


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # 빌리지 초기화
    initialize_village()
    
    # 앱 생성 및 실행
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
