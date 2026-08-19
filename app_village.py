#!/usr/bin/env python3
"""
🏘️ 한국어 에이전트 빌리지 - Gradio 앱

기능:
1. 에이전트들이 자동으로 상호작용
2. 시간 흐름에 따른 시뮬레이션
3. 기억/관계 시각화
4. 사용자 개입 가능
"""

import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import json
from typing import List, Dict

from village.village import Village, create_sample_village
from village.agent import Agent, Personality, Emotion


class VillageApp:
    """빌리지 Gradio 앱"""
    
    def __init__(self):
        self.village = None
        self.init_village()
    
    def init_village(self):
        """빌리지 초기화"""
        self.village = create_sample_village()
    
    def run_simulation(self, num_steps: int) -> str:
        """시뮬레이션 실행"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        results = self.village.simulate(num_steps, verbose=False)
        
        # 결과 포맷팅
        output_parts = [f"=== 시뮬레이션 완료: {num_steps}스텝 ===\n"]
        
        for result in results:
            time = result['time']
            output_parts.append(f"\n⏰ 시간 {time:.1f}")
            
            for conv in result['conversations']:
                output_parts.append(
                    f"  💬 {conv['from']} → {conv['to']}\n"
                    f"     메시지: {conv['message']}\n"
                    f"     응답: {conv['response']}"
                )
            
            for event in result['events']:
                output_parts.append(f"  🌟 {event['description']}")
            
            for agent_name, changes in result['agent_changes'].items():
                if 'emotion' in changes:
                    output_parts.append(
                        f"  😊 {agent_name} 감정 변화: {changes['emotion']}"
                    )
        
        # 최종 상태
        status = self.village.get_village_status()
        output_parts.append(f"\n\n=== 최종 상태 ===")
        output_parts.append(f"총 대화 수: {status['total_conversations']}")
        output_parts.append(f"총 이벤트 수: {status['total_events']}")
        
        for agent_name, agent_status in status['agents'].items():
            output_parts.append(
                f"\n👤 {agent_name}: "
                f"감정={agent_status['emotion']}, "
                f"에너지={agent_status['energy']:.2f}, "
                f"기억={agent_status['memories']}개"
            )
        
        return "\n".join(output_parts)
    
    def get_village_status(self) -> str:
        """현재 빌리지 상태"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        status = self.village.get_village_status()
        
        parts = [
            f"🏘️ {status['name']}",
            f"⏰ 시간: {status['time']:.1f}",
            f"👥 에이전트 수: {status['num_agents']}",
            f"💬 총 대화: {status['total_conversations']}",
            f"🌟 총 이벤트: {status['total_events']}",
            "\n=== 에이전트 상태 ==="
        ]
        
        for agent_name, agent_status in status['agents'].items():
            emotion_emoji = {
                'neutral': '😐',
                'happy': '😊',
                'sad': '😢',
                'angry': '😠',
                'excited': '🤩',
                'calm': '😌',
                'curious': '🤔',
                'anxious': '😰'
            }.get(agent_status['emotion'], '❓')
            
            parts.append(
                f"\n{emotion_emoji} {agent_name}\n"
                f"   감정: {agent_status['emotion']}\n"
                f"   에너지: {'█' * int(agent_status['energy'] * 10)}{'░' * (10 - int(agent_status['energy'] * 10))} {agent_status['energy']:.0%}\n"
                f"   관계 수: {agent_status['relationships']}\n"
                f"   기억 수: {agent_status['memories']}"
            )
        
        # 관계 네트워크
        if status['relationships']:
            parts.append("\n=== 관계 네트워크 ===")
            for rel in status['relationships']:
                trust_bar = '█' * int(rel['trust'] * 5) + '░' * (5 - int(rel['trust'] * 5))
                parts.append(
                    f"  {rel['from']} ↔ {rel['to']}: "
                    f"신뢰[{trust_bar}] "
                    f"({rel['interactions']}회 만남)"
                )
        
        return "\n".join(parts)
    
    def get_conversation_history(self) -> str:
        """대화 이력"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        history = self.village.get_conversation_history(limit=30)
        
        if not history:
            return "아직 대화가 없습니다."
        
        parts = ["=== 최근 대화 이력 ===\n"]
        
        for conv in reversed(history):
            parts.append(
                f"[{conv['time']:.1f}] "
                f"{conv['from']} → {conv['to']}\n"
                f"  💬 {conv['message']}\n"
                f"  ↳ {conv['response']}\n"
            )
        
        return "\n".join(parts)
    
    def get_agent_memory(self, agent_name: str) -> str:
        """에이전트 기억 조회"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        return self.village.get_agent_memory_summary(agent_name)
    
    def chat_with_agent(self, agent_name: str, message: str) -> str:
        """에이전트와 대화"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        if agent_name not in self.village.agents:
            return f"{agent_name} 에이전트를 찾을 수 없습니다."
        
        agent = self.village.agents[agent_name]
        
        # 임시 사용자 에이전트 생성
        user_agent = Agent(
            name="사용자",
            personality=Personality(extraversion=0.7, agreeableness=0.8)
        )
        
        # 대화 실행
        response = agent.interact_with(
            user_agent, 
            message, 
            self.village.current_time
        )
        
        # 빌리지 이력에도 저장
        self.village.conversation_log.append({
            'time': self.village.current_time,
            'from': '사용자',
            'to': agent_name,
            'message': message,
            'response': response
        })
        
        return f"🤖 {agent_name}: {response}"
    
    def add_agent(self, name: str, openness: float, conscientiousness: float,
                  extraversion: float, agreeableness: float, neuroticism: float) -> str:
        """새 에이전트 추가"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        if name in self.village.agents:
            return f"{name} 에이전트는 이미 존재합니다."
        
        agent = Agent(
            name=name,
            personality=Personality(
                openness=openness,
                conscientiousness=conscientiousness,
                extraversion=extraversion,
                agreeableness=agreeableness,
                neuroticism=neuroticism
            )
        )
        
        self.village.add_agent(agent)
        return f"✅ {name} 에이전트가 추가되었습니다!"
    
    def reset_village(self) -> str:
        """빌리지 초기화"""
        self.init_village()
        return "✅ 빌리지가 초기화되었습니다."
    
    def save_village(self) -> str:
        """빌리지 상태 저장"""
        if not self.village:
            return "빌리지가 초기화되지 않았습니다."
        
        filepath = "village_state.json"
        self.village.save_state(filepath)
        return f"✅ 상태가 저장되었습니다: {filepath}"
    
    def create_interface(self) -> gr.Blocks:
        """Gradio 인터페이스 생성"""
        
        with gr.Blocks(
            title="🏘️ 한국어 에이전트 빌리지",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown("""
            # 🏘️ 한국어 에이전트 빌리지
            
            기억과 추론이 가능한 에이전트들이 살아가는 마을 시뮬레이션입니다.
            
            **기능:**
            - 에이전트들이 자동으로 상호작용하며 대화
            - 시간 흐름에 따른 관계 형성 및 기억 축적
            - 에이전트와 직접 대화 가능
            - 관계 네트워크 시각화
            """)
            
            with gr.Tabs():
                
                # 탭 1: 시뮬레이션
                with gr.Tab("🎮 시뮬레이션"):
                    gr.Markdown("### 시간 흐름 시뮬레이션")
                    
                    with gr.Row():
                        steps_input = gr.Slider(
                            minimum=1, maximum=50, value=5, step=1,
                            label="시뮬레이션 스텝 수"
                        )
                        run_btn = gr.Button("▶️ 시뮬레이션 실행", variant="primary")
                    
                    sim_output = gr.Textbox(
                        label="시뮬레이션 결과",
                        lines=20,
                        interactive=False
                    )
                    
                    run_btn.click(
                        fn=self.run_simulation,
                        inputs=steps_input,
                        outputs=sim_output
                    )
                
                # 탭 2: 마을 상태
                with gr.Tab("🏘️ 마을 상태"):
                    gr.Markdown("### 현재 마을 상태")
                    
                    refresh_btn = gr.Button("🔄 새로고침", variant="secondary")
                    status_output = gr.Textbox(
                        label="마을 상태",
                        lines=25,
                        interactive=False
                    )
                    
                    refresh_btn.click(
                        fn=self.get_village_status,
                        outputs=status_output
                    )
                
                # 탭 3: 대화 이력
                with gr.Tab("📜 대화 이력"):
                    gr.Markdown("### 최근 대화 이력")
                    
                    history_refresh = gr.Button("🔄 새로고침")
                    history_output = gr.Textbox(
                        label="대화 이력",
                        lines=20,
                        interactive=False
                    )
                    
                    history_refresh.click(
                        fn=self.get_conversation_history,
                        outputs=history_output
                    )
                
                # 탭 4: 에이전트 대화
                with gr.Tab("💬 에이전트 대화"):
                    gr.Markdown("### 에이전트와 대화하기")
                    
                    with gr.Row():
                        agent_selector = gr.Dropdown(
                            choices=["민수", "지현", "태웅", "수진", "현우"],
                            label="에이전트 선택",
                            value="민수"
                        )
                    
                    chat_input = gr.Textbox(
                        label="메시지",
                        placeholder="에이전트에게 말을 걸어보세요..."
                    )
                    chat_btn = gr.Button("📤 전송", variant="primary")
                    chat_output = gr.Textbox(
                        label="응답",
                        lines=3,
                        interactive=False
                    )
                    
                    chat_btn.click(
                        fn=self.chat_with_agent,
                        inputs=[agent_selector, chat_input],
                        outputs=chat_output
                    )
                    
                    # 기억 조회
                    gr.Markdown("### 에이전트 기억 조회")
                    memory_btn = gr.Button("🧠 기억 조회")
                    memory_output = gr.Textbox(
                        label="기억",
                        lines=15,
                        interactive=False
                    )
                    
                    memory_btn.click(
                        fn=self.get_agent_memory,
                        inputs=agent_selector,
                        outputs=memory_output
                    )
                
                # 탭 5: 에이전트 관리
                with gr.Tab("👥 에이전트 관리"):
                    gr.Markdown("### 새 에이전트 추가")
                    
                    with gr.Row():
                        new_name = gr.Textbox(label="이름", placeholder="홍길동")
                    
                    with gr.Row():
                        openness = gr.Slider(0, 1, 0.5, label="개방성")
                        conscientiousness = gr.Slider(0, 1, 0.5, label="성실성")
                    
                    with gr.Row():
                        extraversion = gr.Slider(0, 1, 0.5, label="외향성")
                        agreeableness = gr.Slider(0, 1, 0.5, label="친화성")
                    
                    neuroticism = gr.Slider(0, 1, 0.5, label="신경성")
                    
                    add_agent_btn = gr.Button("➕ 에이전트 추가", variant="primary")
                    add_result = gr.Textbox(label="결과", interactive=False)
                    
                    add_agent_btn.click(
                        fn=self.add_agent,
                        inputs=[new_name, openness, conscientiousness, 
                                extraversion, agreeableness, neuroticism],
                        outputs=add_result
                    )
                    
                    # 초기화
                    gr.Markdown("---")
                    reset_btn = gr.Button("🔄 빌리지 초기화", variant="stop")
                    save_btn = gr.Button("💾 상태 저장", variant="secondary")
                    
                    reset_btn.click(fn=self.reset_village, outputs=add_result)
                    save_btn.click(fn=self.save_village, outputs=add_result)
            
            # 하단 정보
            gr.Markdown("""
            ---
            ### 📖 사용법
            
            1. **시뮬레이션 탭**: 슬라이더로 스텝 수 설정 후 실행
            2. **마을 상태 탭**: 에이전트들의 현재 상태 확인
            3. **대화 이력 탭**: 자동으로 생성된 대화 확인
            4. **에이전트 대화 탭**: 직접 에이전트와 대화
            5. **에이전트 관리 탭**: 새 에이전트 추가 및 관리
            
            ### 🎯 관찰 포인트
            
            - 에이전트들의 관계가 시간에 따라 변합니다
            - 대화를 많이 할수록 친밀도가 올라갑니다
            - 성격에 따라 대화 패턴이 다릅니다
            - 기억이 축적되면 더 나은 응답을 합니다
            """)
        
        return demo


def main():
    """메인 실행"""
    app = VillageApp()
    demo = app.create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
