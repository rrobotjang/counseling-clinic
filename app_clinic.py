"""
심리상담 클리닉 - 통합 Gradio 앱

강화학습 상담사 + 한국어 언어모델 통합 시스템
"""

import gradio as gr
import torch
import os
import sys
import random
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'korean_chatbot_app_v2'))
sys.path.insert(0, os.path.dirname(__file__))

from village.counselor_agent import CounselorAgent
from village.client_agent import ClientAgent, create_default_clients
from village.clinic_environment import CounselingClinicEnvironment, CounselingPhase
from village.api_client import create_client, OfflineCounselingClient
from village.database import init_database


# ============================================================
# 한국어 모델 로더
# ============================================================

class ChatbotModelLoader:
    """학습된 한국어 챗봇 모델 로더"""
    
    def __init__(self):
        self.model = None
        self.sp = None
        self.max_length = 40
        self.device = torch.device("cpu")
        self.loaded = False
    
    def load(self, checkpoint_dir: str = None):
        """모델 로드"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(__file__), 
                'korean_chatbot_app_v2', 
                'checkpoints'
            )
        
        pt_path = os.path.join(checkpoint_dir, 'chatbot_transformer.pt')
        sp_path = os.path.join(checkpoint_dir, 'spm_korean.model')
        
        if not os.path.exists(pt_path) or not os.path.exists(sp_path):
            print(f"모델 파일을 찾을 수 없습니다: {checkpoint_dir}")
            return False
        
        try:
            import sentencepiece as spm
            
            checkpoint = torch.load(pt_path, map_location=self.device)
            config = checkpoint["config"]
            
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sp_path)
            
            from model import Transformer
            
            self.model = Transformer(
                vocab_size=config["vocab_size"],
                num_layers=config["num_layers"],
                units=config["units"],
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                max_length=config["max_length"],
                dropout=config["dropout"],
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.max_length = config["max_length"]
            self.loaded = True
            
            print("✅ 한국어 모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    @torch.no_grad()
    def generate(self, text: str) -> str:
        """텍스트 응답 생성"""
        if not self.loaded or self.model is None:
            return None
        
        import re
        text = text.strip()
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r"[\s]+", " ", text)
        text = re.sub(r"[^가-힣0-9?.!,\s]+", " ", text)
        text = text.strip()
        
        enc_input_ids = [self.sp.bos_id()] + self.sp.encode(text) + [self.sp.eos_id()]
        enc_input = torch.tensor([enc_input_ids], dtype=torch.long, device=self.device)
        dec_input = torch.tensor([[self.sp.bos_id()]], dtype=torch.long, device=self.device)
        
        for _ in range(self.max_length):
            predictions = self.model(enc_input, dec_input)
            predicted_id = torch.argmax(predictions[:, -1:, :], dim=-1)
            if predicted_id.item() == self.sp.eos_id():
                break
            dec_input = torch.cat([dec_input, predicted_id], dim=-1)
        
        output_seq = dec_input.squeeze(0).tolist()
        reply = self.sp.decode(
            [t for t in output_seq if t not in (self.sp.bos_id(), self.sp.eos_id(), self.sp.pad_id())]
        )
        
        return reply if reply.strip() else None


# ============================================================
# 전역 변수
# ============================================================

chatbot_model = ChatbotModelLoader()
counselor = CounselorAgent(name="김상담사", state_dim=15)
clients = create_default_clients()
clinic_env = CounselingClinicEnvironment()
current_client_name: Optional[str] = None
current_session_active: bool = False


# ============================================================
# 초기화
# ============================================================

counseling_client = None
api_server_mode = False


def initialize_clinic(use_api: bool = False):
    """클리닉 초기화"""
    global counselor, clients, clinic_env, counseling_client, api_server_mode
    
    counselor = CounselorAgent(name="김상담사", state_dim=15)
    clients = create_default_clients()
    clinic_env = CounselingClinicEnvironment()
    api_server_mode = use_api
    
    if use_api:
        counseling_client = create_client("api")
        mode_info = "API 서버 모드"
    else:
        db_path = os.path.join(os.path.dirname(__file__), 'counseling.db')
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'ChatbotData.csv')
        
        if not os.path.exists(db_path):
            init_database(csv_path, db_path)
        
        counseling_client = create_client("offline", db_path)
        mode_info = "오프라인 모드"
    
    client_list = "\n".join([f"• {c.name} ({c.age}세, {c.gender}) - {', '.join(c.issues)}" for c in clients.values()])
    
    return f"심리상담 클리닉이 초기화되었습니다. ({mode_info})\n\n등록된 내담자:\n{client_list}"


def load_chatbot_model():
    """챗봇 모델 로드"""
    success = chatbot_model.load()
    if success:
        return "✅ 한국어 모델 로드 완료! 상담사 응답에 언어 모델이 적용됩니다."
    else:
        return "⚠️ 모델 로드 실패. 기본 템플릿 응답을 사용합니다."


# ============================================================
# 세션 관리
# ============================================================

def start_session(client_name: str):
    """세션 시작"""
    global current_client_name, current_session_active
    
    if client_name not in clients:
        return "내담자를 선택해주세요.", ""
    
    current_client_name = client_name
    current_session_active = True
    
    client = clients[client_name]
    client.start_new_session()
    
    obs = clinic_env.reset(client_name)
    
    greeting = f"안녕하세요, {client.name}님. 저는 오늘 상담을 도와드릴 김상담사입니다.\n\n오늘은 어떤 이야기를 나누고 싶으신가요?"
    
    clinic_env.current_session.dialogue_history.append({
        'role': 'counselor',
        'content': greeting,
        'category': '공감',
        'time': 0
    })
    
    session_info = f"""=== 세션 시작 ===
내담자: {client.name} ({client.age}세, {client.gender})
주요 문제: {', '.join(client.issues)}
배경: {client.background}

현재 감정 상태:
• 불안: {client.emotional_state['anxiety']:.2f}
• 슬픔: {client.emotional_state['sadness']:.2f}
• 희망: {client.emotional_state['hope']:.2f}
• 신뢰: {client.emotional_state['trust']:.2f}
"""
    
    return greeting, session_info


def end_session():
    """세션 종료"""
    global current_session_active
    
    if not current_session_active or current_client_name is None:
        return "활성 세션이 없습니다."
    
    client = clients[current_client_name]
    summary = clinic_env.get_session_summary()
    
    current_session_active = False
    
    result = f"""=== 세션 종료 ===
내담자: {client.name}
총 대화 수: {summary.get('total_dialogues', 0)}
개선도: {summary.get('improvement', 0):.2f}

최종 감정 상태:
• 불안: {client.emotional_state['anxiety']:.2f}
• 슬픔: {client.emotional_state['sadness']:.2f}
• 희망: {client.emotional_state['hope']:.2f}
• 신뢰: {client.emotional_state['trust']:.2f}

상담이 종료되었습니다. 수고하셨습니다."""
    
    return result


# ============================================================
# 상담 대화
# ============================================================

def counseling_chat(user_message: str, history: list, use_model: bool = True):
    """상담 대화 처리"""
    global current_client_name, current_session_active
    
    if not current_session_active or current_client_name is None:
        return history, "⚠️ 먼저 세션을 시작해주세요.", ""
    
    client = clients[current_client_name]
    
    client_info = {
        'emotional_state': client.emotional_state,
        'energy': client.energy,
        'openness': client.openness,
        'phase': clinic_env.current_session.phase.value if clinic_env.current_session else '탐색',
        'dialogue_count': len(clinic_env.current_session.dialogue_history) if clinic_env.current_session else 0,
        'last_user_message': user_message
    }
    
    rl_response, category = counselor.plan_response(client_info)
    
    api_response = None
    if counseling_client:
        try:
            api_result = counseling_client.counsel(
                user_message=user_message,
                counseling_style=category,
                client_emotions=client.emotional_state
            )
            if api_result and 'response' in api_result:
                api_response = api_result['response']
                similarity = api_result.get('similarity', 0)
        except Exception:
            pass
    
    if use_model and chatbot_model.loaded:
        model_response = chatbot_model.generate(user_message)
        if model_response and len(model_response) > 3:
            response = model_response
        elif api_response:
            response = api_response
        else:
            response = rl_response
    elif api_response:
        response = api_response
    else:
        response = rl_response
    
    obs, reward, done, info = clinic_env.step(response, category)
    
    counselor.receive_reward(reward)
    
    client.add_memory(
        session_id=client.session_count,
        time=clinic_env.current_time,
        content=user_message,
        importance=0.6
    )
    
    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    
    session_info = f"""[상담 진행 상황]
단계: {info.get('phase', '탐색')}
행동 유형: {category}
보상: {reward:.2f}

[내담자 상태]
불안: {client.emotional_state['anxiety']:.2f} | 슬픔: {client.emotional_state['sadness']:.2f}
희망: {client.emotional_state['hope']:.2f} | 신뢰: {client.emotional_state['trust']:.2f}
개선도: {info.get('improvement', 0):.2f}"""
    
    return history, history, session_info


# ============================================================
# 시각화
# ============================================================

def get_client_status():
    """내담자 상태 표시"""
    status_lines = ["=== 내담자 상태 ===\n"]
    
    for name, client in clients.items():
        status_lines.extend([
            f"👤 {name} ({client.age}세, {client.gender})",
            f"   문제: {', '.join(client.issues)}",
            f"   불안: {client.emotional_state['anxiety']:.2f}",
            f"   슬픔: {client.emotional_state['sadness']:.2f}",
            f"   희망: {client.emotional_state['hope']:.2f}",
            f"   신뢰: {client.emotional_state['trust']:.2f}",
            f"   에너지: {client.energy:.2f}",
            f"   개방성: {client.openness:.2f}",
            f"   세션 횟수: {client.session_count}",
            f"   기억 수: {len(client.memory)}",
            ""
        ])
    
    return "\n".join(status_lines)


def get_counselor_stats():
    """상담사 통계"""
    stats = counselor.get_statistics()
    
    return f"""=== 상담사 통계 ===
이름: {stats['name']}
학습 횟수: {stats['episodes']}
총 보상: {stats['total_reward']:.2f}
평균 보상: {stats['avg_reward']:.3f}"""


def show_action_probabilities():
    """상담사 행동 확률"""
    if current_client_name is None or not current_session_active:
        return "활성 세션이 없습니다."
    
    client = clients[current_client_name]
    
    client_info = {
        'emotional_state': client.emotional_state,
        'energy': client.energy,
        'openness': client.openness,
        'phase': clinic_env.current_session.phase.value if clinic_env.current_session else '탐색',
        'dialogue_count': len(clinic_env.current_session.dialogue_history) if clinic_env.current_session else 0
    }
    
    state = counselor.get_state_vector(
        emotional_state=client_info['emotional_state'],
        energy=client_info['energy'],
        openness=client_info['openness'],
        phase=client_info['phase'],
        dialogue_count=client_info['dialogue_count']
    )
    
    probs = counselor.get_action_probabilities(state)
    
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    lines = ["=== 상담사 행동 확률 ===\n"]
    for action, prob in sorted_probs:
        bar = "█" * int(prob * 40)
        lines.append(f"{action}: {prob:.3f} {bar}")
    
    return "\n".join(lines)


# ============================================================
# 학습
# ============================================================

def train_counselor(num_episodes: int = 10):
    """상담사 학습"""
    global counselor
    
    results = []
    
    for episode in range(num_episodes):
        client_name = random.choice(list(clients.keys()))
        client = clients[client_name]
        
        clinic_env.reset(client_name)
        client.start_new_session()
        
        episode_reward = 0
        
        for step in range(15):
            client_info = {
                'emotional_state': client.emotional_state,
                'energy': client.energy,
                'openness': client.openness,
                'phase': clinic_env.current_session.phase.value,
                'dialogue_count': len(clinic_env.current_session.dialogue_history)
            }
            
            response, category = counselor.plan_response(client_info)
            
            state = counselor.get_state_vector(
                emotional_state=client_info['emotional_state'],
                energy=client_info['energy'],
                openness=client_info['openness'],
                phase=client_info['phase'],
                dialogue_count=client_info['dialogue_count']
            )
            counselor.select_action(state)
            
            obs, reward, done, info = clinic_env.step(response, category)
            
            counselor.receive_reward(reward)
            episode_reward += reward
            
            if done:
                break
        
        counselor.learn()
        results.append(episode_reward)
    
    avg_reward = sum(results) / len(results) if results else 0
    max_reward = max(results) if results else 0
    min_reward = min(results) if results else 0
    
    return f"""=== 학습 완료 ===
에피소드: {num_episodes}
평균 보상: {avg_reward:.3f}
최대 보상: {max_reward:.3f}
최소 보상: {min_reward:.3f}

상담사가 상담 전략을 학습했습니다."""


# ============================================================
# Gradio 앱
# ============================================================

def create_app():
    """Gradio 앱 생성"""
    
    with gr.Blocks(title="🧠 심리상담 클리닉") as app:
        
        gr.Markdown("""
        # 🧠 심리상담 클리nç
        
        강화학습 상담사와 한국어 언어모델이 통합된 심리상담 시스템
        
        **핵심 기능:**
        - 🤖 RL로 학습된 상담 전략
        - 💬 한국어 모델 기반 자연스러운 응답
        - 👥 다양한 내담자 시뮬레이션
        - 📈 실시간 감정 상태 추적
        """)
        
        with gr.Tabs():
            
            with gr.Tab("⚙️ 초기화"):
                gr.Markdown("### 시스템 초기화")
                
                api_mode = gr.Checkbox(
                    label="API 서버 모드 사용",
                    value=False
                )
                
                with gr.Row():
                    init_btn = gr.Button("🔄 클리닉 초기화", variant="primary")
                    model_btn = gr.Button("📥 한국어 모델 로드", variant="secondary")
                
                init_output = gr.Textbox(label="초기화 결과", lines=10, interactive=False)
                
                init_btn.click(fn=initialize_clinic, inputs=api_mode, outputs=init_output)
                model_btn.click(fn=load_chatbot_model, outputs=init_output)
            
            with gr.Tab("🎯 세션 관리"):
                gr.Markdown("### 상담 세션")
                
                client_selector = gr.Dropdown(
                    choices=list(clients.keys()),
                    label="내담자 선택",
                    value=list(clients.keys())[0] if clients else None
                )
                
                with gr.Row():
                    start_btn = gr.Button("▶️ 세션 시작", variant="primary")
                    end_btn = gr.Button("⏹️ 세션 종료", variant="stop")
                
                session_info = gr.Textbox(label="세션 정보", lines=15, interactive=False)
                
                start_btn.click(
                    fn=start_session,
                    inputs=client_selector,
                    outputs=[session_info, session_info]
                )
                
                end_btn.click(fn=end_session, outputs=session_info)
            
            with gr.Tab("💬 상담 대화"):
                gr.Markdown("### 상담 진행")
                
                use_model_checkbox = gr.Checkbox(
                    label="한국어 모델 사용",
                    value=True
                )
                
                chatbot = gr.Chatbot(label="상담 대화")
                msg_input = gr.Textbox(
                    label="내담자 메시지 입력",
                    placeholder="내담자의 말을 입력하세요..."
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 전송", variant="primary")
                    clear_btn = gr.Button("🗑️ 대화 지우기")
                
                chat_history = gr.State([])
                session_status = gr.Textbox(label="세션 상태", lines=10, interactive=False)
                
                def respond(message, history, use_model):
                    new_history, response, status = counseling_chat(
                        message, history, use_model
                    )
                    return new_history, new_history, status
                
                send_btn.click(
                    fn=respond,
                    inputs=[msg_input, chat_history, use_model_checkbox],
                    outputs=[chatbot, chat_history, session_status]
                )
                
                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, msg_input]
                )
            
            with gr.Tab("📊 상태"):
                gr.Markdown("### 시스템 상태")
                
                refresh_btn = gr.Button("🔄 새로고침", variant="secondary")
                
                client_status = gr.Textbox(label="내담자 상태", lines=20, interactive=False)
                counselor_status = gr.Textbox(label="상담사 통계", lines=5, interactive=False)
                
                refresh_btn.click(
                    fn=lambda: (get_client_status(), get_counselor_stats()),
                    outputs=[client_status, counselor_status]
                )
            
            with gr.Tab("📈 확률"):
                gr.Markdown("### 상담사 행동 확률")
                
                prob_btn = gr.Button("📊 확률 보기", variant="secondary")
                prob_output = gr.Textbox(label="행동 확률", lines=15, interactive=False)
                
                prob_btn.click(fn=show_action_probabilities, outputs=prob_output)
            
            with gr.Tab("🎓 학습"):
                gr.Markdown("### 상담사 학습")
                
                episodes_input = gr.Slider(
                    minimum=5, maximum=100, value=20, step=5,
                    label="학습 에피소드 수"
                )
                
                train_btn = gr.Button("🎓 학습 시작", variant="primary")
                train_output = gr.Textbox(label="학습 결과", lines=10, interactive=False)
                
                train_btn.click(
                    fn=train_counselor,
                    inputs=episodes_input,
                    outputs=train_output
                )
        
        gr.Markdown("""
        ---
        ### 📖 사용법
        
        1. **초기화 탭**: 클리닉 초기화 → 한국어 모델 로드
        2. **세션 관리 탭**: 내담자 선택 → 세션 시작
        3. **상담 대화 탭**: 내담자 메시지 입력 → 상담사 응답 확인
        4. **상태 탭**: 실시간 감정 상태 추적
        5. **확률 탭**: 상담사의 행동 선택 확률 확인
        6. **학습 탭**: 상담 전략 학습
        
        ---
        **통합 시스템 구조:**
        - RL 정책: 어떤 상담 전략(공감, 질문, 반영 등)을 사용할지 결정
        - 한국어 모델: 선택된 전략에 따른 실제 응답 생성
        - 보상 함수: 상담 효과에 기반한 학습
        """)
    
    return app


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="심리상담 클리닉 실행")
    parser.add_argument("--api", action="store_true", help="API 서버 모드로 실행")
    parser.add_argument("--port", type=int, default=7862, help="Gradio 포트")
    parser.add_argument("--api-port", type=int, default=8000, help="API 서버 포트")
    args = parser.parse_args()
    
    if args.api:
        print("API 서버 시작 중...")
        initialize_clinic(use_api=True)
        
        import threading
        from api_server import run_server
        api_thread = threading.Thread(target=run_server, args=("0.0.0.0", args.api_port))
        api_thread.daemon = True
        api_thread.start()
        print(f"✅ API 서버 시작: http://localhost:{args.api_port}")
    else:
        initialize_clinic(use_api=False)
    
    chatbot_model.load()
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
