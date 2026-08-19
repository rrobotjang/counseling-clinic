"""
심리상담 클리닉 - 모니터링 대시보드

실시간 모니터링:
1. 에이전트 상태 시각화
2. 대화 이력 추적
3. 감정 상태 차트
4. 학습 진행 상황
5. 시스템 메트릭
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from village.counselor_agent import CounselorAgent
from village.client_agent import create_default_clients, ClientAgent
from village.clinic_environment import CounselingClinicEnvironment
from village.counseling_data import CounselingResponseSystem
from village.api_client import create_client


# ============================================================
# 전역 상태
# ============================================================

counselor = CounselorAgent(name="김상담사", state_dim=15)
clients = create_default_clients()
clinic_env = CounselingClinicEnvironment()
response_system = CounselingResponseSystem()

current_client_name: Optional[str] = None
current_session_active: bool = False
monitoring_data: List[Dict] = []


# ============================================================
# 모니터링 데이터 수집
# ============================================================

def collect_monitoring_data():
    """모니터링 데이터 수집"""
    data = {
        'timestamp': time.time(),
        'clients': {},
        'counselor_stats': counselor.get_statistics(),
        'session_active': current_session_active,
        'current_client': current_client_name
    }
    
    for name, client in clients.items():
        data['clients'][name] = {
            'emotional_state': client.emotional_state.copy(),
            'energy': client.energy,
            'openness': client.openness,
            'session_count': client.session_count,
            'memory_count': len(client.memory)
        }
    
    return data


# ============================================================
# 차트 생성 함수
# ============================================================

def create_emotion_radar_chart(client_name: str):
    """감정 상태 레이더 차트"""
    if client_name not in clients:
        return go.Figure()
    
    client = clients[client_name]
    
    categories = ['불안', '슬픔', '분노', '희망', '신뢰']
    values = [
        client.emotional_state.get('anxiety', 0),
        client.emotional_state.get('sadness', 0),
        client.emotional_state.get('anger', 0),
        client.emotional_state.get('hope', 0),
        client.emotional_state.get('trust', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=client_name,
        line=dict(color='#636EFA')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title=f"{client_name} 감정 상태",
        height=400
    )
    
    return fig


def create_emotion_timeline_chart(client_name: str):
    """감정 상태 타임라인 차트"""
    if client_name not in clients:
        return go.Figure()
    
    client = clients[client_name]
    
    if not client.memory:
        x = [0]
        anxiety = [client.emotional_state.get('anxiety', 0.5)]
        sadness = [client.emotional_state.get('sadness', 0.3)]
        hope = [client.emotional_state.get('hope', 0.4)]
    else:
        x = list(range(len(client.memory)))
        anxiety = [m.get('emotion_at_time', {}).get('anxiety', 0.5) for m in client.memory]
        sadness = [m.get('emotion_at_time', {}).get('sadness', 0.3) for m in client.memory]
        hope = [m.get('emotion_at_time', {}).get('hope', 0.4) for m in client.memory]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=anxiety, name='불안', line=dict(color='#EF553B')))
    fig.add_trace(go.Scatter(x=x, y=sadness, name='슬픔', line=dict(color='#636EFA')))
    fig.add_trace(go.Scatter(x=x, y=hope, name='희망', line=dict(color='#00CC96')))
    
    fig.update_layout(
        title=f"{client_name} 감정 변화 추이",
        xaxis_title="상담 횟수",
        yaxis_title="감정 수준",
        yaxis=dict(range=[0, 1]),
        height=350
    )
    
    return fig


def create_all_clients_comparison():
    """모든 내담자 비교 차트"""
    data = []
    for name, client in clients.items():
        data.append({
            'name': name,
            '불안': client.emotional_state.get('anxiety', 0),
            '슬픔': client.emotional_state.get('sadness', 0),
            '희망': client.emotional_state.get('hope', 0),
            '신뢰': client.emotional_state.get('trust', 0),
            '에너지': client.energy,
            '개방성': client.openness
        })
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('불안', '슬픔', '희망', '신뢰', '에너지', '개방성'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#EF553B', '#636EFA', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    metrics = ['불안', '슬픔', '희망', '신뢰', '에너지', '개방성']
    for i, metric in enumerate(metrics):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Bar(x=df['name'], y=df[metric], name=metric, marker_color=colors[i]),
            row=row, col=col
        )
    
    fig.update_layout(height=500, showlegend=False, title_text="내담자 비교 분석")
    
    return fig


def create_counselor_action_probabilities():
    """상담사 행동 확률 차트"""
    if current_client_name is None or not current_session_active:
        categories = ["공감", "질문", "반영", "해석", "지시", "정보제공"]
        probs = [1/6] * 6
    else:
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
        
        probs_dict = counselor.get_action_probabilities(state)
        categories = list(probs_dict.keys())
        probs = list(probs_dict.values())
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=probs, marker_color='#636EFA')
    ])
    
    fig.update_layout(
        title="상담사 행동 확률 분포",
        xaxis_title="행동 유형",
        yaxis_title="확률",
        yaxis=dict(range=[0, 1]),
        height=350
    )
    
    return fig


def create_session_history_chart():
    """세션 이력 차트"""
    if not monitoring_data:
        return go.Figure()
    
    timestamps = [d['timestamp'] for d in monitoring_data[-20:]]
    session_active = [1 if d['session_active'] else 0 for d in monitoring_data[-20:]]
    
    fig = go.Figure(data=[
        go.Scatter(x=timestamps, y=session_active, mode='lines+markers', name='세션 상태')
    ])
    
    fig.update_layout(
        title="세션 활성 상태 추이",
        xaxis_title="시간",
        yaxis_title="활성 (1) / 비활성 (0)",
        yaxis=dict(range=[-0.1, 1.1]),
        height=300
    )
    
    return fig


# ============================================================
# 텍스트 모니터링 함수
# ============================================================

def get_system_status():
    """시스템 상태 텍스트"""
    data = collect_monitoring_data()
    monitoring_data.append(data)
    
    status_lines = [
        "═══════════════════════════════════════════",
        "         🧠 심리상담 클리닉 모니터링",
        "═══════════════════════════════════════════",
        "",
        f"📊 시스템 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"👥 등록 내담자: {len(clients)}명",
        f"🤖 상담사: {data['counselor_stats']['name']}",
        f"📈 학습 횟수: {data['counselor_stats']['episodes']}",
        f"💰 총 보상: {data['counselor_stats']['total_reward']:.2f}",
        "",
        "───────────────────────────────────────────",
        "         📋 내담자 상태",
        "───────────────────────────────────────────"
    ]
    
    for name, client in clients.items():
        status_lines.extend([
            "",
            f"  👤 {name} ({client.age}세, {client.gender})",
            f"     문제: {', '.join(client.issues)}",
            f"     불안: {client.emotional_state['anxiety']:.2f} | 슬픔: {client.emotional_state['sadness']:.2f}",
            f"     희망: {client.emotional_state['hope']:.2f} | 신뢰: {client.emotional_state['trust']:.2f}",
            f"     에너지: {client.energy:.2f} | 개방성: {client.openness:.2f}",
            f"     세션: {client.session_count}회 | 기억: {len(client.memory)}개"
        ])
    
    status_lines.extend([
        "",
        "───────────────────────────────────────────",
        "         🎯 현재 세션",
        "───────────────────────────────────────────"
    ])
    
    if current_session_active and current_client_name:
        client = clients[current_client_name]
        status_lines.extend([
            f"  상태: 활성",
            f"  내담자: {current_client_name}",
            f"  단계: {clinic_env.current_session.phase.value if clinic_env.current_session else 'N/A'}",
            f"  대화 수: {len(clinic_env.current_session.dialogue_history) if clinic_env.current_session else 0}"
        ])
    else:
        status_lines.append("  상태: 비활성")
    
    return "\n".join(status_lines)


def get_client_detail(client_name: str):
    """내담자 상세 정보"""
    if client_name not in clients:
        return "내담자를 선택해주세요."
    
    client = clients[client_name]
    
    detail_lines = [
        f"═══════════════════════════════════════════",
        f"         📋 {client_name} 상세 정보",
        f"═══════════════════════════════════════════",
        "",
        f"기본 정보:",
        f"  이름: {client.name}",
        f"  나이: {client.age}세",
        f"  성별: {client.gender}",
        f"  문제: {', '.join(client.issues)}",
        "",
        f"감정 상태:",
        f"  불안: {client.emotional_state['anxiety']:.2f}",
        f"  슬픔: {client.emotional_state['sadness']:.2f}",
        f"  분노: {client.emotional_state.get('anger', 0):.2f}",
        f"  희망: {client.emotional_state['hope']:.2f}",
        f"  신뢰: {client.emotional_state['trust']:.2f}",
        "",
        f"기타:",
        f"  에너지: {client.energy:.2f}",
        f"  개방성: {client.openness:.2f}",
        f"  세션 횟수: {client.session_count}",
        f"  기억 수: {len(client.memory)}"
    ]
    
    if client.memory:
        detail_lines.extend(["", "최근 기억:"])
        for mem in client.memory[-5:]:
            detail_lines.append(f"  [{mem.session_id}세션] {mem.content[:50]}...")
    
    return "\n".join(detail_lines)


def get_conversation_log():
    """대화 로그"""
    if not current_session_active or current_client_name is None:
        return "활성 세션이 없습니다."
    
    session = clinic_env.current_session
    if not session:
        return "세션 데이터가 없습니다."
    
    log_lines = [
        f"═══════════════════════════════════════════",
        f"         💬 대화 로그 ({current_client_name})",
        f"═══════════════════════════════════════════",
        ""
    ]
    
    for msg in session.dialogue_history:
        role = "상담사" if msg['role'] == 'counselor' else "내담자"
        category = f" [{msg.get('category', '')}]" if msg.get('category') else ""
        log_lines.append(f"  {role}{category}: {msg['content']}")
    
    return "\n".join(log_lines)


# ============================================================
# 학습 모니터링
# ============================================================

def get_learning_status():
    """학습 상태"""
    stats = counselor.get_statistics()
    
    status_lines = [
        "═══════════════════════════════════════════",
        "         🎓 학습 상태",
        "═══════════════════════════════════════════",
        "",
        f"상담사: {stats['name']}",
        f"총 학습 횟수: {stats['episodes']}",
        f"총 보상: {stats['total_reward']:.2f}",
        f"평균 보상: {stats['avg_reward']:.3f}",
        "",
        "───────────────────────────────────────────",
        "         📊 행동 통계",
        "───────────────────────────────────────────"
    ]
    
    if current_session_active and current_client_name:
        client = clients[current_client_name]
        state = counselor.get_state_vector(
            emotional_state=client.emotional_state,
            energy=client.energy,
            openness=client.openness,
            phase=clinic_env.current_session.phase.value if clinic_env.current_session else '탐색',
            dialogue_count=len(clinic_env.current_session.dialogue_history) if clinic_env.current_session else 0
        )
        
        probs = counselor.get_action_probabilities(state)
        for action, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            status_lines.append(f"  {action}: {prob:.3f} {bar}")
    
    return "\n".join(status_lines)


# ============================================================
# Gradio 앱
# ============================================================

def create_monitoring_app():
    """모니터링 대시보드 앱"""
    
    with gr.Blocks(
        title="🧠 심리상담 클리닉 모니터링",
        theme=gr.themes.Soft(),
        css="""
        .status-box {
            background-color: #f0f8ff;
            border: 1px solid #4a90d9;
            border-radius: 8px;
            padding: 15px;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🧠 심리상담 클리닉 모니터링 대시보드
        
        실시간으로 상담 세션과 에이전트 상태를 모니터링합니다.
        """)
        
        with gr.Tabs():
            
            with gr.Tab("📊 대시보드"):
                gr.Markdown("### 시스템 종합 대시보드")
                
                refresh_btn = gr.Button("🔄 새로고침", variant="primary", size="lg")
                
                with gr.Row():
                    system_status = gr.Textbox(
                        label="시스템 상태",
                        lines=30,
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                
                with gr.Row():
                    emotion_radar = gr.Plot(label="감정 레이더 차트")
                    action_probs = gr.Plot(label="행동 확률")
                
                client_selector = gr.Dropdown(
                    choices=list(clients.keys()),
                    label="내담자 선택",
                    value=list(clients.keys())[0] if clients else None
                )
                
                def refresh_dashboard(client_name):
                    return (
                        get_system_status(),
                        create_emotion_radar_chart(client_name),
                        create_counselor_action_probabilities()
                    )
                
                refresh_btn.click(
                    fn=refresh_dashboard,
                    inputs=client_selector,
                    outputs=[system_status, emotion_radar, action_probs]
                )
            
            with gr.Tab("📈 차트"):
                gr.Markdown("### 상세 차트")
                
                chart_client = gr.Dropdown(
                    choices=list(clients.keys()),
                    label="내담자 선택",
                    value=list(clients.keys())[0] if clients else None
                )
                
                with gr.Row():
                    emotion_timeline = gr.Plot(label="감정 변화 추이")
                    all_clients_chart = gr.Plot(label="내담자 비교")
                
                session_chart = gr.Plot(label="세션 이력")
                
                def update_charts(client_name):
                    return (
                        create_emotion_timeline_chart(client_name),
                        create_all_clients_comparison(),
                        create_session_history_chart()
                    )
                
                chart_client.change(
                    fn=update_charts,
                    inputs=chart_client,
                    outputs=[emotion_timeline, all_clients_chart, session_chart]
                )
            
            with gr.Tab("👤 내담자"):
                gr.Markdown("### 내담자 상세 정보")
                
                detail_client = gr.Dropdown(
                    choices=list(clients.keys()),
                    label="내담자 선택",
                    value=list(clients.keys())[0] if clients else None
                )
                
                detail_btn = gr.Button("📋 상세 정보 보기", variant="secondary")
                client_detail = gr.Textbox(
                    label="내담자 상세",
                    lines=25,
                    interactive=False
                )
                
                detail_btn.click(
                    fn=get_client_detail,
                    inputs=detail_client,
                    outputs=client_detail
                )
            
            with gr.Tab("💬 대화"):
                gr.Markdown("### 실시간 대화 로그")
                
                log_refresh = gr.Button("🔄 로그 새로고침", variant="secondary")
                conversation_log = gr.Textbox(
                    label="대화 로그",
                    lines=25,
                    interactive=False
                )
                
                log_refresh.click(
                    fn=get_conversation_log,
                    outputs=conversation_log
                )
            
            with gr.Tab("🎓 학습"):
                gr.Markdown("### 학습 상태")
                
                learn_refresh = gr.Button("🔄 학습 상태 새로고침", variant="secondary")
                learning_status = gr.Textbox(
                    label="학습 상태",
                    lines=20,
                    interactive=False
                )
                
                learn_refresh.click(
                    fn=get_learning_status,
                    outputs=learning_status
                )
        
        gr.Markdown("""
        ---
        ### 📖 모니터링 가이드
        
        1. **대시보드**: 전체 시스템 상태 한눈에 보기
        2. **차트**: 감정 변화, 내담자 비교 시각화
        3. **내담자**: 개별 내담자 상세 정보
        4. **대화**: 실시간 대화 로그 추적
        5. **학습**: 상담사 학습 진행 상황
        
        **새로고침**: 각 탭에서 버튼을 클릭하여 최신 상태로 업데이트
        """)
    
    return app


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("모니터링 대시보드 시작...")
    
    app = create_monitoring_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_error=True
    )
