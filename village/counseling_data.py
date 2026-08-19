"""
상담 데이터 기반 응답 시스템

ChatbotData.csv의 라벨 1 (감정 관련) 데이터를 활용하여
상담 맥락에 맞는 응답 생성
"""

import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import re


class CounselingResponseSystem:
    """
    상담 데이터 기반 응답 시스템
    
    역할:
    1. 상담 데이터 로드 및 분석
    2. 맥락별 적절한 응답 선택
    3. 강화학습과 연동
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'ChatbotData.csv')
        
        self.data_path = data_path
        self.qa_pairs: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        self.emotion_keywords: Dict[str, List[str]] = {
            '이별': ['이별', '헤어지', '끊', '⌓分开'],
            '우울': ['우울', '슬프', '힘들', '고통', '아프'],
            '불안': ['불안', '걱정', '걱정', '怖い', '무서'],
            '스트레스': ['스트레스', '지치', '피곤', '힘든'],
            '외로움': ['외롭', '혼자', '쓸쓸', '왕따'],
            '화남': ['화나', '짜증', '분노', 'angry'],
            '기쁨': ['기쁘', '행복', '좋은', '신나'],
            '사랑': ['사랑', '좋아해', '연애', '♥']
        }
        
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if not os.path.exists(self.data_path):
            print(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
            return
        
        df = pd.read_csv(self.data_path)
        
        for _, row in df.iterrows():
            q = str(row['Q']).strip()
            a = str(row['A']).strip()
            label = int(row['label'])
            
            if q and a:
                self.qa_pairs[label].append((q, a))
        
        print(f"✅ 상담 데이터 로드 완료:")
        print(f"   - 라벨 0 (일상): {len(self.qa_pairs[0])}개")
        print(f"   - 라벨 1 (감정/상담): {len(self.qa_pairs[1])}개")
        print(f"   - 라벨 2 (기타): {len(self.qa_pairs[2])}개")
    
    def detect_emotion(self, text: str) -> List[str]:
        """텍스트에서 감정 키워드 감지"""
        detected = []
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(emotion)
                    break
        
        return detected
    
    def find_similar_context(self, user_input: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        사용자 입력과 유사한 컨텍스트 찾기
        
        Returns:
            List of (question, answer, similarity_score)
        """
        user_emotions = self.detect_emotion(user_input)
        
        candidates = []
        
        for label in [1, 0, 2]:
            for q, a in self.qa_pairs[label]:
                q_emotions = self.detect_emotion(q)
                
                if user_emotions and q_emotions:
                    overlap = len(set(user_emotions) & set(q_emotions))
                    if overlap > 0:
                        score = overlap / max(len(user_emotions), len(q_emotions))
                        candidates.append((q, a, score))
        
        if not candidates:
            for label in [1, 0, 2]:
                for q, a in self.qa_pairs[label][:100]:
                    candidates.append((q, a, 0.1))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates[:top_k]
    
    def generate_response(self, user_input: str, counseling_style: str = "공감") -> str:
        """
        상담 스타일에 따른 응답 생성
        
        Args:
            user_input: 사용자 입력
            counseling_style: 상담 스타일 (공감, 질문, 반영, 해석, 지시, 정보제공)
        
        Returns:
            생성된 응답
        """
        similar = self.find_similar_context(user_input)
        
        if not similar:
            return self._get_default_response(counseling_style)
        
        best_match = similar[0]
        
        response = best_match[1]
        
        response = self._adapt_response_style(response, counseling_style)
        
        return response
    
    def _adapt_response_style(self, base_response: str, style: str) -> str:
        """응답 스타일 조정"""
        style_prefixes = {
            "공감": ["그렇군요. ", "이해합니다. ", "정말 힘드셨겠네요. "],
            "질문": ["그렇게 느끼셨군요. ", ""],
            "반영": ["말씀하신 것을 보면, ", ""],
            "해석": ["제 생각에는, ", "그렇게 볼 수도 있겠네요. "],
            "지시": ["이렇게 해보시는 것은 어떨까요? ", ""],
            "정보제공": ["사실 ", "알려진 바로는 "]
        }
        
        prefixes = style_prefixes.get(style, [""])
        prefix = random.choice(prefixes)
        
        return prefix + base_response
    
    def _get_default_response(self, style: str) -> str:
        """기본 응답"""
        defaults = {
            "공감": "많이 힘드셨군요. 더 자세히 말씀해 주세요.",
            "질문": "그렇게 느끼셨군요. 좀 더 자세히 말씀해 주실 수 있을까요?",
            "반영": "지금 말씀하시는 것을 보면, 정말 중요한 문제인 것 같아요.",
            "해석": "그렇게 볼 수도 있겠네요. 다른 관점도 있을 수 있어요.",
            "지시": "이런 방법은 어떨까요? 작은 것부터 시작해보세요.",
            "정보제공": "그런 경우에는 여러 가지 방법이 있을 수 있어요."
        }
        return defaults.get(style, "계속 말씀해 주세요.")
    
    def get_counseling_data(self, limit: int = 100) -> List[Dict]:
        """상담 데이터 (라벨 1) 반환"""
        data = []
        for q, a in self.qa_pairs[1][:limit]:
            data.append({
                'question': q,
                'answer': a,
                'emotions': self.detect_emotion(q)
            })
        return data
    
    def get_statistics(self) -> Dict:
        """데이터 통계"""
        return {
            'total_pairs': sum(len(v) for v in self.qa_pairs.values()),
            'daily_conversations': len(self.qa_pairs[0]),
            'counseling_data': len(self.qa_pairs[1]),
            'other_data': len(self.qa_pairs[2]),
            'emotion_categories': list(self.emotion_keywords.keys())
        }


class CounselingSession:
    """상담 세션 관리"""
    
    def __init__(self, response_system: CounselingResponseSystem):
        self.response_system = response_system
        self.history: List[Dict] = []
        self.emotional_state: Dict[str, float] = {
            'anxiety': 0.5,
            'sadness': 0.4,
            'hope': 0.3,
            'trust': 0.2
        }
        self.session_count: int = 0
    
    def start_session(self):
        """세션 시작"""
        self.session_count += 1
        self.history = []
        self.emotional_state = {
            'anxiety': 0.5,
            'sadness': 0.4,
            'hope': 0.3,
            'trust': 0.2
        }
    
    def process_message(self, user_input: str, counseling_style: str) -> Dict:
        """
        메시지 처리
        
        Returns:
            {
                'response': str,
                'detected_emotions': List[str],
                'emotional_state': Dict[str, float],
                'similar_contexts': List[Tuple[str, str, float]]
            }
        """
        detected_emotions = self.response_system.detect_emotion(user_input)
        
        response = self.response_system.generate_response(user_input, counseling_style)
        
        similar = self.response_system.find_similar_context(user_input)
        
        self._update_emotional_state(detected_emotions, counseling_style)
        
        self.history.append({
            'user': user_input,
            'response': response,
            'style': counseling_style,
            'emotions': detected_emotions,
            'emotional_state': self.emotional_state.copy()
        })
        
        return {
            'response': response,
            'detected_emotions': detected_emotions,
            'emotional_state': self.emotional_state.copy(),
            'similar_contexts': similar
        }
    
    def _update_emotional_state(self, detected_emotions: List[str], style: str):
        """감정 상태 업데이트"""
        if '이별' in detected_emotions or '우울' in detected_emotions:
            self.emotional_state['sadness'] = min(1.0, self.emotional_state['sadness'] + 0.1)
            self.emotional_state['anxiety'] = min(1.0, self.emotional_state['anxiety'] + 0.05)
        
        if '불안' in detected_emotions:
            self.emotional_state['anxiety'] = min(1.0, self.emotional_state['anxiety'] + 0.1)
        
        if '기쁨' in detected_emotions:
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.1)
            self.emotional_state['sadness'] = max(0.0, self.emotional_state['sadness'] - 0.1)
        
        if style == "공감":
            self.emotional_state['trust'] = min(1.0, self.emotional_state['trust'] + 0.05)
            self.emotional_state['sadness'] = max(0.0, self.emotional_state['sadness'] - 0.05)
        
        if style == "반영":
            self.emotional_state['trust'] = min(1.0, self.emotional_state['trust'] + 0.08)
        
        if style == "지시":
            self.emotional_state['hope'] = min(1.0, self.emotional_state['hope'] + 0.05)
    
    def get_session_summary(self) -> Dict:
        """세션 요약"""
        return {
            'session_count': self.session_count,
            'total_messages': len(self.history),
            'final_emotional_state': self.emotional_state.copy(),
            'styles_used': list(set([h['style'] for h in self.history])),
            'emotions_detected': list(set([e for h in self.history for e in h['emotions']]))
        }


def create_counseling_system(data_path: str = None) -> Tuple[CounselingResponseSystem, CounselingSession]:
    """상담 시스템 생성"""
    response_system = CounselingResponseSystem(data_path)
    session = CounselingSession(response_system)
    return response_system, session
