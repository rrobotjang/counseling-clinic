"""
상담 API 클라이언트

에이전트가 API 서버와 통신하여 실제 응답을 생성합니다:
1. 실시간 상담 응답 생성
2. 세션 관리
3. 학습 이력 전송
4. 유사 맥락 검색
"""

import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API 설정"""
    base_url: str = "http://localhost:8000"
    timeout: int = 10


class CounselingAPIClient:
    """상담 API 클라이언트"""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """API 요청"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=self.config.timeout)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=self.config.timeout)
            else:
                raise ValueError(f"지원하지 않는 메서드: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            return {"error": "API 서버에 연결할 수 없습니다."}
        except requests.exceptions.Timeout:
            return {"error": "요청 시간이 초과되었습니다."}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def counsel(self, user_message: str, counseling_style: str = "공감",
                client_emotions: Dict[str, float] = None) -> Dict:
        """
        상담 응답 생성
        
        Args:
            user_message: 사용자 메시지
            counseling_style: 상담 스타일
            client_emotions: 내담자 감정 상태
        
        Returns:
            응답 데이터
        """
        data = {
            "user_message": user_message,
            "counseling_style": counseling_style,
            "client_emotions": client_emotions
        }
        
        result = self._make_request("POST", "/api/counsel", data)
        
        if "error" in result:
            return self._get_fallback_response(user_message, counseling_style)
        
        return result
    
    def search_similar(self, user_message: str, limit: int = 5) -> List[Dict]:
        """유사 맥락 검색"""
        data = {"user_message": user_message, "limit": limit}
        result = self._make_request("POST", "/api/search", data)
        
        if "error" in result:
            return []
        
        return result.get("results", [])
    
    def start_session(self, client_name: str, counselor_name: str = "김상담사") -> Dict:
        """세션 시작"""
        data = {
            "client_name": client_name,
            "counselor_name": counselor_name
        }
        
        result = self._make_request("POST", "/api/session/start", data)
        
        if "error" in result:
            return {"session_id": -1, "message": "세션 시작 실패"}
        
        return result
    
    def end_session(self, session_id: int, final_emotions: Dict[str, float]) -> Dict:
        """세션 종료"""
        data = {
            "session_id": session_id,
            "final_emotions": final_emotions
        }
        
        result = self._make_request("POST", "/api/session/end", data)
        
        if "error" in result:
            return {"message": "세션 종료 실패"}
        
        return result
    
    def save_learning(self, counselor_name: str, episode: int, 
                     reward: float, strategy: str, client_state: Dict) -> Dict:
        """학습 이력 저장"""
        data = {
            "counselor_name": counselor_name,
            "episode": episode,
            "reward": reward,
            "strategy": strategy,
            "client_state": client_state
        }
        
        result = self._make_request("POST", "/api/learning", data)
        
        if "error" in result:
            return {"message": "학습 이력 저장 실패"}
        
        return result
    
    def get_statistics(self) -> Dict:
        """통계 조회"""
        result = self._make_request("GET", "/api/stats")
        
        if "error" in result:
            return {"qa_pairs": 0, "sessions": 0, "dialogues": 0}
        
        return result
    
    def _get_fallback_response(self, user_message: str, style: str) -> Dict:
        """대체 응답 (API 실패 시)"""
        fallback_responses = {
            "공감": "많이 힘드셨군요. 더 자세히 말씀해 주세요.",
            "질문": "그렇게 느끼셨군요. 좀 더 자세히 말씀해 주실 수 있을까요?",
            "반영": "지금 말씀하시는 것을 보면, 정말 중요한 문제인 것 같아요.",
            "해석": "그렇게 볼 수도 있겠네요. 다른 관점도 있을 수 있어요.",
            "지시": "이런 방법은 어떨까요? 작은 것부터 시작해보세요.",
            "정보제공": "그런 경우에는 여러 가지 방법이 있을 수 있어요."
        }
        
        return {
            "response": fallback_responses.get(style, "계속 말씀해 주세요."),
            "source_id": None,
            "similarity": 0.0,
            "detected_emotions": [],
            "counseling_style": style,
            "reward": 0.3
        }


class OfflineCounselingClient:
    """
    오프라인 상담 클라이언트
    
    API 서버 없이 로컬에서 동작
    """
    
    def __init__(self, db_path: str = None):
        from village.database import CounselingDatabase
        
        if db_path is None:
            import os
            db_path = os.path.join(os.path.dirname(__file__), 'counseling.db')
        
        self.db = CounselingDatabase(db_path)
    
    def counsel(self, user_message: str, counseling_style: str = "공감",
                client_emotions: Dict[str, float] = None) -> Dict:
        """상담 응답 생성 (로컬)"""
        result = self.db.get_best_response(user_message, counseling_style)
        
        return {
            "response": result['response'],
            "source_id": result['source_id'],
            "similarity": result['similarity'],
            "detected_emotions": result['detected_emotions'],
            "counseling_style": counseling_style,
            "reward": 0.3
        }
    
    def search_similar(self, user_message: str, limit: int = 5) -> List[Dict]:
        """유사 맥락 검색 (로컬)"""
        return self.db.search_similar(user_message, limit)
    
    def get_statistics(self) -> Dict:
        """통계 조회 (로컬)"""
        return self.db.get_statistics()
    
    def close(self):
        """연결 종료"""
        self.db.close()


def create_client(mode: str = "auto", api_url: str = None) -> object:
    """
    클라이언트 생성
    
    Args:
        mode: "api", "offline", "auto"
        api_url: API 서버 URL
    
    Returns:
        클라이언트 인스턴스
    """
    if mode == "api":
        config = APIConfig(base_url=api_url or "http://localhost:8000")
        return CounselingAPIClient(config)
    
    elif mode == "offline":
        return OfflineCounselingClient()
    
    else:
        try:
            config = APIConfig(base_url=api_url or "http://localhost:8000")
            client = CounselingAPIClient(config)
            client.get_statistics()
            return client
        except Exception:
            return OfflineCounselingClient()
