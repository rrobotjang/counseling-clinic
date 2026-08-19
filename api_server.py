"""
상담 응답 생성 API 서버

FastAPI 기반 백엔드:
1. 실시간 상담 응답 생성
2. 세션 관리
3. 학습 이력 저장
4. 유사 맥락 검색
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from village.database import CounselingDatabase, init_database
from village.counselor_agent import CounselorAgent


app = FastAPI(
    title="심리상담 클리닉 API",
    description="강화학습 상담사와 상담 데이터 기반 응답 생성 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db: Optional[CounselingDatabase] = None
counselor: Optional[CounselorAgent] = None


class CounselRequest(BaseModel):
    user_message: str
    counseling_style: str = "공감"
    client_emotions: Optional[Dict[str, float]] = None
    session_id: Optional[int] = None


class CounselResponse(BaseModel):
    response: str
    source_id: Optional[int]
    similarity: float
    detected_emotions: List[str]
    counseling_style: str
    reward: float


class SessionStartRequest(BaseModel):
    client_name: str
    counselor_name: str = "김상담사"


class SessionEndRequest(BaseModel):
    session_id: int
    final_emotions: Dict[str, float]


class LearningRequest(BaseModel):
    counselor_name: str
    episode: int
    reward: float
    strategy: str
    client_state: Dict


@app.on_event("startup")
async def startup():
    global db, counselor
    
    db_path = os.path.join(os.path.dirname(__file__), 'counseling.db')
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'ChatbotData.csv')
    
    db = init_database(csv_path, db_path)
    counselor = CounselorAgent(name="API상담사", state_dim=15)
    
    stats = db.get_statistics()
    print(f"✅ API 서버 시작")
    print(f"   QA 데이터: {stats['qa_pairs']}개")
    print(f"   세션 수: {stats['sessions']}개")


@app.get("/")
async def root():
    return {
        "message": "심리상담 클리닉 API",
        "version": "1.0.0",
        "endpoints": [
            "/api/counsel - 상담 응답 생성",
            "/api/search - 유사 맥락 검색",
            "/api/session/start - 세션 시작",
            "/api/session/end - 세션 종료",
            "/api/learning - 학습 이력 저장",
            "/api/stats - 통계 조회"
        ]
    }


@app.post("/api/counsel", response_model=CounselResponse)
async def counsel(request: CounselRequest):
    """상담 응답 생성"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    result = db.get_best_response(request.user_message, request.counseling_style)
    
    reward = 0.3
    if result['similarity'] > 0.5:
        reward = 0.5
    elif result['similarity'] > 0.3:
        reward = 0.4
    
    if request.counseling_style == "공감":
        reward += 0.1
    elif request.counseling_style == "반영":
        reward += 0.15
    
    return CounselResponse(
        response=result['response'],
        source_id=result['source_id'],
        similarity=result['similarity'],
        detected_emotions=result['detected_emotions'],
        counseling_style=request.counseling_style,
        reward=reward
    )


@app.post("/api/search")
async def search_similar(user_message: str, limit: int = 5):
    """유사 맥락 검색"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    results = db.search_similar(user_message, limit)
    return {"results": results, "count": len(results)}


@app.post("/api/session/start")
async def start_session(request: SessionStartRequest):
    """세션 시작"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    db.cursor.execute('''
        INSERT INTO sessions (client_name, counselor_name)
        VALUES (?, ?)
    ''', (request.client_name, request.counselor_name))
    db.conn.commit()
    
    session_id = db.cursor.lastrowid
    
    return {
        "session_id": session_id,
        "client_name": request.client_name,
        "counselor_name": request.counselor_name,
        "message": "세션이 시작되었습니다."
    }


@app.post("/api/session/end")
async def end_session(request: SessionEndRequest):
    """세션 종료"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    db.cursor.execute('''
        UPDATE sessions 
        SET end_time = CURRENT_TIMESTAMP, final_emotions = ?
        WHERE id = ?
    ''', (str(request.final_emotions), request.session_id))
    db.conn.commit()
    
    return {
        "session_id": request.session_id,
        "message": "세션이 종료되었습니다."
    }


@app.post("/api/learning")
async def save_learning(request: LearningRequest):
    """학습 이력 저장"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    db.save_learning(
        request.counselor_name,
        request.episode,
        request.reward,
        request.strategy,
        request.client_state
    )
    
    return {"message": "학습 이력이 저장되었습니다."}


@app.get("/api/stats")
async def get_stats():
    """통계 조회"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    stats = db.get_statistics()
    return stats


@app.get("/api/qa/{qa_id}")
async def get_qa(qa_id: int):
    """QA 데이터 조회"""
    if db is None:
        raise HTTPException(status_code=500, detail="데이터베이스가 초기화되지 않았습니다.")
    
    db.cursor.execute('''
        SELECT id, question, answer, label, emotions
        FROM qa_pairs
        WHERE id = ?
    ''', (qa_id,))
    
    row = db.cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="QA를 찾을 수 없습니다.")
    
    return {
        "id": row[0],
        "question": row[1],
        "answer": row[2],
        "label": row[3],
        "emotions": row[4]
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """서버 실행"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
