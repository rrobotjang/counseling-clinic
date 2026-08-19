"""
상담 데이터베이스 관리

SQLite를 사용하여 상담 데이터를 저장하고 관리합니다:
1. 상담 QA 데이터 저장
2. 세션 이력 저장
3. 감정 상태 기록
4. 유사 맥락 검색
"""

import sqlite3
import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class CounselingDatabase:
    """상담 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'counseling.db')
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        self._create_tables()
    
    def _create_tables(self):
        """테이블 생성"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                label INTEGER NOT NULL,
                emotions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_name TEXT NOT NULL,
                counselor_name TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                improvement_score REAL DEFAULT 0.0,
                final_emotions TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS dialogue_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                emotions TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS counselor_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                counselor_name TEXT NOT NULL,
                episode INTEGER NOT NULL,
                reward REAL NOT NULL,
                strategy_used TEXT,
                client_state TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def load_qa_data(self, csv_path: str):
        """CSV에서 QA 데이터 로드"""
        if not os.path.exists(csv_path):
            print(f"파일을 찾을 수 없습니다: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        emotion_keywords = {
            '이별': ['이별', '헤어지', '끊'],
            '우울': ['우울', '슬프', '힘들', '고통'],
            '불안': ['불안', '걱정', '무서'],
            '스트레스': ['스트레스', '지치', '피곤'],
            '외로움': ['외롭', '혼자', '쓸쓸'],
            '화남': ['화나', '짜증', '분노'],
            '기쁨': ['기쁘', '행복', '좋은'],
            '사랑': ['사랑', '좋아해', '연애']
        }
        
        inserted = 0
        for _, row in df.iterrows():
            q = str(row['Q']).strip()
            a = str(row['A']).strip()
            label = int(row['label'])
            
            if not q or not a:
                continue
            
            detected_emotions = []
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in q.lower():
                        detected_emotions.append(emotion)
                        break
            
            emotions_json = json.dumps(detected_emotions)
            
            self.cursor.execute('''
                INSERT INTO qa_pairs (question, answer, label, emotions)
                VALUES (?, ?, ?, ?)
            ''', (q, a, label, emotions_json))
            
            inserted += 1
        
        self.conn.commit()
        print(f"✅ {inserted}개 QA 데이터 로드 완료")
    
    def search_similar(self, text: str, limit: int = 5) -> List[Dict]:
        """유사한 QA 검색"""
        self.cursor.execute('''
            SELECT id, question, answer, label, emotions
            FROM qa_pairs
            WHERE question LIKE ? OR answer LIKE ?
            LIMIT ?
        ''', (f'%{text}%', f'%{text}%', limit))
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'label': row[3],
                'emotions': json.loads(row[4]) if row[4] else []
            })
        
        return results
    
    def search_by_emotions(self, emotions: List[str], limit: int = 5) -> List[Dict]:
        """감정 기반 검색"""
        if not emotions:
            return []
        
        conditions = []
        params = []
        for emotion in emotions:
            conditions.append("emotions LIKE ?")
            params.append(f'%{emotion}%')
        
        where_clause = " OR ".join(conditions)
        params.append(limit)
        
        self.cursor.execute(f'''
            SELECT id, question, answer, label, emotions
            FROM qa_pairs
            WHERE {where_clause}
            ORDER BY RANDOM()
            LIMIT ?
        ''', params)
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'label': row[3],
                'emotions': json.loads(row[4]) if row[4] else []
            })
        
        return results
    
    def get_best_response(self, user_input: str, counseling_style: str = "공감") -> Dict:
        """최적 응답 검색"""
        detected_emotions = self._detect_emotions(user_input)
        
        results = self.search_similar(user_input, limit=10)
        
        if not results:
            results = self.search_by_emotions(detected_emotions, limit=10)
        
        if not results:
            self.cursor.execute('''
                SELECT id, question, answer, label, emotions
                FROM qa_pairs
                WHERE label = 1
                ORDER BY RANDOM()
                LIMIT 1
            ''')
            row = self.cursor.fetchone()
            if row:
                results = [{
                    'id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'label': row[3],
                    'emotions': json.loads(row[4]) if row[4] else []
                }]
        
        if results:
            best = results[0]
            response = self._adapt_response(best['answer'], counseling_style)
            return {
                'response': response,
                'source_id': best['id'],
                'similarity': self._calculate_similarity(user_input, best['question']),
                'detected_emotions': detected_emotions
            }
        
        return {
            'response': "계속 말씀해 주세요.",
            'source_id': None,
            'similarity': 0.0,
            'detected_emotions': detected_emotions
        }
    
    def _detect_emotions(self, text: str) -> List[str]:
        """감정 감지"""
        emotion_keywords = {
            '이별': ['이별', '헤어지', '끊'],
            '우울': ['우울', '슬프', '힘들', '고통'],
            '불안': ['불안', '걱정', '무서'],
            '스트레스': ['스트레스', '지치', '피곤'],
            '외로움': ['외롭', '혼자', '쓸쓸'],
            '화남': ['화나', '짜증', '분노'],
            '기쁨': ['기쁘', '행복', '좋은'],
            '사랑': ['사랑', '좋아해', '연애']
        }
        
        detected = []
        text_lower = text.lower()
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(emotion)
                    break
        
        return detected
    
    def _adapt_response(self, base_response: str, style: str) -> str:
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
        import random
        prefix = random.choice(prefixes)
        
        return prefix + base_response
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """간단한 유사도 계산"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def save_session(self, client_name: str, counselor_name: str, 
                     messages: List[Dict], final_emotions: Dict) -> int:
        """세션 저장"""
        self.cursor.execute('''
            INSERT INTO sessions (client_name, counselor_name, total_messages, final_emotions)
            VALUES (?, ?, ?, ?)
        ''', (client_name, counselor_name, len(messages), json.dumps(final_emotions)))
        
        session_id = self.cursor.lastrowid
        
        for msg in messages:
            self.cursor.execute('''
                INSERT INTO dialogue_history (session_id, role, content, category, emotions)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, msg['role'], msg['content'], 
                  msg.get('category'), json.dumps(msg.get('emotions', []))))
        
        self.conn.commit()
        return session_id
    
    def save_learning(self, counselor_name: str, episode: int, 
                     reward: float, strategy: str, client_state: Dict):
        """학습 이력 저장"""
        self.cursor.execute('''
            INSERT INTO counselor_learning 
            (counselor_name, episode, reward, strategy_used, client_state)
            VALUES (?, ?, ?, ?, ?)
        ''', (counselor_name, episode, reward, strategy, json.dumps(client_state)))
        
        self.conn.commit()
    
    def get_statistics(self) -> Dict:
        """데이터베이스 통계"""
        self.cursor.execute('SELECT COUNT(*) FROM qa_pairs')
        qa_count = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM sessions')
        session_count = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM dialogue_history')
        dialogue_count = self.cursor.fetchone()[0]
        
        return {
            'qa_pairs': qa_count,
            'sessions': session_count,
            'dialogues': dialogue_count
        }
    
    def close(self):
        """연결 종료"""
        self.conn.close()


def init_database(csv_path: str = None, db_path: str = None) -> CounselingDatabase:
    """데이터베이스 초기화"""
    db = CounselingDatabase(db_path)
    
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'ChatbotData.csv')
    
    # 이미 데이터가 로드되어 있으면 스킵
    db.cursor.execute('SELECT COUNT(*) FROM qa_pairs')
    existing_count = db.cursor.fetchone()[0]
    
    if existing_count > 0:
        print(f"✅ 기존 QA 데이터 {existing_count}개 로드됨")
        return db
    
    if os.path.exists(csv_path):
        db.load_qa_data(csv_path)
    
    return db
