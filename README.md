# 🤖 한국어 트랜스포머 챗봇

## 빠른 시작

### 1. 학습 (선택사항)
이미 학습된 모델이 있는 경우 건너뜁니다.

```bash
# Jupyter에서 실행
jupyter notebook 한국어_트랜스포머_챗봇.ipynb
```

### 2. 앱 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 앱 실행
python app.py
```

### 3. 접속

- 로컬: http://localhost:7860
- 외부 접속 시 `share=True`로 공유 링크 생성 가능

## 파일 구조

```
Ex08/
├── app.py                          # Gradio 챗봇 앱
├── 한국어_트랜스포머_챗봇.ipynb    # 학습 노트북
├── requirements.txt                # 의존성
├── data/
│   └── ChatbotData.csv            # 한국어 챗봇 데이터
├── spm_korean.model               # SentencePiece 모델
└── korean_chatbot.pth             # 학습된 모델 가중치
```

## 모델 정보

- **아키텍처**: Transformer (Encoder-Decoder)
- **파라미터**: ~8M
- **학습 데이터**: songys/Chatbot_data (11,823 QA 쌍)
- **토크나이저**: SentencePiece BPE (vocab 8,000)
- **하이퍼파라미터**:
  - d_model: 256
  - num_heads: 8
  - num_layers: 2
  - ff_dim: 512
