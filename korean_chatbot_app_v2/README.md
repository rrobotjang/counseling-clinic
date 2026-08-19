# 한국어 트랜스포머 챗봇 - 배포용 앱

> **✅ 검증 완료**: `checkpoints/chatbot_transformer.pt`(원본 `chatbot_model.pth`를 재포장)는
> `songys/Chatbot_data` 코퍼스로 재생성한 `checkpoints/spm_korean.model` 토크나이저와 함께
> 정상적으로 자연스러운 한국어 답변을 생성하는 것을 확인했습니다.
> (예: "안녕?" → "안녕하세요 .", "고마워" → "감사합니다 .")

`한국어_트랜스포머_챗봇.ipynb` 노트북의 학습 코드를 그대로 모듈화하고,
**학습(train.py)**과 **서빙(app.py, Gradio 웹앱)**을 분리해서 실제 배포 가능한 형태로 재구성했습니다.

## 파일 구성

```
korean_chatbot_app/
├── model.py         # Transformer 모델 정의 (PositionalEncoding, MultiHeadAttention, Encoder/Decoder)
├── data.py           # 데이터 다운로드 + 전처리 + SentencePiece + Dataset
├── train.py           # 학습 스크립트 -> checkpoints/ 에 모델 저장
├── app.py             # Gradio 배포용 챗봇 웹앱
├── requirements.txt
└── README.md
```

## 1. 설치

```bash
pip install -r requirements.txt
```

## 2. 학습

```bash
python train.py --epochs 25
```

- `songys/Chatbot_data`를 자동으로 다운로드하고, SentencePiece 토크나이저와
  Transformer 모델을 학습한 뒤 `checkpoints/` 폴더에 저장합니다.
  - `checkpoints/spm_korean.model` / `.vocab`
  - `checkpoints/chatbot_transformer.pt` (가중치 + 하이퍼파라미터 설정 포함)
- 우선 잘 도는지만 빠르게 확인하고 싶다면:
  ```bash
  python train.py --epochs 2 --quick
  ```
  (데이터 200개만 사용해서 1~2분 내로 끝남 — 품질보다 파이프라인 동작 확인용)

## 3. 배포(서빙) 실행

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:7860` 접속하면 채팅 UI가 뜹니다.

### 다른 환경(예: Hugging Face Spaces)에 배포하려면

1. `korean_chatbot_app/` 폴더 전체 + 학습된 `checkpoints/` 폴더를 그대로 업로드
2. `app.py`가 `checkpoints/chatbot_transformer.pt`를 자동으로 로드하도록 되어 있으므로
   별도 코드 수정 없이 Space에서 바로 실행 가능
3. `CHATBOT_CKPT` 환경변수로 체크포인트 경로를 바꿀 수도 있습니다:
   ```bash
   CHATBOT_CKPT=/path/to/other_checkpoint.pt python app.py
   ```

## 하이퍼파라미터 커스터마이즈

```bash
python train.py \
  --epochs 30 \
  --num_layers 3 \
  --d_model 256 \
  --num_heads 8 \
  --units 512 \
  --batch_size 64
```

## 주의

- `checkpoints/` 폴더가 없으면 `app.py` 실행 시 안내 메시지와 함께 종료됩니다 — 먼저 `train.py`를 실행하세요.
- GPU가 있으면 자동으로 사용합니다(`torch.cuda.is_available()`).
