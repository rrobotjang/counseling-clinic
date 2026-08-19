# -*- coding: utf-8 -*-
"""
배포용 웹앱 (Gradio)
사용법:
  python app.py
  -> 로컬에서 http://127.0.0.1:7860 으로 접속해서 챗봇과 대화

Hugging Face Spaces 등에 그대로 올려서 배포할 수도 있습니다.
(이 경우 requirements.txt 와 checkpoints/ 폴더를 함께 업로드하면 됩니다)
"""

import os
import torch
import gradio as gr

from model import Transformer
from data import preprocess_sentence, load_sentencepiece

CKPT_PATH = os.environ.get("CHATBOT_CKPT", "checkpoints/chatbot_transformer.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            f"먼저 'python train.py' 를 실행해서 모델을 학습/저장해주세요."
        )
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    config = checkpoint["config"]

    sp = load_sentencepiece(config["spm_model_path"])

    model = Transformer(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        units=config["units"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        max_length=config["max_length"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, sp, config["max_length"]


print("모델 로딩 중...")
model, sp, MAX_LENGTH = load_model_and_tokenizer(CKPT_PATH)
print("모델 로딩 완료. 디바이스:", DEVICE)


@torch.no_grad()
def decoder_inference(sentence: str) -> list:
    sentence = preprocess_sentence(sentence)
    enc_input_ids = [sp.bos_id()] + sp.encode(sentence) + [sp.eos_id()]
    enc_input = torch.tensor([enc_input_ids], dtype=torch.long, device=DEVICE)
    dec_input = torch.tensor([[sp.bos_id()]], dtype=torch.long, device=DEVICE)

    for _ in range(MAX_LENGTH):
        predictions = model(enc_input, dec_input)
        predicted_id = torch.argmax(predictions[:, -1:, :], dim=-1)
        if predicted_id.item() == sp.eos_id():
            break
        dec_input = torch.cat([dec_input, predicted_id], dim=-1)

    return dec_input.squeeze(0).tolist()


def chatbot_reply(message: str, history=None) -> str:
    if not message or not message.strip():
        return "문장을 입력해주세요."
    output_seq = decoder_inference(message)
    reply = sp.decode(
        [t for t in output_seq if t not in (sp.bos_id(), sp.eos_id(), sp.pad_id())]
    )
    return reply if reply.strip() else "음... 뭐라고 답해야 할지 모르겠어요."


demo = gr.ChatInterface(
    fn=chatbot_reply,
    title="한국어 트랜스포머 챗봇",
    description=(
        "Transformer(Encoder-Decoder) 구조로 학습한 한국어 챗봇입니다. "
        "songys/Chatbot_data로 학습되었습니다."
    ),
    examples=["안녕?", "오늘 기분이 좋아", "심심해", "밥 먹었어?", "고마워"],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
