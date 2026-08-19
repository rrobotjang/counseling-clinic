# -*- coding: utf-8 -*-
"""
학습 스크립트
사용법:
  python train.py                     # 기본 설정으로 전체 학습
  python train.py --epochs 5 --quick  # 빠른 스모크 테스트 (데이터 일부만 사용)

학습이 끝나면 서빙에 필요한 아래 파일들이 checkpoints/ 폴더에 저장됩니다.
  - checkpoints/spm_korean.model / .vocab   (SentencePiece 토크나이저)
  - checkpoints/chatbot_transformer.pt      (모델 가중치 + 설정)
"""

import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data import (
    download_chatbot_data,
    load_qa_pairs,
    train_sentencepiece,
    ChatbotDataset,
)
from model import Transformer


def get_lr_lambda(d_model, warmup_steps=4000):
    d_model = float(d_model)

    def lr_lambda(step):
        step = step + 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return lr_lambda


def accuracy_function(y_pred, y_true, pad_id=0):
    preds = y_pred.argmax(dim=-1)
    mask = y_true != pad_id
    correct = (preds == y_true) & mask
    return correct.float().sum() / mask.float().sum()


def train_step(model, batch, optimizer, loss_function, device):
    model.train()
    enc_input, dec_input, target = [x.to(device) for x in batch]
    optimizer.zero_grad()
    logits = model(enc_input, dec_input)
    loss = loss_function(logits.permute(0, 2, 1), target)
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy_function(logits, target).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--units", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--quick", action="store_true", help="샘플 데이터 일부로 빠른 스모크 테스트")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 1) 데이터 준비
    csv_path = download_chatbot_data("ChatbotData.csv")
    pairs = load_qa_pairs(csv_path)
    if args.quick:
        pairs = pairs[:200]
    print(f"QA 쌍 수: {len(pairs)}")

    # 2) SentencePiece 학습
    spm_prefix = os.path.join(args.out_dir, "spm_korean")
    vocab_size = min(args.vocab_size, 4000) if args.quick else args.vocab_size
    sp = train_sentencepiece(pairs, model_prefix=spm_prefix, vocab_size=vocab_size)
    print(f"Vocab size: {sp.GetPieceSize()}")

    # 3) Dataset / DataLoader
    dataset = ChatbotDataset(pairs, sp, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4) 모델
    model = Transformer(
        vocab_size=sp.GetPieceSize(),
        num_layers=args.num_layers,
        units=args.units,
        d_model=args.d_model,
        num_heads=args.num_heads,
        max_length=args.max_length,
        dropout=args.dropout,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_lambda(args.d_model, warmup_steps=args.warmup_steps)
    )

    # 5) 학습 루프
    history = {"loss": [], "acc": []}
    for epoch in range(args.epochs):
        total_loss, total_acc = 0.0, 0.0
        for step, batch in enumerate(dataloader):
            loss, acc = train_step(model, batch, optimizer, loss_function, device)
            total_loss += loss
            total_acc += acc
            scheduler.step()
            if step % 100 == 0:
                print(f"[Epoch {epoch + 1}, Step {step}] Loss: {loss:.4f}, Acc: {acc:.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)
        print(f"Epoch {epoch + 1} 완료 - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

    # 6) 체크포인트 저장 (서빙 앱이 그대로 읽을 수 있도록 config 포함)
    config = {
        "vocab_size": sp.GetPieceSize(),
        "num_layers": args.num_layers,
        "units": args.units,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "spm_model_path": spm_prefix + ".model",
    }
    ckpt_path = os.path.join(args.out_dir, "chatbot_transformer.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n체크포인트 저장 완료: {ckpt_path}")
    print(f"이제 'python app.py' 로 배포용 챗봇 웹앱을 실행할 수 있습니다.")


if __name__ == "__main__":
    main()
