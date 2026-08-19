# -*- coding: utf-8 -*-
"""데이터 다운로드, 전처리, SentencePiece 학습, Dataset 정의"""

import os
import re
import requests
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

CHATBOT_DATA_URL = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"


def download_chatbot_data(local_path: str = "ChatbotData.csv") -> str:
    if os.path.exists(local_path):
        return local_path
    response = requests.get(CHATBOT_DATA_URL, timeout=30)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def preprocess_sentence(sentence: str) -> str:
    """
    한국어 문장 전처리:
    1. 양쪽 공백 제거
    2. 단어와 구두점 사이에 공백 추가
    3. 중복 공백 제거
    4. 한국어(가-힣), 숫자, 구두점만 유지
    """
    sentence = sentence.strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r"[\s]+", " ", sentence)
    sentence = re.sub(r"[^가-힣0-9?.!,\s]+", " ", sentence)
    sentence = re.sub(r"[\s]+", " ", sentence)  # 특수문자 제거 후 생긴 중복 공백 재정리
    sentence = sentence.strip()
    return sentence


def load_qa_pairs(csv_path: str):
    df = pd.read_csv(csv_path)
    pairs = list(zip(df["Q"].tolist(), df["A"].tolist()))
    processed_pairs = []
    for q, a in pairs:
        q_processed = preprocess_sentence(str(q))
        a_processed = preprocess_sentence(str(a))
        if q_processed and a_processed:
            processed_pairs.append((q_processed, a_processed))
    return processed_pairs


def train_sentencepiece(processed_pairs, corpus_file="korean_corpus.txt",
                         model_prefix="spm_korean", vocab_size=8000):
    with open(corpus_file, "w", encoding="utf-8") as f:
        for q, a in processed_pairs:
            f.write(q + "\n")
            f.write(a + "\n")

    # 코퍼스가 작으면 SentencePiece가 요청한 vocab_size만큼 subword를 못 만들어
    # "Vocabulary size too high" 에러가 나므로, 데이터 규모에 맞춰 자동으로 낮춰준다.
    approx_chars = sum(len(q) + len(a) for q, a in processed_pairs)
    safe_vocab_size = min(vocab_size, max(100, approx_chars // 2))
    if safe_vocab_size < vocab_size:
        print(f"[안내] 코퍼스 크기 대비 vocab_size를 {vocab_size} -> {safe_vocab_size} 로 조정합니다.")
        vocab_size = safe_vocab_size

    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        max_sentence_length=999999,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(model_prefix + ".model")
    return sp


def load_sentencepiece(model_path: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


class ChatbotDataset(Dataset):
    def __init__(self, pairs, sp, max_length=40):
        super().__init__()
        self.sp = sp
        self.max_length = max_length
        self.data = []

        for q_text, a_text in pairs:
            q_ids = sp.EncodeAsIds(q_text)
            a_ids = sp.EncodeAsIds(a_text)
            q_ids = [sp.bos_id()] + q_ids + [sp.eos_id()]
            a_ids = [sp.bos_id()] + a_ids + [sp.eos_id()]
            if len(q_ids) > max_length:
                q_ids = q_ids[:max_length]
            if len(a_ids) > max_length:
                a_ids = a_ids[:max_length]
            q_ids = q_ids + [sp.pad_id()] * (max_length - len(q_ids))
            a_ids = a_ids + [sp.pad_id()] * (max_length - len(a_ids))
            decoder_input = a_ids[:-1]
            decoder_label = a_ids[1:]
            self.data.append(
                (
                    torch.tensor(q_ids, dtype=torch.long),
                    torch.tensor(decoder_input, dtype=torch.long),
                    torch.tensor(decoder_label, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
