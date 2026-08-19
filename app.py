#!/usr/bin/env python3
"""
한국어 트랜스포머 챗봇 - Gradio 앱
사용법: python app.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import re
import os

# ============================================================
# 하이퍼파라미터 (학습 시와 동일하게 설정)
# ============================================================
MAX_LENGTH = 40
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

# ============================================================
# 모델 아키텍처 (학습 노트북과 동일)
# ============================================================

def create_padding_mask(x):
    mask = (x == 0).float()
    mask = mask.unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(x):
    seq_len = x.size(1)
    look_ahead_mask = 1 - torch.tril(torch.ones((seq_len, seq_len)))
    padding_mask = create_padding_mask(x)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)
    look_ahead_mask = look_ahead_mask.to(x.device)
    combined_mask = torch.max(look_ahead_mask, padding_mask)
    return combined_mask

class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.position = position
        self.pos_encoding = self._build_pos_encoding(position, d_model)

    def _get_angles(self, position, i, d_model):
        return 1.0 / (10000.0 ** ((2.0 * (i // 2)) / d_model)) * position

    def _build_pos_encoding(self, position, d_model):
        pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle_rads = self._get_angles(pos, i, d_model)
        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])
        pos_encoding = torch.zeros(position, d_model)
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :].to(x.device)

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-1, -2))
    depth = key.size(-1)
    logits = matmul_qk / math.sqrt(depth)
    if mask is not None:
        logits = logits + (mask * -1e9)
    attention_weights = F.softmax(logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        self.out_dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.out_dense(concat_attention)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        return out2

class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(position=MAX_LENGTH, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.encdec_mha = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_outputs, look_ahead_mask=None, padding_mask=None):
        self_attn_out = self.self_mha(x, x, x, mask=look_ahead_mask)
        self_attn_out = self.dropout1(self_attn_out)
        out1 = self.norm1(x + self_attn_out)
        encdec_attn_out = self.encdec_mha(out1, enc_outputs, enc_outputs, mask=padding_mask)
        encdec_attn_out = self.dropout2(encdec_attn_out)
        out2 = self.norm2(out1 + encdec_attn_out)
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(out2 + ffn_out)
        return out3

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(position=MAX_LENGTH, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_outputs, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, enc_outputs, look_ahead_mask, padding_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size, num_layers=num_layers,
            ff_dim=units, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.decoder = Decoder(
            vocab_size=vocab_size, num_layers=num_layers,
            ff_dim=units, d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, dec_inputs):
        enc_padding_mask = create_padding_mask(inputs)
        look_ahead_mask = create_look_ahead_mask(dec_inputs)
        dec_padding_mask = create_padding_mask(inputs)
        enc_outputs = self.encoder(x=inputs, mask=enc_padding_mask)
        dec_outputs = self.decoder(
            x=dec_inputs, enc_outputs=enc_outputs,
            look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask
        )
        logits = self.final_linear(dec_outputs)
        return logits

# ============================================================
# 한국어 전처리
# ============================================================

def preprocess_sentence(sentence):
    """한국어 문장 전처리"""
    sentence = sentence.strip()
    sentence = re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence = re.sub(r'[\s]+', ' ', sentence)
    sentence = re.sub(r'[^가-힣0-9?.!,\s]+', ' ', sentence)
    sentence = sentence.strip()
    return sentence

# ============================================================
# 모델/토크나이저 로드
# ============================================================

def load_model_and_tokenizer(model_dir='.'):
    """학습된 모델과 토크나이저 로드"""
    # SentencePiece 로드
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(model_dir, 'spm_korean.model'))
    
    VOCAB_SIZE = sp.GetPieceSize()
    
    # 모델 생성
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )
    
    # 가중치 로드
    checkpoint_path = os.path.join(model_dir, 'korean_chatbot.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        print(f"✅ 모델 로드 완료: {checkpoint_path}")
    else:
        print(f"⚠️ 체크포인트 없음: {checkpoint_path}")
        print("   새 모델로 시작합니다.")
    
    model.eval()
    return model, sp

# ============================================================
# 추론 함수
# ============================================================

def decoder_inference(model, sentence, tokenizer, device='cpu'):
    """디코더 추론"""
    sentence = preprocess_sentence(sentence)
    enc_input_ids = [tokenizer.bos_id()] + tokenizer.encode(sentence) + [tokenizer.eos_id()]
    enc_input = torch.tensor([enc_input_ids], dtype=torch.long, device=device)
    
    dec_input = torch.tensor([[tokenizer.bos_id()]], dtype=torch.long, device=device)
    
    for i in range(MAX_LENGTH):
        predictions = model(enc_input, dec_input)
        predicted_id = torch.argmax(predictions[:, -1:, :], dim=-1)
        
        if predicted_id.item() == tokenizer.eos_id():
            break
        
        dec_input = torch.cat([dec_input, predicted_id], dim=-1)
    
    return dec_input.squeeze().tolist()

def chatbot_response(model, sentence, tokenizer, device='cpu'):
    """챗봇 응답 생성"""
    model.eval()
    output_seq = decoder_inference(model, sentence, tokenizer, device)
    predicted_sentence = tokenizer.decode(
        [t for t in output_seq if t != tokenizer.bos_id() and t != tokenizer.eos_id() and t != tokenizer.pad_id()]
    )
    return predicted_sentence

# ============================================================
# Gradio 앱
# ============================================================

def create_gradio_app():
    """Gradio 챗봇 앱 생성"""
    import gradio as gr
    
    # 모델 로드
    print("🤖 한국어 챗봇 모델 로딩 중...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, sp = load_model_and_tokenizer(model_dir='.')
    model = model.to(device)
    print(f"✅ 모델 로드 완료 (디바이스: {device})")
    
    def respond(message, history):
        """Gradio 챗봇 응답 함수"""
        if not message.strip():
            return ""
        
        response = chatbot_response(model, message, sp, device)
        return response
    
    # Gradio 인터페이스
    with gr.Blocks(
        title="🇰🇷 한국어 트랜스포머 챗봇",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🤖 한국어 트랜스포머 챗봇
        
        한국어 데이터로 학습된 트랜스포머 모델 기반 챗봇입니다.
        
        **사용법**: 아래 입력창에 한국어 문장을 입력하세요!
        """)
        
        chatbot = gr.ChatInterface(
            fn=respond,
            title="",
            examples=["안녕?", "오늘 기분이 어때?", "날씨가 좋아", "감사합니다"],
            retry_btn="🔄 다시 시도",
            undo_btn="↩️ 취소",
            clear_btn="🗑️ 전체 삭제"
        )
    
    return demo

# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # True로 설정하면 공유 링크 생성
        show_error=True
    )
