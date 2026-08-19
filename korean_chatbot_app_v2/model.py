# -*- coding: utf-8 -*-
"""
한국어 트랜스포머 챗봇 모델 정의
원본 노트북(한국어_트랜스포머_챗봇.ipynb)의 모델 구조를 그대로 모듈화한 것입니다.
train.py(학습)와 app.py(서빙) 양쪽에서 공통으로 import 해서 사용합니다.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 마스크
# ---------------------------------------------------------------------------
def create_padding_mask(x: torch.Tensor) -> torch.Tensor:
    mask = (x == 0).float()
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(1)
    look_ahead_mask = 1 - torch.tril(torch.ones((seq_len, seq_len), device=x.device))
    padding_mask = create_padding_mask(x)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)
    combined_mask = torch.max(look_ahead_mask, padding_mask)
    return combined_mask


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, position: int, d_model: int):
        super().__init__()
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
        return pos_encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding[:, : x.size(1), :].to(x.device)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------
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
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
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
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        return self.out_dense(concat_attention)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        attn_output = self.dropout1(self.mha(x, x, x, mask))
        out1 = self.norm1(x + attn_output)
        ffn_output = self.dropout2(self.ffn(out1))
        return self.norm2(out1 + ffn_output)


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, max_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(position=max_length, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.encdec_mha = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_outputs, look_ahead_mask=None, padding_mask=None):
        self_attn_out = self.dropout1(self.self_mha(x, x, x, mask=look_ahead_mask))
        out1 = self.norm1(x + self_attn_out)
        encdec_attn_out = self.dropout2(self.encdec_mha(out1, enc_outputs, enc_outputs, mask=padding_mask))
        out2 = self.norm2(out1 + encdec_attn_out)
        ffn_out = self.dropout3(self.ffn(out2))
        return self.norm3(out2 + ffn_out)


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, ff_dim, d_model, num_heads, max_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(position=max_length, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, enc_outputs, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, enc_outputs, look_ahead_mask, padding_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, max_length, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.encoder = Encoder(vocab_size, num_layers, units, d_model, num_heads, max_length, dropout)
        self.decoder = Decoder(vocab_size, num_layers, units, d_model, num_heads, max_length, dropout)
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, dec_inputs):
        enc_padding_mask = create_padding_mask(inputs)
        look_ahead_mask = create_look_ahead_mask(dec_inputs)
        dec_padding_mask = create_padding_mask(inputs)
        enc_outputs = self.encoder(inputs, mask=enc_padding_mask)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask)
        return self.final_linear(dec_outputs)
