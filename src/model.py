import torch
from typing import Optional
from torch import Tensor
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование.

    Добавляет векторное представление позиции к входным эмбеддингам,
    используя функции синуса и косинуса разной частоты.

    Args:
        d_model (int): размерность эмбеддингов.
        max_len (int): максимальная длина последовательности.

    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *\
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x
    
class MultiHeadAttention(nn.Module):
    """
    Многоголовое внимание со скалярным произведением.

    Проецирует запрос (q), ключи (k) и значения (v) в несколько голов внимания,
    применяет масштабируемый механизм внимания независимо к каждой голове
    и объединяет результаты.

    Args:
        d_model (int): размерность входных и выходных эмбеддингов.
        n_heads (int): количество голов внимания.
        dropout (float): вероятность отключения нейронов.
    
    Shape:
        q, k, v: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, seq_len, seq_len) или None
        output: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _attention(self, 
                   q: Tensor, 
                   k: Tensor, 
                   v: Tensor, 
                   mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        x, _ = self._attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        return self.out(x)

class EncoderLayer(nn.Module):
    """
    Слой энкодера для трансформера.

    Состоит из двух подслоев:
    1. Многоголовое внимание.
    2. Полносвязная нейронная сеть (FFN).

    Args:
        d_model (int): размерность входных и выходных эмбеддингов.
        n_heads (int): количество голов внимания.
        d_ff (int): размерность скрытого слоя FFN.
        dropout (float): вероятность отключения нейронов.

    Shape:
        x (tgt): (batch_size, seq_len, d_model)
        mask: (batch_size, 1, seq_len, seq_len) или None
        output: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Self-Attention + Residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
    
class DecoderLayer(nn.Module):
    """
    Слой декодера для трансформера.

    Состоит из трёх подслоев:
    1. Многоголовое самвонимание
    2. Многоголовое внимание к выходу энкодера
    3. Полносвязная нейронная сеть (FFN)

    Args:
        d_model (int): размерность входных и выходных эмбеддингов.
        n_heads (int): количество голов внимания.
        d_ff (int): размерность скрытого слоя FFN.
        dropout (float): вероятность отключения нейронов.
    
    Shape:
        x (tgt): (batch_size, seq_len, d_model)
        enc_out: (batch_size, src_seq_len, d_model)
        src_mask: (batch_size, 1, src_seq_len, src_seq_len) или None
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) или None
        output: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: Tensor,
                enc_out: Tensor,
                src_mask: Optional[Tensor],
                tgt_mask: Optional[Tensor]) -> Tensor:
        # Masked Self-Attention
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Encoder-Decoder Attention
        enc_attn_out = self.enc_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(enc_attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
    
class MiniTransformer(nn.Module):
    """
    Мини-трансформер для seq2seq задач.

    Архитектура состоит из энкодера, декодера и выходного линейного слоя.

    Args:
        vocab_size (int): размер словаря.
        d_model (int): размерность эмбеддингов.
        n_heads (int): количество голов внимания.
        n_layers (int): количество слоёв энкодера и декодера.
        d_ff (int): размерность скрытого слоя FFN.
        max_len (int): максимальная длина последовательности.
        dropout (float): вероятность отключения нейронов.
    
    Shape:
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        src_mask: (batch_size, 1, src_seq_len, src_seq_len) или None
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) или None
        output: (batch_size, tgt_seq_len, vocab_size)
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 128, 
                 n_heads: int = 4, 
                 n_layers: int = 2, 
                 d_ff: int = 512, 
                 max_len: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        
    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, 
               tgt: Tensor, 
               enc_out: Tensor,
               src_mask: Optional[Tensor] = None, 
               tgt_mask: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.out(x)

    def forward(self, src: Tensor, 
                tgt: Tensor, 
                src_mask: Optional[Tensor] = None, 
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_mask)