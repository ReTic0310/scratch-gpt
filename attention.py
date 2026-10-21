import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torchtune.modules import KVCache

class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.kv_cache = None
        self.head_dim = embedding_dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=max_seq_len
        )
        

    def forward(self, x: torch.Tensor, *, use_cache: bool = False):
        B, T = x.shape[:2]

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}."
            )

        if use_cache:
            if self.kv_cache is None:
                raise RuntimeError("KV Cache is not initialized. Call setup_cache() first.")
            if self.kv_cache.size + T > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {self.kv_cache.size + T} "
                    f"exceeds max_seq_len {self.max_seq_len}."
                )
            if B != self.kv_cache.batch_size:
                raise ValueError(f"Batch size mismatch: expected {self.kv_cache.batch_size}, got {B}.")

            input_pos = torch.arange(self.kv_cache.size, self.kv_cache.size + T, device=x.device)
        else:
            input_pos = None

        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, -1)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        q = self.rope(q, input_pos=input_pos)
        k = self.rope(k, input_pos=input_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = True

        if use_cache:
            assert self.kv_cache is not None

            if self.kv_cache.size > 0:
                if T != 1:
                    raise ValueError(
                        "Decode requires exactly one token."
                    )
                is_causal = False


            k, v = self.kv_cache.update(k, v)

            k = k[:, :, :self.kv_cache.size, :]
            v = v[:, :, :self.kv_cache.size, :]

        sdpa = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=0.0
        )

        attn_output = sdpa.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)

        return self.output_proj(attn_output)

    def setup_cache(self, batch_size: int, dtype: torch.dtype):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        
        if self.kv_cache is not None:
            raise RuntimeError("KV Cache is already initialized.")
        
        
        self.kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=self.max_seq_len,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=dtype
        ).to(self.qkv_proj.weight.device)

    def reset_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()
