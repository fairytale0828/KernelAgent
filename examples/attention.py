import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple attention model:
      - projects input to Q, K, V
      - computes scaled dot-product attention
      - applies softmax over the last dimension
      - returns attended output

    Input:  x [batch_size, seq_len, embed_dim]
    Output: y [batch_size, seq_len, embed_dim]
    """
    def __init__(self, embed_dim, num_heads=8, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # QKV projections + output projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        B, S, E = x.shape  # batch, seq_len, embed_dim

        # Project to Q, K, V: [B, S, E]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: [B, H, S, D]
        H = self.num_heads
        D = self.head_dim
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        # Scaled dot-product attention:
        # scores: [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # Softmax over last dim (keys dimension)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum: [B, H, S, D]
        ctx = torch.matmul(attn, v)

        # Merge heads back: [B, S, E]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, E)

        # Output projection
        out = self.out_proj(ctx)
        return out


# Example shapes (match the "get_inputs" style)
batch_size = 16
seq_len = 1024
embed_dim = 1024
num_heads = 8

def get_inputs():
    return [torch.rand(batch_size, seq_len, embed_dim)]

def get_init_inputs():
    return [embed_dim, num_heads]