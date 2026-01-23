#!/usr/bin/env python3
"""
Example: Scale + Mask + Softmax (causal or additive).

Torch reference for Megatron's fused softmax operator.
"""

import torch
import torch.nn as nn


def make_causal_mask(sq: int, sk: int, device, dtype):
    mask = torch.full((sq, sk), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def apply_additive_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x + mask


class Model(nn.Module):
    """Scaled masked softmax reference implementation."""

    def __init__(self, scale=1.0, causal=True, softmax_in_fp32=True):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.softmax_in_fp32 = softmax_in_fp32

    def forward(self, attn_scores: torch.Tensor, mask: torch.Tensor = None):
        # attn_scores: [b, np, sq, sk]
        assert attn_scores.dim() == 4

        orig_dtype = attn_scores.dtype
        x = attn_scores
        if self.softmax_in_fp32 and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        if self.scale is not None:
            x = x * self.scale

        if self.causal and mask is None:
            sq, sk = x.size(2), x.size(3)
            causal_mask = make_causal_mask(sq, sk, x.device, x.dtype)
            mask = causal_mask.view(1, 1, sq, sk)

        if mask is not None:
            x = apply_additive_mask(x, mask)

        probs = torch.softmax(x, dim=-1)

        if self.softmax_in_fp32 and orig_dtype in (torch.float16, torch.bfloat16):
            probs = probs.to(orig_dtype)

        return probs


def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    num_heads = 4
    sq = 128
    sk = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attn_scores = torch.randn(batch_size, num_heads, sq, sk, device=device, dtype=torch.float16)
    return (attn_scores, None)


def get_init_inputs():
    """Get initialization inputs for the model."""
    return {
        "scale": 1.0 / (128 ** 0.5),
        "causal": True,
        "softmax_in_fp32": True,
    }


if __name__ == "__main__":
    model = Model(**get_init_inputs())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)

    print(f"Input shape: {inputs[0].shape}")
    print(f"Output shape: {output.shape}")
    print("\n✅ Model test passed!")
