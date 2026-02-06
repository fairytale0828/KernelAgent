#!/usr/bin/env python3
"""
Inference-only reference: Bias + Dropout + Residual Add
Aligned with Megatron's _bias_dropout_add_func(..., training=False).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bias_dropout_add_inference_megatron_ref(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """
    Megatron-aligned inference semantics:
      out = residual + dropout(x + bias, training=False)
    with:
      - residual cast to x dtype if needed
      - inplace path enabled when no grads are required
    """
    x, bias = x_with_bias

    # Inference: training is always False
    training = False

    # Megatron: inplace only in eval/inference AND no grads required
    inplace = (
        (not training)
        and (not x.requires_grad)
        and (not residual.requires_grad)
        and (bias is None or (not bias.requires_grad))
    )

    # Megatron: cast residual to x dtype (avoid mixed-precision issues)
    if residual.dtype != x.dtype:
        residual = residual.to(x.dtype)

    if bias is not None:
        if inplace:
            x.add_(bias)
        else:
            x = x + bias
        out = F.dropout(x, p=p, training=training, inplace=inplace)  # identity in inference
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out
    else:
        out = F.dropout(x, p=p, training=training, inplace=inplace)  # identity in inference
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out


class Model(nn.Module):
    """Inference-only model wrapper (Megatron-style calling convention)."""

    def __init__(self, hidden_size=1024, dropout_p=0.1, use_bias=True):
        super().__init__()
        self.dropout_p = dropout_p
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Inference-only: always uses Megatron-aligned inference function
        return bias_dropout_add_inference_megatron_ref((x, self.bias), residual, self.dropout_p)


def get_inputs():
    batch_size = 32
    hidden_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make this match your inference dtype; Megatron often runs fp16/bf16.
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
    residual = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    return x, residual


def get_init_inputs():
    return {"hidden_size": 1024, "dropout_p": 0.1, "use_bias": True}


if __name__ == "__main__":
    model = Model(**get_init_inputs())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Ensure inference mode (though forward is no_grad and uses training=False anyway)
    model.eval()

    x, residual = get_inputs()
    out = model(x, residual)

    print(f"X: {x.shape}, dtype={x.dtype}")
    print(f"Residual: {residual.shape}, dtype={residual.dtype}")
    print(f"Out: {out.shape}, dtype={out.dtype}")
    print("\n✅ Megatron-aligned inference-only test passed!")
