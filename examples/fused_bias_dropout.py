#!/usr/bin/env python3
"""
Example: Bias + Dropout + Residual Add.

Torch reference for Megatron's fused bias-dropout-add operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bias_dropout_add(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, p: float, training: bool):
    if bias is not None:
        x = x + bias
    if residual.dtype != x.dtype:
        residual = residual.to(x.dtype)
    out = F.dropout(x, p=p, training=training)
    return out + residual


class Model(nn.Module):
    """Simple model with bias + dropout + residual add."""

    def __init__(self, hidden_size=1024, dropout_p=0.1, use_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None

    def forward(self, x, residual):
        return _bias_dropout_add(x, self.bias, residual, self.dropout_p, self.training)


def get_inputs():
    """Generate test inputs."""
    batch_size = 32
    hidden_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    residual = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)

    return (x, residual)


def get_init_inputs():
    """Get initialization inputs for the model."""
    return {
        "hidden_size": 1024,
        "dropout_p": 0.1,
        "use_bias": True,
    }


if __name__ == "__main__":
    model = Model(**get_init_inputs())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)

    print(f"Input shape: {inputs[0].shape}")
    print(f"Residual shape: {inputs[1].shape}")
    print(f"Output shape: {output.shape}")
    print("\n✅ Model test passed!")
