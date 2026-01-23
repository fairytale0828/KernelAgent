#!/usr/bin/env python3
"""
Example: Bias + GeLU (tanh approximation).

Torch reference for Megatron's fused bias-gelu operator.
"""

import torch
import torch.nn as nn


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class Model(nn.Module):
    """Simple model with bias + GeLU."""

    def __init__(self, hidden_size=1024, use_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if use_bias else None

    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias
        return gelu_tanh(x)


def get_inputs():
    """Generate test inputs."""
    batch_size = 32
    hidden_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    return (x,)


def get_init_inputs():
    """Get initialization inputs for the model."""
    return {
        "hidden_size": 1024,
        "use_bias": True,
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
