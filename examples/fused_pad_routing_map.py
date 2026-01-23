#!/usr/bin/env python3
"""
Example: Pad routing map to multiple per expert.

Torch reference for Megatron's fused_pad_routing_map operator.
"""

import torch
import torch.nn as nn


def pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    input_map = routing_map.transpose(0, 1).contiguous().int()  # [num_experts, num_tokens]
    output_map = input_map.clone()

    for expert_idx in range(num_experts):
        row = input_map[expert_idx]
        num_ones = int(row.sum().item())
        remainder = num_ones % pad_multiple
        num_to_pad = pad_multiple - remainder if remainder != 0 else 0

        if num_to_pad > 0:
            is_zero = row == 0
            zero_ranks = torch.cumsum(is_zero.int(), dim=0)
            mask_to_flip = (zero_ranks <= num_to_pad) & is_zero
            output_map[expert_idx] = torch.where(mask_to_flip, torch.ones_like(row), row)

    return output_map.transpose(0, 1)


class Model(nn.Module):
    """Simple model applying pad_routing_map."""

    def __init__(self, pad_multiple=4):
        super().__init__()
        self.pad_multiple = pad_multiple

    def forward(self, routing_map):
        return pad_routing_map(routing_map, self.pad_multiple)


def get_inputs():
    """Generate test inputs."""
    num_tokens = 128
    num_experts = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    routing_map = torch.randint(0, 2, (num_tokens, num_experts), device=device, dtype=torch.int32)
    return (routing_map,)


def get_init_inputs():
    """Get initialization inputs for the model."""
    return {
        "pad_multiple": 4,
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
