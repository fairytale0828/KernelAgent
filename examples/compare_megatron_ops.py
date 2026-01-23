#!/usr/bin/env python3
"""
Compare Torch reference implementations with local Torch examples.
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _rng_state(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    return cpu_state, cuda_state


def _set_rng_state(device: torch.device, state: Tuple[torch.ByteTensor, torch.ByteTensor]):
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _compare_tensors(name, a, b, rtol, atol):
    close = torch.allclose(a, b, rtol=rtol, atol=atol)
    max_abs = (a - b).abs().max().item()
    print(f"[{name}] allclose={close} rtol={rtol} atol={atol} max_abs={max_abs:.6g}")
    return close


def _bias_dropout_add_ref(x, bias, residual, p, training):
    if bias is not None:
        x = x + bias
    if residual.dtype != x.dtype:
        residual = residual.to(x.dtype)
    out = F.dropout(x, p=p, training=training)
    return out + residual


def _gelu_tanh(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def _pad_routing_map_ref(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    input_map = routing_map.transpose(0, 1).contiguous().int()
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


def _softmax_ref(attn_scores: torch.Tensor, scale: float, causal: bool):
    orig_dtype = attn_scores.dtype
    x = attn_scores
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    if scale is not None:
        x = x * scale

    if causal:
        sq, sk = x.size(2), x.size(3)
        mask = torch.full((sq, sk), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1).view(1, 1, sq, sk)
        x = x + mask

    probs = torch.softmax(x, dim=-1)

    if orig_dtype in (torch.float16, torch.bfloat16):
        probs = probs.to(orig_dtype)

    return probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tolerances
    rtol_fp16 = 1e-2
    atol_fp16 = 1e-2
    rtol_fp32 = 1e-4
    atol_fp32 = 1e-4

    print(f"Using device: {device}")

    # Bias + Dropout + Residual Add
    try:
        import fused_bias_dropout as example_bias_dropout

        batch_size = 32
        hidden_size = 1024
        p = 0.1
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
        residual = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
        bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)

        state = _rng_state(device)
        _set_rng_state(device, state)
        ref_out = _bias_dropout_add_ref(x, bias, residual, p, True)
        _set_rng_state(device, state)
        ex_out = example_bias_dropout._bias_dropout_add(x, bias, residual, p, True)
        _compare_tensors("bias_dropout_add/train", ref_out, ex_out, rtol_fp32, atol_fp32)

        state = _rng_state(device)
        _set_rng_state(device, state)
        ref_out = _bias_dropout_add_ref(x, bias, residual, p, False)
        _set_rng_state(device, state)
        ex_out = example_bias_dropout._bias_dropout_add(x, bias, residual, p, False)
        _compare_tensors("bias_dropout_add/eval", ref_out, ex_out, rtol_fp32, atol_fp32)
    except Exception as exc:
        print(f"[bias_dropout_add] skipped: {exc}")

    # Bias + GeLU
    try:
        import fused_bias_gelu as example_bias_gelu

        batch_size = 32
        hidden_size = 1024
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
        bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)

        ref_out = _gelu_tanh(x + bias)
        ex_out = example_bias_gelu.gelu_tanh(x + bias)
        _compare_tensors("bias_gelu", ref_out, ex_out, rtol_fp32, atol_fp32)
    except Exception as exc:
        print(f"[bias_gelu] skipped: {exc}")

    # Pad routing map
    try:
        import fused_pad_routing_map as example_pad

        num_tokens = 128
        num_experts = 8
        pad_multiple = 4
        routing_map = torch.randint(
            0, 2, (num_tokens, num_experts), device=device, dtype=torch.int32
        )

        ref_out = _pad_routing_map_ref(routing_map, pad_multiple)
        ex_out = example_pad.pad_routing_map(routing_map, pad_multiple)
        _compare_tensors("pad_routing_map", ref_out, ex_out, rtol_fp32, atol_fp32)
    except Exception as exc:
        print(f"[pad_routing_map] skipped: {exc}")

    # Fused softmax
    try:
        import fused_softmax as example_softmax

        batch_size = 2
        num_heads = 4
        sq = 128
        sk = 128
        scale = 1.0 / math.sqrt(sk)
        attn_scores = torch.randn(
            batch_size, num_heads, sq, sk, device=device, dtype=torch.float16
        )

        ref_out = _softmax_ref(attn_scores, scale=scale, causal=True)

        model = example_softmax.Model(scale=scale, causal=True, softmax_in_fp32=True).to(device)
        ex_out = model(attn_scores, None)
        _compare_tensors("softmax/causal", ref_out, ex_out, rtol_fp16, atol_fp16)
    except Exception as exc:
        print(f"[softmax] skipped: {exc}")


if __name__ == "__main__":
    main()
