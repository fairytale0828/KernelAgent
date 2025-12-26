#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enhanced dispatcher with algorithmic fusion support.
Wraps the original dispatcher and adds Mirage-style problem descriptions.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import textwrap

# Import original dispatcher functions
from .dispatch_kernel_agent import (
    run as original_dispatch_run,
    _build_reference_code,
    _fmt_shape,
    _py_tuple,
    _pick_weights
)


def synthesize_problem_description_enhanced(item: Dict[str, Any]) -> str:
    """Enhanced problem description that handles algorithmic fusions."""
    
    # Check if this is an algorithmic fusion
    if item.get("type") == "algorithmic_fusion":
        return _synthesize_algorithmic_problem_description(item)
    
    # Otherwise use original logic (import from original file)
    # For now, implement a simplified version
    return _synthesize_standard_problem(item)


def _synthesize_standard_problem(item: Dict[str, Any]) -> str:
    """Standard subgraph problem description."""
    id_ = str(item.get("id", "unknown"))
    type_ = str(item.get("type", ""))
    layout = item.get("data_layout") or "NCHW"
    dtype = item.get("dtype") or "float32"
    input_shape = item.get("input_shape")
    output_shape = item.get("output_shape")
    inputs_multi = item.get("inputs")
    weights_fused = item.get("weights_fused")
    
    ref_code, _ = _build_reference_code(item)
    
    header = textwrap.dedent(
        f"""
        Implement a Triton kernel that computes the following subgraph end-to-end.

        Subgraph ID: {id_}
        Type: {type_}
        Data layout: {layout}
        DType: {dtype}

        Shapes:
        - input: {_fmt_shape(inputs_multi[0]) if isinstance(inputs_multi, list) else _fmt_shape(input_shape)}
        - output: {_fmt_shape(output_shape)}

        Weights (fused): {json.dumps(weights_fused, indent=2) if isinstance(weights_fused, dict) else "null"}

        Operations in order (with parameters):
        {json.dumps(item.get("ops", []), indent=2)}

        Requirements:
        - Return a complete Python file with a @triton.jit kernel and a wrapper function named kernel_function(...).
        - kernel_function must accept input tensor(s) and any required weights/bias parameters (match shapes above).
        - Implement the exact semantics of the listed ops in the given order for the provided shapes.
        - Use {layout} layout and {dtype} dtype semantics.

        Reference PyTorch implementation (exact semantics to match):
        """
    ).strip()
    
    problem = header + "\n\n```python\n" + ref_code + "```\n"
    return problem


def _synthesize_algorithmic_problem_description(item: Dict[str, Any]) -> str:
    """Generate Mirage-style problem description for algorithmic fusions."""
    
    algo_type = item.get("algorithm_type", "")
    
    if algo_type == "norm_gemm_fused":
        return _generate_norm_gemm_fusion_prompt(item)
    elif algo_type == "streaming_attention":
        return _generate_streaming_attention_prompt(item)
    elif algo_type == "gemm_custom_epilogue":
        return _generate_epilogue_fusion_prompt(item)
    else:
        # Fallback to standard description
        return _synthesize_standard_problem(item)


def _generate_norm_gemm_fusion_prompt(item: Dict) -> str:
    """Generate Norm+GEMM fusion detailed algorithm description."""
    metadata = item.get("algorithm_metadata", {})
    hints = item.get("generation_hints", {})
    norm_type = hints.get("norm_type", "rms_norm")
    
    # Extract shapes and weights
    input_shape = item.get("input_shape", [])
    output_shape = item.get("output_shape", [])
    weights = item.get("weights_fused", {})
    
    prompt = textwrap.dedent(f"""
    Implement a Triton kernel that fuses {norm_type.upper()} with GEMM using algorithmic fusion.
    
    **Algorithm Strategy (Mirage-style):**
    This is NOT a simple epilogue fusion. Instead, we fuse the normalization's reduction
    INTO the GEMM's main accumulation loop.
    
    **Key Algorithm Insight:**
    While computing Y = X @ W, we simultaneously compute the normalization statistics
    (mean, variance, or RMS) in the same pass over X. This eliminates the need to 
    materialize the normalized intermediate tensor.
    
    **Pseudo-algorithm for RMSNorm + GEMM:**
    ```python
    # For each output position (batch_idx, out_dim):
    for i in range(batch_size):
        for j in range(output_dim):
            acc = 0.0  # GEMM accumulator
            sum_sq = 0.0  # RMSNorm accumulator
            
            # Shared loop over input dimension!
            for k in range(input_dim):
                x_val = X[i, k]
                w_val = W[k, j]  # or W[j, k] depending on layout
                
                acc += x_val * w_val  # GEMM accumulation
                sum_sq += x_val * x_val  # RMS reduction
            
            # Epilogue: apply normalization
            rms = sqrt(sum_sq / input_dim + eps)
            gamma_val = gamma[k] if gamma exists else 1.0
            
            # Final output: normalized GEMM result
            output[i, j] = (acc / rms) * gamma_val
            
            # Optional: add bias
            if bias exists:
                output[i, j] += bias[j]
    ```
    
    **Implementation Requirements:**
    1. Single @triton.jit kernel (no separate norm kernel)
    2. Maintain both GEMM and norm accumulators in registers
    3. Apply normalization scale in epilogue
    4. Input shape: {input_shape}
    5. Output shape: {output_shape}
    6. Weights: {json.dumps(weights, indent=2)}
    7. Normalization type: {norm_type}
    8. Epsilon: 1e-5 (for numerical stability)
    
    **Triton Implementation Hints:**
    - Use BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K for tiling
    - Load X and W tiles in the K loop
    - Accumulate both GEMM result and sum_sq in the same loop
    - After K loop completes, compute RMS and apply to accumulated result
    - Store final output
    - Example block sizes: BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    
    **Performance Targets:**
    - Memory traffic: ~{metadata.get('expected_speedup', 1.5)}x reduction vs separate kernels
    - Avoid materializing intermediate normalized tensor (saves {metadata.get('memory_saved_bytes', 0)} bytes)
    
    **Reference:**
    This algorithm is inspired by Mirage's norm-gemm fusion and similar to how
    FlashAttention fuses reductions into main loops.
    
    **Test Requirements:**
    - Compare against PyTorch reference: (X / rms(X)) @ W
    - Use rtol=1e-3, atol=1e-3 for float32
    - Print "PASS" on success, exit with code 0
    
    Return a complete Python file with:
    1. @triton.jit kernel implementing the fused algorithm
    2. kernel_function(...) wrapper that handles grid launch
    3. Self-test that validates correctness against PyTorch
    """)
    
    return prompt


def _generate_streaming_attention_prompt(item: Dict) -> str:
    """Generate streaming Attention detailed algorithm description."""
    metadata = item.get("algorithm_metadata", {})
    block_size = metadata.get("block_size", 64)
    
    input_shapes = item.get("input_shape", {})
    output_shape = item.get("output_shape", [])
    
    prompt = textwrap.dedent(f"""
    Implement a Triton kernel for Streaming Attention (FlashAttention-style algorithm).
    
    **Algorithm Strategy:**
    Instead of materializing the full attention matrix S = softmax(QK^T), we process
    K and V in blocks and maintain running statistics for online softmax.
    
    **Key Algorithm (Block-wise K/V iteration):**
    ```python
    # Initialize
    O = zeros(...)  # Output accumulator
    m = -inf  # Running max
    l = 0.0  # Running sum
    
    for k_block in range(0, seq_len_k, BLOCK_SIZE):
        # Load K/V block
        K_block = load K[k_block:k_block+BLOCK_SIZE, :]
        V_block = load V[k_block:k_block+BLOCK_SIZE, :]
        
        # Compute attention scores for this block
        S_block = Q @ K_block.T  # [seq_q, BLOCK_SIZE]
        
        # Online softmax: update running max
        m_new = max(m, row_max(S_block))
        
        # Correction factor for previous blocks
        correction = exp(m - m_new)
        O = O * correction
        l = l * correction
        
        # Process current block
        P_block = exp(S_block - m_new)
        l = l + row_sum(P_block)
        O = O + P_block @ V_block
        
        m = m_new
    
    # Final normalization
    O = O / l
    ```
    
    **Implementation Requirements:**
    1. Block size: {block_size}
    2. Maintain running max (m) and sum (l) in registers
    3. Apply correction factor when max changes
    4. Never materialize full attention matrix
    5. Input shapes: {json.dumps(input_shapes, indent=2)}
    6. Output shape: {output_shape}
    
    **Memory Complexity:**
    - Traditional: O(seq_len^2)
    - This algorithm: O(block_size) = O({block_size})
    
    **Expected Performance:**
    - Memory traffic reduction: ~{metadata.get('expected_speedup', 3)}x
    - Enables longer sequences
    
    **Reference:**
    FlashAttention (Dao et al., 2022) and FlashAttention-2 (Dao, 2023)
    
    Return a complete Python file with kernel_function(Q, K, V) and self-test.
    """)
    
    return prompt


def _generate_epilogue_fusion_prompt(item: Dict) -> str:
    """Generate GEMM with epilogue fusion prompt."""
    metadata = item.get("algorithm_metadata", {})
    epilogue_ops = item.get("ops", [{}])[0].get("epilogue_ops", [])
    
    input_shape = item.get("input_shape", [])
    output_shape = item.get("output_shape", [])
    weights = item.get("weights_fused", {})
    
    prompt = textwrap.dedent(f"""
    Implement a Triton kernel for GEMM with custom epilogue fusion.
    
    **Algorithm Strategy:**
    Fuse the following pointwise operations into GEMM epilogue:
    {' -> '.join(epilogue_ops) if epilogue_ops else 'bias + activation'}
    
    **Key Insight:**
    After GEMM accumulation completes, apply all epilogue operations in registers
    before storing to global memory. This avoids intermediate tensor materialization.
    
    **Pseudo-algorithm:**
    ```python
    for each output tile (i, j):
        acc = 0.0
        
        # GEMM accumulation
        for k in range(K):
            x_val = load X[i, k]
            w_val = load W[k, j]
            acc += x_val * w_val
        
        # Epilogue (all in registers!)
        result = acc
        # Apply each epilogue op in sequence
        {_generate_epilogue_code(epilogue_ops)}
        
        # Store final result
        output[i, j] = result
    ```
    
    **Implementation Requirements:**
    1. Input shape: {input_shape}
    2. Output shape: {output_shape}
    3. Weights: {json.dumps(weights, indent=2)}
    4. Epilogue operations: {epilogue_ops}
    5. All epilogue ops must be computed in registers
    
    **Performance Target:**
    - Avoid {len(epilogue_ops) if epilogue_ops else 1} intermediate tensor stores
    - Expected speedup: {metadata.get('expected_speedup', 1.3)}x
    
    Return a complete Python file with kernel_function(...) and self-test.
    """)
    
    return prompt


def _generate_epilogue_code(ops: List[str]) -> str:
    """Generate epilogue code snippet."""
    if not ops:
        return "# Add bias or activation here"
    
    lines = []
    for op in ops:
        if op == "relu":
            lines.append("result = max(result, 0.0)")
        elif op == "gelu":
            lines.append("result = gelu(result)")
        elif op == "sigmoid":
            lines.append("result = 1.0 / (1.0 + exp(-result))")
        elif op == "add":
            lines.append("result = result + bias")
        elif op == "mul":
            lines.append("result = result * scale")
    return "\n        ".join(lines)
