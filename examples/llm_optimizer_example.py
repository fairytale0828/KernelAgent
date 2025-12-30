#!/usr/bin/env python3
"""
Example: Using LLM-First Optimizer for complex operators.

This example demonstrates how to use the new LLM-First optimization system
to automatically discover and implement optimizations for complex operators.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Fuser.optimize_cli import run_optimization


def example_attention_optimization():
    """
    Example: Optimize a scaled dot-product attention implementation.
    
    The system will automatically try:
    - Baseline naive attention
    - FlashAttention-style streaming softmax
    - Different fusion boundaries
    - Different schedule configurations
    """
    print("=" * 60)
    print("Example 1: Attention Optimization")
    print("=" * 60)
    
    # Create a simple attention problem
    problem_code = '''
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    """Scaled dot-product attention."""
    
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
    def forward(self, q, k, v):
        """
        Args:
            q: [batch, seq_len, d_model]
            k: [batch, seq_len, d_model]
            v: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = q.shape
        
        # Reshape for multi-head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return output

def get_init_inputs():
    """Initialize model inputs."""
    return {}

def get_inputs():
    """Get test inputs."""
    batch, seq_len, d_model = 2, 128, 512
    return {
        'q': torch.randn(batch, seq_len, d_model, device='cuda', dtype=torch.float16),
        'k': torch.randn(batch, seq_len, d_model, device='cuda', dtype=torch.float16),
        'v': torch.randn(batch, seq_len, d_model, device='cuda', dtype=torch.float16),
    }
'''
    
    # Save problem to file
    problem_path = Path("/tmp/attention_problem.py")
    problem_path.write_text(problem_code)
    
    # Run optimization
    result = run_optimization(
        problem_path=problem_path,
        model="gpt-5",
        workers=2,  # Use 2 workers for demo
        plans_per_worker=3,
        refinements_per_plan=2,
        stream_mode="none",  # Quiet mode for demo
        verify=True,
    )
    
    print("\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Reason: {result['reason']}")
    if result.get('verification'):
        print(f"  Verification: {result['verification']['passed']}")
    
    return result


def example_rmsnorm_gemm_optimization():
    """
    Example: Optimize RMSNorm + GEMM fusion.
    
    The system will automatically try:
    - Baseline: separate RMSNorm and GEMM
    - Epilogue scaling: move scaling to GEMM output
    - Gamma folding: fold gamma into weights
    - Fully fused single-kernel implementation
    """
    print("\n" + "=" * 60)
    print("Example 2: RMSNorm + GEMM Optimization")
    print("=" * 60)
    
    problem_code = '''
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    """RMSNorm followed by linear projection."""
    
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.weight = torch.nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.eps = 1e-6
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, intermediate_size]
        """
        # RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.gamma
        
        # Linear projection
        output = F.linear(x, self.weight)
        return output

def get_init_inputs():
    """Initialize model inputs."""
    return {}

def get_inputs():
    """Get test inputs."""
    batch, seq_len, hidden_size = 2, 512, 1024
    return {
        'x': torch.randn(batch, seq_len, hidden_size, device='cuda', dtype=torch.float16),
    }
'''
    
    # Save problem to file
    problem_path = Path("/tmp/rmsnorm_gemm_problem.py")
    problem_path.write_text(problem_code)
    
    # Run optimization
    result = run_optimization(
        problem_path=problem_path,
        model="gpt-5",
        workers=2,
        plans_per_worker=3,
        refinements_per_plan=2,
        stream_mode="none",
        verify=True,
    )
    
    print("\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Reason: {result['reason']}")
    if result.get('verification'):
        print(f"  Verification: {result['verification']['passed']}")
    
    return result


def main():
    """Run all examples."""
    print("LLM-First Optimizer Examples")
    print("=" * 60)
    print()
    print("These examples demonstrate automatic optimization discovery")
    print("for complex operators using the 4-agent system:")
    print("  - Planner: Generate multiple optimization plans")
    print("  - Coder: Implement selected plans")
    print("  - Critic: Audit code before execution")
    print("  - Refiner: Fix errors and improve performance")
    print()
    
    # Example 1: Attention
    try:
        result1 = example_attention_optimization()
    except Exception as e:
        print(f"Example 1 failed: {e}")
        result1 = None
    
    # Example 2: RMSNorm + GEMM
    try:
        result2 = example_rmsnorm_gemm_optimization()
    except Exception as e:
        print(f"Example 2 failed: {e}")
        result2 = None
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Example 1 (Attention): {'✓ Success' if result1 and result1['success'] else '✗ Failed'}")
    print(f"Example 2 (RMSNorm+GEMM): {'✓ Success' if result2 and result2['success'] else '✗ Failed'}")
    print()
    print("For more details, check the output directories:")
    if result1:
        print(f"  Example 1: {result1.get('run_dir', 'N/A')}")
    if result2:
        print(f"  Example 2: {result2.get('run_dir', 'N/A')}")


if __name__ == "__main__":
    main()
