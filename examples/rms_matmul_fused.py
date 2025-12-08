#!/usr/bin/env python3
"""
Example: RMSNorm + Matmul fusion test case.

This demonstrates the algorithmic fusion capability of the enhanced Fuser pipeline.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Simple model with RMSNorm followed by Linear layer."""
    
    def __init__(self, input_dim=512, output_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # RMSNorm layer
        self.norm = nn.RMSNorm(input_dim, eps=1e-5)
        
        # Linear layer (GEMM)
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, x):
        # RMSNorm
        x = self.norm(x)
        
        # Linear (GEMM)
        x = self.linear(x)
        
        return x


def get_inputs():
    """Generate test inputs."""
    batch_size = 32
    input_dim = 512
    
    # Random input tensor
    x = torch.randn(batch_size, input_dim, device='cuda', dtype=torch.float32)
    
    return (x,)


def get_init_inputs():
    """Get initialization inputs for the model."""
    return {
        'input_dim': 512,
        'output_dim': 1024
    }


if __name__ == "__main__":
    # Test the model
    model = Model(**get_init_inputs()).cuda()
    inputs = get_inputs()
    
    with torch.no_grad():
        output = model(*inputs)
    
    print(f"Input shape: {inputs[0].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\n✅ Model test passed!")
    print("\nTo run with Fuser algorithmic optimization:")
    print("python -m Fuser.pipeline_enhanced \\")
    print("  --problem examples/rmsnorm_matmul_example.py \\")
    print("  --extract-model deepseek-chat \\")
    print("  --dispatch-model deepseek-chat \\")
    print("  --compose-model deepseek-chat \\")
    print("  --verify \\")
    print("  --enable-algorithmic-rewrite")