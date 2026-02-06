import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LoRA-style linear layer: W×X + B×A×X
    
    This computes: output = W @ X + (B @ A) @ X
    where B and A form a low-rank adaptation.
    """
    
    def __init__(self, in_features=1024, out_features=1024, rank=8):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Main weight matrix W: (out_features, in_features)
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        
        # Low-rank matrices
        # B: (out_features, rank)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        # A: (rank, in_features)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: W×X + B×A×X
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # x: (batch_size, in_features)
        # Transpose for matrix multiplication: (in_features, batch_size)
        x_t = x.t()
        
        # Main path: W @ X
        # W: (out_features, in_features), x_t: (in_features, batch_size)
        # Result: (out_features, batch_size)
        wx = torch.matmul(self.W, x_t)
        
        # LoRA path: B @ A @ X
        # Step 1: A @ X
        # A: (rank, in_features), x_t: (in_features, batch_size)
        # Result: (rank, batch_size)
        ax = torch.matmul(self.A, x_t)
        
        # Step 2: B @ (A @ X)
        # B: (out_features, rank), ax: (rank, batch_size)
        # Result: (out_features, batch_size)
        bax = torch.matmul(self.B, ax)
        
        # Combine: W×X + B×A×X
        # (out_features, batch_size)
        output = wx + bax
        
        # Transpose back to (batch_size, out_features)
        return output.t()


batch_size = 32
in_features = 1024
out_features = 1024
rank = 8


def get_inputs():
    x = torch.rand(batch_size, in_features)
    return [x]


def get_init_inputs():
    return [in_features, out_features, rank]
