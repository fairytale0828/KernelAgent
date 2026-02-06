"""
代价模型

用于评估表达式的执行代价，指导最优提取。
支持多种代价模型：
- IOAware: 优化 global memory IO
- FLOPS: 优化计算量
- Balanced: 平衡 IO 和计算
- Learned: 基于学习的代价模型 (预留)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .egraph import EGraph, ENode, EClass
    from .ir import TensorMeta


@dataclass
class Cost:
    """代价值"""
    value: float
    breakdown: Dict[str, float]  # 分项代价
    
    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            value=self.value + other.value,
            breakdown={
                k: self.breakdown.get(k, 0) + other.breakdown.get(k, 0)
                for k in set(self.breakdown) | set(other.breakdown)
            }
        )
    
    def __lt__(self, other: "Cost") -> bool:
        return self.value < other.value
    
    def __le__(self, other: "Cost") -> bool:
        return self.value <= other.value
    
    @classmethod
    def zero(cls) -> "Cost":
        return cls(value=0.0, breakdown={})
    
    @classmethod
    def infinity(cls) -> "Cost":
        return cls(value=float('inf'), breakdown={})


class CostModel(ABC):
    """代价模型基类"""
    
    @abstractmethod
    def enode_cost(self, enode: "ENode", child_costs: List[Cost]) -> Cost:
        """
        计算 ENode 的代价
        
        Args:
            enode: 要评估的 ENode
            child_costs: 子节点的代价列表
        
        Returns:
            该 ENode 的代价
        """
        pass
    
    def total_cost(self, costs: List[Cost]) -> Cost:
        """计算总代价"""
        if not costs:
            return Cost.zero()
        result = costs[0]
        for c in costs[1:]:
            result = result + c
        return result


class IOAwareCostModel(CostModel):
    """
    IO 感知代价模型
    
    优先减少 global memory IO，适合 memory-bound 操作
    """
    
    # 操作的 IO 代价系数
    IO_COSTS = {
        # 逐元素操作 (读写各一次)
        "add": 2.0, "sub": 2.0, "mul": 2.0, "div": 2.0,
        "neg": 2.0, "abs": 2.0, "sqrt": 2.0, "rsqrt": 2.0,
        "exp": 2.0, "log": 2.0, "tanh": 2.0, "sigmoid": 2.0,
        "relu": 2.0, "gelu": 2.0, "silu": 2.0,
        "sin": 2.0, "cos": 2.0, "where": 3.0, "clamp": 2.0,
        
        # 归约操作 (读多写少)
        "reduce_sum": 1.5, "reduce_mean": 1.5, 
        "reduce_max": 1.5, "reduce_min": 1.5,
        "sum": 1.5, "mean": 1.5, "max": 1.5, "min": 1.5,
        "softmax": 3.0, "log_softmax": 3.0,
        "layer_norm": 4.0, "rms_norm": 3.0,
        
        # MatMul 类 (计算密集，但有中间张量)
        "matmul": 3.0, "bmm": 3.0, "linear": 3.0,
        "matmul_bias": 2.5,  # 融合后更好
        "matmul_nt": 2.8, "matmul_tn": 2.8,
        "scaled_dot_product_attention": 2.0,  # 融合后最优
        
        # 布局操作 (可能需要 IO)
        "reshape": 0.5, "view": 0.1, "transpose": 1.5,
        "permute": 1.5, "contiguous": 2.0,
        "squeeze": 0.1, "unsqueeze": 0.1, "flatten": 0.5,
        "broadcast": 1.0, "expand": 1.0,
        
        # 类型转换
        "cast": 2.0,
        
        # 常量
        "zero": 0.0, "one": 0.0, "const": 0.0,
        
        # 融合操作 (代价更低)
        "matmul_lora": 2.0,
        "multi_head_attention": 2.0,
    }
    
    # 权重系数
    W_IO = 1.0
    W_TEMP = 0.5
    W_KERNEL = 0.3
    
    def __init__(self, 
                 w_io: float = 1.0,
                 w_temp: float = 0.5,
                 w_kernel: float = 0.3):
        self.W_IO = w_io
        self.W_TEMP = w_temp
        self.W_KERNEL = w_kernel
    
    def enode_cost(self, enode: "ENode", child_costs: List[Cost]) -> Cost:
        """计算 ENode 的 IO 代价"""
        op = enode.op
        base_cost = self.IO_COSTS.get(op, 1.0)
        
        # 子节点代价总和
        children_total = self.total_cost(child_costs)
        
        # 计算分项代价
        io_cost = base_cost * self.W_IO
        temp_cost = len(enode.children) * 0.1 * self.W_TEMP  # 中间张量估计
        kernel_cost = 1.0 * self.W_KERNEL  # 每个操作一个 kernel
        
        total = io_cost + temp_cost + kernel_cost + children_total.value
        
        return Cost(
            value=total,
            breakdown={
                "io": io_cost + children_total.breakdown.get("io", 0),
                "temp": temp_cost + children_total.breakdown.get("temp", 0),
                "kernel": kernel_cost + children_total.breakdown.get("kernel", 0),
            }
        )


class FLOPSCostModel(CostModel):
    """
    FLOPs 代价模型
    
    优先减少计算量，适合 compute-bound 操作
    """
    
    # 操作的 FLOPs 代价系数 (相对于输入大小)
    FLOPS_COSTS = {
        # 逐元素操作 (1x)
        "add": 1.0, "sub": 1.0, "mul": 1.0, "div": 1.0,
        "neg": 1.0, "abs": 1.0,
        
        # 复杂逐元素 (多次操作)
        "sqrt": 5.0, "rsqrt": 5.0,
        "exp": 10.0, "log": 10.0,
        "tanh": 15.0, "sigmoid": 10.0,
        "relu": 1.0, "gelu": 20.0, "silu": 15.0,
        "sin": 15.0, "cos": 15.0,
        
        # 归约 (n 次操作)
        "reduce_sum": 1.0, "reduce_mean": 1.5,
        "reduce_max": 1.0, "reduce_min": 1.0,
        "softmax": 5.0, "log_softmax": 6.0,
        "layer_norm": 8.0, "rms_norm": 5.0,
        
        # MatMul (O(n^3) 或 O(mnk))
        "matmul": 100.0, "bmm": 100.0, "linear": 100.0,
        
        # 布局操作 (0 FLOPs)
        "reshape": 0.0, "view": 0.0, "transpose": 0.0,
        "permute": 0.0, "contiguous": 0.0,
        "squeeze": 0.0, "unsqueeze": 0.0, "flatten": 0.0,
        "broadcast": 0.0, "expand": 0.0,
        
        # 类型转换
        "cast": 1.0,
    }
    
    def enode_cost(self, enode: "ENode", child_costs: List[Cost]) -> Cost:
        """计算 ENode 的 FLOPs 代价"""
        op = enode.op
        base_cost = self.FLOPS_COSTS.get(op, 1.0)
        
        children_total = self.total_cost(child_costs)
        total = base_cost + children_total.value
        
        return Cost(
            value=total,
            breakdown={
                "flops": base_cost + children_total.breakdown.get("flops", 0),
            }
        )


class BalancedCostModel(CostModel):
    """
    平衡代价模型
    
    综合考虑 IO、FLOPs、临时张量和 kernel 数量
    """
    
    def __init__(self,
                 w_io: float = 1.0,
                 w_flops: float = 0.5,
                 w_temp: float = 0.3,
                 w_kernel: float = 0.2):
        self.io_model = IOAwareCostModel(w_io=w_io)
        self.flops_model = FLOPSCostModel()
        self.w_flops = w_flops
        self.w_temp = w_temp
        self.w_kernel = w_kernel
    
    def enode_cost(self, enode: "ENode", child_costs: List[Cost]) -> Cost:
        """计算综合代价"""
        io_cost = self.io_model.enode_cost(enode, child_costs)
        flops_cost = self.flops_model.enode_cost(enode, child_costs)
        
        total = (io_cost.value + 
                 flops_cost.value * self.w_flops)
        
        return Cost(
            value=total,
            breakdown={
                **io_cost.breakdown,
                "flops": flops_cost.breakdown.get("flops", 0),
            }
        )


class LatencyCostModel(CostModel):
    """
    延迟代价模型
    
    基于操作的实际执行延迟估计
    """
    
    # 操作延迟 (微秒，相对值)
    LATENCY_US = {
        # 逐元素 (快)
        "add": 0.1, "sub": 0.1, "mul": 0.1, "div": 0.2,
        "relu": 0.1, "sigmoid": 0.3, "tanh": 0.4,
        "exp": 0.3, "log": 0.3, "sqrt": 0.2,
        
        # 归约 (中等)
        "reduce_sum": 1.0, "reduce_mean": 1.2,
        "softmax": 2.0, "layer_norm": 3.0,
        
        # MatMul (慢)
        "matmul": 10.0, "linear": 10.0, "bmm": 12.0,
        
        # 布局 (取决于是否需要复制)
        "transpose": 1.0, "reshape": 0.1, "contiguous": 2.0,
        
        # 融合操作 (优化后)
        "scaled_dot_product_attention": 8.0,
        "matmul_bias": 10.5,
    }
    
    def enode_cost(self, enode: "ENode", child_costs: List[Cost]) -> Cost:
        """计算延迟代价"""
        op = enode.op
        latency = self.LATENCY_US.get(op, 1.0)
        
        children_total = self.total_cost(child_costs)
        total = latency + children_total.value
        
        return Cost(
            value=total,
            breakdown={
                "latency_us": latency + children_total.breakdown.get("latency_us", 0),
            }
        )


def get_cost_model(name: str, **kwargs) -> CostModel:
    """
    获取代价模型实例
    
    Args:
        name: 模型名称 ("io_aware", "flops", "balanced", "latency")
        **kwargs: 模型参数
    
    Returns:
        CostModel 实例
    """
    models = {
        "io_aware": IOAwareCostModel,
        "flops": FLOPSCostModel,
        "balanced": BalancedCostModel,
        "latency": LatencyCostModel,
    }
    
    if name not in models:
        raise ValueError(f"Unknown cost model: {name}. Available: {list(models.keys())}")
    
    return models[name](**kwargs)
