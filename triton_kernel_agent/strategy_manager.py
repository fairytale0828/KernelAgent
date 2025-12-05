"""Triton优化策略管理器"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class OptimizationStrategy:
    """优化策略定义"""
    strategy_id: str
    name: str
    description: str
    guidelines: str


class StrategyManager:
    """管理Triton算子优化策略集合"""

    def __init__(self):
        """初始化策略管理器"""
        self.strategies = self._init_strategies()

    def _init_strategies(self) -> Dict[str, OptimizationStrategy]:
        """初始化策略集合"""
        strategies = {
            "Triton.TileOnly.v1": OptimizationStrategy(
                strategy_id="Triton.TileOnly.v1",
                name="分块优化策略",
                description="侧重选择合适的BLOCK_M/BLOCK_N/BLOCK_K和num_warps，优化分块策略。",
                guidelines="""
重点关注：
- 选择合适的BLOCK_M、BLOCK_N、BLOCK_K大小
- 调整num_warps以匹配硬件warp数量
- 确保分块大小是2的幂次方
- 考虑shared memory限制
- 平衡寄存器使用和occupancy
"""
            ),

            "Triton.TileVectorize.v1": OptimizationStrategy(
                strategy_id="Triton.TileVectorize.v1",
                name="分块+向量化策略",
                description="在分块基础上尽量使用向量化load/store，提高内存访问效率。",
                guidelines="""
重点关注：
- 在分块基础上使用向量化load/store
- 使用tl.load和tl.store的mask参数
- 确保内存访问对齐
- 利用向量化指令提高带宽利用率
- 考虑cache line大小
"""
            ),

            "Triton.ReduceOpt.v1": OptimizationStrategy(
                strategy_id="Triton.ReduceOpt.v1",
                name="归约优化策略",
                description="优化归约和访存模式，减少非coalesced访问，使用高效的归约算法。",
                guidelines="""
重点关注：
- 使用tl.reduce等高效归约操作
- 优化归约维度的选择
- 减少非coalesced内存访问
- 使用shared memory进行中间结果缓存
- 考虑warp-level归约优化
"""
            ),

            "Triton.SharedMemOpt.v1": OptimizationStrategy(
                strategy_id="Triton.SharedMemOpt.v1",
                name="共享内存优化策略",
                description="充分利用共享内存进行数据缓存和重用，减少全局内存访问。",
                guidelines="""
重点关注：
- 使用shared memory缓存频繁访问的数据
- 优化shared memory的bank conflict
- 合理规划shared memory布局
- 平衡shared memory使用和occupancy
- 使用double buffering技术
"""
            ),

            "Triton.WarpOpt.v1": OptimizationStrategy(
                strategy_id="Triton.WarpOpt.v1",
                name="Warp级优化策略",
                description="优化warp级别的执行，使用warp shuffle和协作组操作。",
                guidelines="""
重点关注：
- 优化warp内的数据共享
- 减少warp divergence
- 使用warp-level原语
- 优化线程块内的同步
- 考虑warp调度和执行效率
"""
            ),

            "Triton.CoalescedAccess.v1": OptimizationStrategy(
                strategy_id="Triton.CoalescedAccess.v1",
                name="合并访问优化策略",
                description="确保内存访问模式的合并，优化全局内存访问效率。",
                guidelines="""
重点关注：
- 确保连续线程访问连续内存
- 优化stride访问模式
- 使用合适的数据布局
- 避免非对齐访问
- 最大化内存带宽利用率
"""
            ),

            "Triton.TensorCore.v1": OptimizationStrategy(
                strategy_id="Triton.TensorCore.v1",
                name="Tensor Core优化策略",
                description="充分利用Tensor Core进行混合精度计算，优化矩阵乘法性能。",
                guidelines="""
重点关注：
- 使用tl.dot进行矩阵乘法
- 选择合适的数据类型（fp16/bf16）
- 确保矩阵维度满足Tensor Core要求
- 优化数据布局以匹配Tensor Core
- 平衡精度和性能
"""
            ),

            "Triton.AsyncCopy.v1": OptimizationStrategy(
                strategy_id="Triton.AsyncCopy.v1",
                name="异步拷贝优化策略",
                description="使用异步内存拷贝和流水线技术，隐藏内存延迟。",
                guidelines="""
重点关注：
- 使用异步内存拷贝指令
- 实现软件流水线
- 重叠计算和内存访问
- 使用多级缓冲
- 优化数据预取策略
"""
            ),
        }
        return strategies

    def get_strategy(self, strategy_id: str) -> OptimizationStrategy:
        """获取指定策略"""
        return self.strategies.get(strategy_id)

    def get_all_strategies(self) -> List[OptimizationStrategy]:
        """获取所有策略"""
        return list(self.strategies.values())

    def get_strategy_ids(self) -> List[str]:
        """获取所有策略ID"""
        return list(self.strategies.keys())

    def get_strategy_guidelines(self, strategy_id: str) -> str:
        """获取策略的优化指导"""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            return f"{strategy.description}\n\n{strategy.guidelines}"
        return ""
