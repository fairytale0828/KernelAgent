"""
子图聚合器

在 E-Graph 代数优化之后，识别融合模式并聚合子图，同时保留优化提示。

流程：
1. 分析 E-Graph 输出的细粒度子图
2. 识别可融合的模式（Flash Attention, MLP Block 等）
3. 聚合相关子图为融合子图
4. 保留 E-Graph 发现的优化提示（online_softmax, late_scaling 等）
5. 输出供 Dispatch 使用的聚合子图
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# 融合模式定义
# =============================================================================

@dataclass
class FusionPattern:
    """融合模式定义"""
    name: str                           # 模式名称
    op_pattern: List[str]               # 操作模式 (支持正则)
    fused_type: str                     # 融合后的类型
    priority: int = 10                  # 优先级
    min_ops: int = 2                    # 最少操作数
    max_ops: int = 20                   # 最多操作数
    description: str = ""               # 描述
    codegen_hints: Dict[str, Any] = field(default_factory=dict)  # 代码生成提示

    def matches(self, ops: List[str]) -> bool:
        """检查操作序列是否匹配此模式"""
        if len(ops) < self.min_ops or len(ops) > self.max_ops:
            return False
        return self._pattern_match(ops, self.op_pattern)

    def _pattern_match(self, ops: List[str], pattern: List[str]) -> bool:
        """模式匹配"""
        if not pattern:
            return True

        # 简单的顺序匹配
        pattern_idx = 0
        for op in ops:
            if pattern_idx >= len(pattern):
                break

            p = pattern[pattern_idx]
            if p == "*":
                # 通配符，跳过
                pattern_idx += 1
            elif p.startswith("^"):
                # 正则匹配
                if re.match(p, op):
                    pattern_idx += 1
            elif op == p:
                pattern_idx += 1

        return pattern_idx >= len(pattern)


# 预定义融合模式
FUSION_PATTERNS: List[FusionPattern] = [
    # Flash Attention: matmul -> [scale] -> softmax -> matmul
    FusionPattern(
        name="flash_attention",
        op_pattern=["matmul", "^(mul|div|online_softmax|softmax)", "matmul"],
        fused_type="flash_attention",
        priority=100,
        min_ops=3,
        max_ops=20,
        description="Flash Attention: Q@K^T -> scale -> softmax -> @V",
        codegen_hints={
            "tiling": True,
            "online_softmax": True,
            "recompute_softmax": True,
            "block_size_q": 64,
            "block_size_kv": 64,
        },
    ),

    # Scaled Softmax Attention (不含最后的 matmul)
    FusionPattern(
        name="scaled_softmax",
        op_pattern=["matmul", "^(mul|div)", "^(softmax|online_softmax)"],
        fused_type="scaled_softmax_attention",
        priority=90,
        min_ops=3,
        max_ops=15,
        description="Scaled Softmax: matmul -> scale -> softmax",
        codegen_hints={
            "fuse_scale": True,
            "online_softmax": True,
        },
    ),

    # MLP Block: matmul -> [bias] -> activation -> matmul -> [bias]
    FusionPattern(
        name="mlp_block",
        op_pattern=["matmul", "^(gelu|relu|silu)", "matmul"],
        fused_type="fused_mlp",
        priority=80,
        min_ops=3,
        max_ops=15,
        description="MLP: Linear -> Activation -> Linear",
        codegen_hints={
            "fuse_bias": True,
            "fuse_activation": True,
        },
    ),

    # SwiGLU: matmul -> silu -> mul -> matmul
    FusionPattern(
        name="swiglu",
        op_pattern=["matmul", "silu", "mul", "matmul"],
        fused_type="swiglu",
        priority=85,
        min_ops=4,
        max_ops=10,
        description="SwiGLU activation pattern",
        codegen_hints={
            "gate_activation": "silu",
        },
    ),

    # RMSNorm + Linear (Late Scaling)
    FusionPattern(
        name="rmsnorm_linear",
        op_pattern=["reduce_mean", "rsqrt", "mul", "matmul"],
        fused_type="rmsnorm_linear_fused",
        priority=75,
        min_ops=4,
        max_ops=12,
        description="RMSNorm followed by Linear with late scaling",
        codegen_hints={
            "late_scaling": True,
            "fuse_norm": True,
        },
    ),

    # LayerNorm + Linear
    FusionPattern(
        name="layernorm_linear",
        op_pattern=["reduce_mean", "sub", "rsqrt", "mul", "matmul"],
        fused_type="layernorm_linear_fused",
        priority=70,
        min_ops=5,
        max_ops=15,
        description="LayerNorm followed by Linear",
        codegen_hints={
            "fuse_norm": True,
        },
    ),

    # MatMul + Bias + Activation
    FusionPattern(
        name="matmul_bias_activation",
        op_pattern=["matmul", "add", "^(relu|gelu|silu|sigmoid|tanh)"],
        fused_type="matmul_bias_act",
        priority=60,
        min_ops=3,
        max_ops=6,
        description="MatMul with bias and activation",
        codegen_hints={
            "fuse_bias": True,
            "fuse_activation": True,
        },
    ),

    # Softmax + MatMul
    FusionPattern(
        name="softmax_matmul",
        op_pattern=["^(softmax|online_softmax)", "matmul"],
        fused_type="softmax_matmul",
        priority=85,
        min_ops=2,
        max_ops=5,
        description="Softmax followed by MatMul",
        codegen_hints={
            "online_softmax": True,
            "fuse_softmax_matmul": True,
        },
    ),
]


# =============================================================================
# 优化提示提取器
# =============================================================================

class OptimizationHintExtractor:
    """从 E-Graph 输出中提取优化提示"""

    @staticmethod
    def extract(subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """提取子图中的优化提示"""
        hints = {}

        ops = subgraph.get("ops", [])
        transforms = subgraph.get("transforms", [])

        # 检查操作类型
        op_names = [op.get("op", "") for op in ops if isinstance(op, dict)]

        # Online Softmax
        if "online_softmax" in op_names:
            hints["online_softmax"] = True

        # Late Scaling (scale 在 matmul 之后)
        if "matmul" in op_names and ("mul" in op_names or "div" in op_names):
            # 检查顺序
            matmul_idx = next(
                (i for i, op in enumerate(op_names) if op == "matmul"), -1)
            scale_idx = next((i for i, op in enumerate(
                op_names) if op in ("mul", "div")), -1)
            if matmul_idx >= 0 and scale_idx > matmul_idx:
                hints["late_scaling"] = True

        # MatMul + Bias 融合
        if "matmul_bias" in op_names:
            hints["fused_matmul_bias"] = True

        # 从 transforms 中提取
        for transform in transforms:
            if "late_scaling" in transform.lower():
                hints["late_scaling"] = True
            if "online_softmax" in transform.lower():
                hints["online_softmax"] = True
            if "matmul_bias" in transform.lower():
                hints["fused_matmul_bias"] = True

        # 从 cost_estimate 中提取
        cost = subgraph.get("cost_estimate", {})
        if cost:
            hints["estimated_io_cost"] = cost.get("io", 0)
            hints["estimated_kernel_cost"] = cost.get("kernel", 0)

        return hints


# =============================================================================
# 子图聚合器
# =============================================================================

class SubgraphAggregator:
    """
    子图聚合器

    识别融合模式，聚合子图，保留优化提示
    """

    def __init__(
        self,
        patterns: Optional[List[FusionPattern]] = None,
        min_fusion_benefit: float = 0.1,
    ):
        self.patterns = patterns or FUSION_PATTERNS
        # 按优先级排序
        self.patterns.sort(key=lambda p: -p.priority)
        self.min_fusion_benefit = min_fusion_benefit
        self.hint_extractor = OptimizationHintExtractor()

    def aggregate(
        self,
        subgraphs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        聚合子图

        Args:
            subgraphs: E-Graph 输出的细粒度子图列表

        Returns:
            (聚合后的子图列表, 统计信息)
        """
        stats = {
            "input_count": len(subgraphs),
            "fusions_found": 0,
            "fusions_applied": 0,
            "patterns_matched": {},
            "optimization_hints_preserved": 0,
        }

        if len(subgraphs) <= 1:
            # 即使只有一个子图，也要提取优化提示
            if subgraphs:
                hints = self.hint_extractor.extract(subgraphs[0])
                if hints:
                    subgraphs[0]["codegen_hints"] = hints
                    stats["optimization_hints_preserved"] = 1
            return subgraphs, stats

        # 提取每个子图的主要操作
        sg_ops = self._extract_ops(subgraphs)

        # 提取每个子图的优化提示
        sg_hints = [self.hint_extractor.extract(sg) for sg in subgraphs]

        # 查找融合机会
        fusion_candidates = self._find_fusion_candidates(subgraphs, sg_ops)
        stats["fusions_found"] = len(fusion_candidates)

        # 应用融合
        result = []
        fused_indices: Set[int] = set()

        for pattern, indices in fusion_candidates:
            # 检查是否已经被融合
            if any(i in fused_indices for i in indices):
                continue

            # 执行融合
            fused_sg = self._fuse_subgraphs(
                [subgraphs[i] for i in indices],
                [sg_hints[i] for i in indices],
                pattern
            )

            result.append(fused_sg)
            fused_indices.update(indices)
            stats["fusions_applied"] += 1
            stats["patterns_matched"][pattern.name] = \
                stats["patterns_matched"].get(pattern.name, 0) + 1

        # 添加未融合的子图（保留其优化提示）
        for i, sg in enumerate(subgraphs):
            if i not in fused_indices:
                if sg_hints[i]:
                    sg["codegen_hints"] = sg_hints[i]
                    stats["optimization_hints_preserved"] += 1
                result.append(sg)

        stats["output_count"] = len(result)
        return result, stats

    def _extract_ops(self, subgraphs: List[Dict[str, Any]]) -> List[List[str]]:
        """提取每个子图的操作列表"""
        result = []
        for sg in subgraphs:
            ops = sg.get("ops", [])
            op_names = []
            for op in ops:
                if isinstance(op, dict):
                    op_name = op.get("op", "")
                    # 跳过 weight 和 const 节点
                    if op_name and op_name not in ("weight", "const"):
                        op_names.append(op_name)
            result.append(op_names)
        return result

    def _find_fusion_candidates(
        self,
        subgraphs: List[Dict[str, Any]],
        sg_ops: List[List[str]]
    ) -> List[Tuple[FusionPattern, List[int]]]:
        """查找融合候选"""
        candidates = []
        n = len(subgraphs)

        # 尝试每个模式
        for pattern in self.patterns:
            # 滑动窗口
            for window_size in range(2, min(n + 1, 8)):  # 最多聚合 7 个子图
                for start in range(n - window_size + 1):
                    indices = list(range(start, start + window_size))

                    # 收集窗口内的所有操作
                    window_ops = []
                    for i in indices:
                        window_ops.extend(sg_ops[i])

                    # 检查是否匹配模式
                    if pattern.matches(window_ops):
                        candidates.append((pattern, indices))

        # 按优先级和覆盖范围排序
        candidates.sort(key=lambda x: (-x[0].priority, -len(x[1])))

        return candidates

    def _fuse_subgraphs(
        self,
        subgraphs: List[Dict[str, Any]],
        hints_list: List[Dict[str, Any]],
        pattern: FusionPattern
    ) -> Dict[str, Any]:
        """融合多个子图"""
        # 收集所有操作
        all_ops = []
        for sg in subgraphs:
            all_ops.extend(sg.get("ops", []))

        # 确定输入输出形状
        input_shape = subgraphs[0].get("input_shape")
        if not input_shape:
            inputs = subgraphs[0].get("inputs")
            if inputs and isinstance(inputs, list) and len(inputs) > 0:
                input_shape = inputs[0] if isinstance(
                    inputs[0], list) else inputs

        output_shape = subgraphs[-1].get("output_shape")

        # 收集权重
        weights_fused = {}
        for sg in subgraphs:
            wf = sg.get("weights_fused")
            if wf:
                weights_fused.update(wf)

        # 合并优化提示
        merged_hints = dict(pattern.codegen_hints)  # 从模式开始
        for hints in hints_list:
            merged_hints.update(hints)

        # 生成融合子图 ID
        base_ids = [sg.get("id", "sg") for sg in subgraphs]
        fused_id = f"{pattern.fused_type}_{'_'.join(base_ids[:2])}"
        if len(base_ids) > 2:
            fused_id += f"_plus{len(base_ids) - 2}"

        # 收集 source 信息
        sources = [sg.get("source") for sg in subgraphs if sg.get("source")]
        source_info = None
        if sources:
            source_info = {
                "module": sources[0].get("module", "Fused"),
                "code": "\n".join(s.get("code", "") for s in sources if s.get("code")),
            }

        # 构建融合子图
        fused_sg = {
            "id": fused_id,
            "type": pattern.fused_type,
            "ops": all_ops,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "weights_fused": weights_fused if weights_fused else None,
            "fusion_pattern": pattern.name,
            "fused_from": base_ids,
            "codegen_hints": merged_hints,
            "description": pattern.description,
        }

        if source_info:
            fused_sg["source"] = source_info

        # 保留原始的 where 信息
        wheres = [sg.get("where") for sg in subgraphs if sg.get("where")]
        if wheres:
            fused_sg["where"] = " + ".join(wheres[:3])
            if len(wheres) > 3:
                fused_sg["where"] += f" + {len(wheres) - 3} more"

        return fused_sg

    def aggregate_json(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        聚合 JSON 文件

        Args:
            input_path: 输入文件路径 (E-Graph 输出)
            output_path: 输出文件路径

        Returns:
            统计信息
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            subgraphs = json.load(f)

        result, stats = self.aggregate(subgraphs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        return stats


# =============================================================================
# 便捷函数
# =============================================================================

def aggregate_subgraphs(
    subgraphs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    聚合子图

    Args:
        subgraphs: 子图列表

    Returns:
        (聚合后的子图列表, 统计信息)
    """
    aggregator = SubgraphAggregator()
    return aggregator.aggregate(subgraphs)


def aggregate_json_file(
    input_path: str,
    output_path: str
) -> Dict[str, Any]:
    """
    聚合 JSON 文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径

    Returns:
        统计信息
    """
    aggregator = SubgraphAggregator()
    return aggregator.aggregate_json(Path(input_path), Path(output_path))
