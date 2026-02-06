#!/usr/bin/env python3
"""
计算图构建器 - 方案1实现

从 subgraphs_transformed.json 生成多个完整的计算图变体。
每个计算图是一个完整的、可独立执行的计算流程。

核心思路：
1. 识别变体关系（linear_1 和 linear_1_v1 是替代关系）
2. 生成所有可能的变体组合（限制数量）
3. 每个组合是一个完整的计算图
4. 输出到 compute_graph/ 目录
"""

from __future__ import annotations

import json
import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable


@dataclass
class VariantGroup:
    """变体组"""
    base_id: str  # 基础 ID（如 linear_1）
    variants: List[Dict[str, Any]]  # 所有变体（包括原始）
    position: int  # 在计算图中的位置
    
    def get_best_variant(self, strategy: str = "lowest_cost") -> Dict[str, Any]:
        """根据策略选择最佳变体"""
        if strategy == "lowest_cost":
            # 选择代价最低的
            return min(self.variants, key=lambda v: self._get_cost(v))
        elif strategy == "original":
            # 选择原始版本（没有 _v 后缀）
            for v in self.variants:
                if v["id"] == self.base_id:
                    return v
            return self.variants[0]
        elif strategy == "max_fusion":
            # 选择融合版本（有 _v 后缀的）
            for v in self.variants:
                if v["id"] != self.base_id:
                    return v
            return self.variants[0]
        else:
            return self.variants[0]
    
    def _get_cost(self, variant: Dict[str, Any]) -> float:
        """获取变体的代价"""
        cost_est = variant.get("cost_estimate", {})
        if isinstance(cost_est, dict):
            return sum(cost_est.values())
        return float('inf')


@dataclass
class ComputeGraphVariant:
    """完整计算图变体"""
    id: str
    name: str
    subgraphs: List[Dict[str, Any]]
    variant_selections: Dict[str, str]  # base_id -> selected_variant_id
    total_cost: float
    is_original: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "subgraphs": self.subgraphs,
            "variant_selections": self.variant_selections,
            "total_cost": self.total_cost,
            "is_original": self.is_original,
            "metadata": {
                "num_subgraphs": len(self.subgraphs),
                "subgraph_types": [sg.get("type") for sg in self.subgraphs],
                "subgraph_ids": [sg.get("id") for sg in self.subgraphs],
            }
        }
    
    def signature(self) -> str:
        """计算签名（用于去重）"""
        # 基于操作序列的签名
        ops = []
        for sg in self.subgraphs:
            for op in sg.get("ops", []):
                if isinstance(op, dict):
                    ops.append(op.get("op", ""))
        h = hashlib.md5()
        h.update(json.dumps(ops, sort_keys=True).encode())
        return h.hexdigest()[:16]


class ComputeGraphBuilder:
    """
    计算图构建器
    
    从 subgraphs_transformed.json 生成多个完整计算图
    """
    
    def __init__(self, max_variants: int = 10):
        self.max_variants = max_variants
    
    def build_from_transformed_subgraphs(
        self,
        transformed_subgraphs: List[Dict[str, Any]],
    ) -> Tuple[List[ComputeGraphVariant], Dict[str, Any]]:
        """
        从变换后的子图构建计算图变体
        
        Args:
            transformed_subgraphs: subgraphs_transformed.json 的内容
        
        Returns:
            (计算图变体列表, 统计信息)
        """
        # Step 1: 识别变体组
        variant_groups = self._identify_variant_groups(transformed_subgraphs)
        
        # Step 2: 确定拓扑顺序
        ordered_groups = self._order_variant_groups(variant_groups, transformed_subgraphs)
        
        # Step 3: 生成变体组合
        compute_graphs = self._generate_combinations(ordered_groups)
        
        # Step 4: 去重和排序
        compute_graphs = self._deduplicate_and_rank(compute_graphs)
        
        # Step 5: 限制数量
        compute_graphs = compute_graphs[:self.max_variants]
        
        # 统计信息
        stats = {
            "total_input_subgraphs": len(transformed_subgraphs),
            "variant_groups": len(variant_groups),
            "generated_combinations": len(compute_graphs),
            "variant_group_details": {
                group.base_id: len(group.variants)
                for group in ordered_groups
            }
        }
        
        return compute_graphs, stats
    
    def _identify_variant_groups(
        self,
        subgraphs: List[Dict[str, Any]],
    ) -> Dict[str, VariantGroup]:
        """识别变体组"""
        groups: Dict[str, VariantGroup] = {}
        
        for sg in subgraphs:
            sg_id = sg.get("id", "")
            
            # 提取基础 ID（去掉 _v1, _v2 等后缀）
            base_id = self._extract_base_id(sg_id)
            
            if base_id not in groups:
                groups[base_id] = VariantGroup(
                    base_id=base_id,
                    variants=[],
                    position=0,  # 稍后确定
                )
            
            groups[base_id].variants.append(sg)
        
        return groups
    
    def _extract_base_id(self, sg_id: str) -> str:
        """提取基础 ID"""
        # 移除 _v1, _v2, _v3 等后缀
        for i in range(1, 10):
            suffix = f"_v{i}"
            if sg_id.endswith(suffix):
                return sg_id[:-len(suffix)]
        return sg_id
    
    def _order_variant_groups(
        self,
        groups: Dict[str, VariantGroup],
        subgraphs: List[Dict[str, Any]],
    ) -> List[VariantGroup]:
        """
        确定变体组的拓扑顺序
        
        基于数据流依赖分析，而不仅仅是列表顺序
        """
        # 构建依赖图
        # 每个子图的输出可能是另一个子图的输入
        
        # 首先收集每个 base_id 的输入输出信息
        base_id_inputs: Dict[str, Set[str]] = {}  # base_id -> 输入依赖的 base_ids
        base_id_outputs: Dict[str, str] = {}  # output_name -> base_id
        
        for sg in subgraphs:
            sg_id = sg.get("id", "")
            base_id = self._extract_base_id(sg_id)
            
            if base_id not in base_id_inputs:
                base_id_inputs[base_id] = set()
            
            # 分析输入依赖
            inputs = sg.get("inputs", [])
            ops = sg.get("ops", [])
            
            # 从 ops 中提取输入引用
            for op in ops:
                if isinstance(op, dict):
                    op_inputs = op.get("inputs", [])
                    for inp in op_inputs:
                        if isinstance(inp, str):
                            # 检查是否引用了其他子图的输出
                            inp_base = self._extract_base_id(inp)
                            if inp_base in groups and inp_base != base_id:
                                base_id_inputs[base_id].add(inp_base)
            
            # 记录输出
            output_id = sg.get("output_id") or sg_id
            base_id_outputs[output_id] = base_id
        
        # 拓扑排序
        ordered_base_ids = self._topological_sort(base_id_inputs, groups.keys())
        
        # 如果拓扑排序失败（有环或无法确定），回退到原始顺序
        if not ordered_base_ids:
            ordered_base_ids = []
            seen = set()
            for sg in subgraphs:
                base_id = self._extract_base_id(sg.get("id", ""))
                if base_id not in seen and base_id in groups:
                    seen.add(base_id)
                    ordered_base_ids.append(base_id)
        
        # 按顺序返回变体组
        ordered = []
        for i, base_id in enumerate(ordered_base_ids):
            if base_id in groups:
                group = groups[base_id]
                group.position = i
                ordered.append(group)
        
        return ordered
    
    def _topological_sort(
        self,
        dependencies: Dict[str, Set[str]],
        all_nodes: Iterable[str],
    ) -> List[str]:
        """
        拓扑排序
        
        Args:
            dependencies: 节点 -> 依赖的节点集合
            all_nodes: 所有节点
        
        Returns:
            排序后的节点列表，如果有环则返回空列表
        """
        all_nodes_set = set(all_nodes)
        
        # 计算入度
        in_degree: Dict[str, int] = {node: 0 for node in all_nodes_set}
        for node, deps in dependencies.items():
            if node in all_nodes_set:
                for dep in deps:
                    if dep in all_nodes_set:
                        in_degree[node] = in_degree.get(node, 0) + 1
        
        # Kahn's algorithm
        queue = [node for node in all_nodes_set if in_degree.get(node, 0) == 0]
        result = []
        
        while queue:
            # 选择入度为 0 的节点
            node = queue.pop(0)
            result.append(node)
            
            # 更新依赖此节点的其他节点的入度
            for other_node, deps in dependencies.items():
                if other_node in all_nodes_set and node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        # 检查是否所有节点都被处理
        if len(result) != len(all_nodes_set):
            return []  # 有环
        
        return result
    
    def _generate_combinations(
        self,
        ordered_groups: List[VariantGroup],
    ) -> List[ComputeGraphVariant]:
        """生成变体组合"""
        if not ordered_groups:
            return []
        
        # 计算总组合数
        total_combinations = 1
        for group in ordered_groups:
            total_combinations *= len(group.variants)
        
        # 如果组合数太多，使用采样策略
        if total_combinations > self.max_variants * 2:
            return self._generate_sampled_combinations(ordered_groups)
        else:
            return self._generate_all_combinations(ordered_groups)
    
    def _generate_all_combinations(
        self,
        ordered_groups: List[VariantGroup],
    ) -> List[ComputeGraphVariant]:
        """生成所有组合（适用于组合数较少的情况）"""
        compute_graphs = []
        
        # 获取每个组的变体列表
        variant_lists = [group.variants for group in ordered_groups]
        
        # 生成笛卡尔积
        for i, combination in enumerate(itertools.product(*variant_lists)):
            # 检查是否是原始组合（所有都是基础版本）
            is_original = all(
                sg["id"] == ordered_groups[j].base_id
                for j, sg in enumerate(combination)
            )
            
            # 计算总代价
            total_cost = sum(
                self._get_subgraph_cost(sg) for sg in combination
            )
            
            # 记录选择
            variant_selections = {
                ordered_groups[j].base_id: sg["id"]
                for j, sg in enumerate(combination)
            }
            
            # 创建计算图
            graph = ComputeGraphVariant(
                id=f"compute_graph_{i}",
                name=f"variant_{i}" if not is_original else "original",
                subgraphs=list(combination),
                variant_selections=variant_selections,
                total_cost=total_cost,
                is_original=is_original,
            )
            
            compute_graphs.append(graph)
        
        return compute_graphs
    
    def _generate_sampled_combinations(
        self,
        ordered_groups: List[VariantGroup],
    ) -> List[ComputeGraphVariant]:
        """生成采样的组合（适用于组合数太多的情况）"""
        compute_graphs = []
        
        # 策略 1: 原始组合（全部选基础版本）
        original_combination = [
            group.get_best_variant("original")
            for group in ordered_groups
        ]
        compute_graphs.append(self._create_compute_graph(
            0, original_combination, ordered_groups, is_original=True
        ))
        
        # 策略 2: 最低代价组合
        lowest_cost_combination = [
            group.get_best_variant("lowest_cost")
            for group in ordered_groups
        ]
        compute_graphs.append(self._create_compute_graph(
            1, lowest_cost_combination, ordered_groups, is_original=False
        ))
        
        # 策略 3: 最大融合组合
        max_fusion_combination = [
            group.get_best_variant("max_fusion")
            for group in ordered_groups
        ]
        compute_graphs.append(self._create_compute_graph(
            2, max_fusion_combination, ordered_groups, is_original=False
        ))
        
        # 策略 4-N: 每次只改变一个位置的变体
        idx = 3
        for i, group in enumerate(ordered_groups):
            if len(group.variants) > 1:
                # 对于有多个变体的组，尝试每个变体
                for variant in group.variants:
                    if variant["id"] == group.base_id:
                        continue  # 跳过原始版本（已经在策略1中）
                    
                    # 创建组合：其他位置用原始版本，这个位置用变体
                    combination = [
                        group.get_best_variant("original")
                        for group in ordered_groups
                    ]
                    combination[i] = variant
                    
                    compute_graphs.append(self._create_compute_graph(
                        idx, combination, ordered_groups, is_original=False
                    ))
                    idx += 1
                    
                    if idx >= self.max_variants * 2:
                        break
            
            if idx >= self.max_variants * 2:
                break
        
        return compute_graphs
    
    def _create_compute_graph(
        self,
        idx: int,
        combination: List[Dict[str, Any]],
        ordered_groups: List[VariantGroup],
        is_original: bool,
    ) -> ComputeGraphVariant:
        """创建计算图对象"""
        total_cost = sum(self._get_subgraph_cost(sg) for sg in combination)
        
        variant_selections = {
            ordered_groups[j].base_id: sg["id"]
            for j, sg in enumerate(combination)
        }
        
        return ComputeGraphVariant(
            id=f"compute_graph_{idx}",
            name="original" if is_original else f"variant_{idx}",
            subgraphs=list(combination),
            variant_selections=variant_selections,
            total_cost=total_cost,
            is_original=is_original,
        )
    
    def _get_subgraph_cost(self, sg: Dict[str, Any]) -> float:
        """获取子图代价"""
        cost_est = sg.get("cost_estimate", {})
        if isinstance(cost_est, dict):
            return sum(cost_est.values())
        return 0.0
    
    def _deduplicate_and_rank(
        self,
        compute_graphs: List[ComputeGraphVariant],
    ) -> List[ComputeGraphVariant]:
        """去重并排序"""
        # 去重
        seen_signatures: Set[str] = set()
        unique_graphs = []
        
        for graph in compute_graphs:
            sig = graph.signature()
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_graphs.append(graph)
        
        # 排序：原始版本优先，然后按代价排序
        unique_graphs.sort(key=lambda g: (not g.is_original, g.total_cost))
        
        return unique_graphs
    
    def save_compute_graphs(
        self,
        compute_graphs: List[ComputeGraphVariant],
        output_dir: Path,
    ) -> Path:
        """保存计算图到文件"""
        # 创建 compute_graph 目录
        compute_graph_dir = output_dir / "compute_graph"
        compute_graph_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存每个计算图
        for graph in compute_graphs:
            graph_path = compute_graph_dir / f"{graph.id}.json"
            with open(graph_path, 'w', encoding='utf-8') as f:
                json.dump(graph.to_dict(), f, indent=2)
        
        # 保存清单
        manifest = {
            "total_graphs": len(compute_graphs),
            "original_graph_id": next(
                (g.id for g in compute_graphs if g.is_original),
                compute_graphs[0].id if compute_graphs else None
            ),
            "variant_graph_ids": [
                g.id for g in compute_graphs if not g.is_original
            ],
            "graphs": [
                {
                    "id": g.id,
                    "name": g.name,
                    "is_original": g.is_original,
                    "total_cost": g.total_cost,
                    "variant_selections": g.variant_selections,
                }
                for g in compute_graphs
            ]
        }
        
        manifest_path = compute_graph_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        return compute_graph_dir


def build_compute_graphs_from_file(
    transformed_subgraphs_path: Path,
    output_dir: Path,
    max_variants: int = 10,
) -> Tuple[Path, Dict[str, Any]]:
    """
    从 subgraphs_transformed.json 构建计算图
    
    Args:
        transformed_subgraphs_path: subgraphs_transformed.json 路径
        output_dir: 输出目录（通常是 run_dir）
        max_variants: 最大变体数量
    
    Returns:
        (compute_graph 目录路径, 统计信息)
    """
    # 读取变换后的子图
    with open(transformed_subgraphs_path, 'r', encoding='utf-8') as f:
        transformed_subgraphs = json.load(f)
    
    # 构建计算图
    builder = ComputeGraphBuilder(max_variants=max_variants)
    compute_graphs, stats = builder.build_from_transformed_subgraphs(
        transformed_subgraphs
    )
    
    # 保存到文件
    compute_graph_dir = builder.save_compute_graphs(compute_graphs, output_dir)
    
    # 更新统计信息
    stats["output_dir"] = str(compute_graph_dir)
    stats["saved_graphs"] = len(compute_graphs)
    
    return compute_graph_dir, stats
