# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
通用代数优化器

基于声明式规则和 E-Matching 引擎实现自动优化探索
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .ir import SubgraphIR, parse_subgraph_json, ir_to_json, OpNode, TensorMeta
from .egraph import EGraph, ENode
from .pattern_match import (
    RewriteRule,
    EMatchEngine,
    ConstraintChecker,
    Match,
    RuleApplication,
    ShapeInfo,
    parse_pattern,
)
from .rules.rule_library import get_rule_set, get_all_rules
from .cost_model import CostModel, IOAwareCostModel, get_cost_model, Cost
from .extractor import Extractor, ExtractedExpr, KBestExtractor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OptimizerConfig:
    """优化器配置"""
    # 规则配置
    rule_set: str = "default"          # 规则集名称
    custom_rules: List[RewriteRule] = field(default_factory=list)

    # 饱和搜索配置
    max_iterations: int = 30           # 最大迭代次数
    max_eclasses: int = 10000          # 最大 E-Class 数量
    max_enodes: int = 100000           # 最大 E-Node 数量
    time_budget_ms: int = 10000        # 时间预算（毫秒）

    # 提取配置
    top_k: int = 5                     # 提取 Top-K 候选
    cost_model: str = "io_aware"       # 代价模型

    # 输出配置
    include_original: bool = True      # 包含原始表达式
    include_proof: bool = True         # 包含变换证明
    deduplicate: bool = True           # 去重

    # 展开配置
    expand_ops: bool = True            # 展开高级操作
    expand_norms: bool = True          # 展开归一化操作


@dataclass
class OptimizationResult:
    """优化结果"""
    original_ir: SubgraphIR
    variants: List[Tuple[SubgraphIR, ExtractedExpr]]
    stats: Dict[str, Any]
    applied_rules: List[RuleApplication]


# =============================================================================
# Subgraph to E-Graph Converter
# =============================================================================

class SubgraphToEGraph:
    """将 SubgraphIR 转换为 E-Graph"""

    def __init__(self, egraph: EGraph):
        self.egraph = egraph
        self._node_to_eclass: Dict[str, int] = {}
        self._shape_info: Dict[int, ShapeInfo] = {}

    def convert(self, ir: SubgraphIR) -> Tuple[int, Dict[int, ShapeInfo]]:
        """
        转换 SubgraphIR 到 E-Graph

        Returns:
            (root_eclass_id, shape_info)
        """
        # 添加输入节点
        for input_name, input_meta in ir.inputs:
            enode = ENode(
                op="input",
                children=(),
                attrs=(("name", input_name),),
            )
            eclass_id = self.egraph.add(enode)
            self._node_to_eclass[input_name] = eclass_id

            if input_meta and input_meta.shape:
                self._shape_info[eclass_id] = ShapeInfo(
                    shape=list(input_meta.shape),
                    dtype=input_meta.dtype.value if input_meta.dtype else "float32",
                )

        # 添加权重节点
        for weight_name, weight_meta in ir.weights.items():
            enode = ENode(
                op="weight",
                children=(),
                attrs=(("name", weight_name),),
            )
            eclass_id = self.egraph.add(enode)
            self._node_to_eclass[weight_name] = eclass_id

            if weight_meta and weight_meta.shape:
                self._shape_info[eclass_id] = ShapeInfo(
                    shape=list(weight_meta.shape),
                    dtype=weight_meta.dtype.value if weight_meta.dtype else "float32",
                )

        # 按拓扑顺序添加操作节点
        for node_id in ir.topological_order():
            if node_id in self._node_to_eclass:
                continue

            node = ir.nodes.get(node_id)
            if node is None:
                continue

            # 获取子节点的 E-Class ID
            children = []
            for child_id in node.children:
                if child_id in self._node_to_eclass:
                    children.append(self._node_to_eclass[child_id])
                else:
                    # 可能是常量或未知节点
                    const_node = ENode(
                        op="const",
                        children=(),
                        attrs=(("value", child_id),),
                    )
                    const_id = self.egraph.add(const_node)
                    self._node_to_eclass[child_id] = const_id
                    children.append(const_id)

            # 创建 E-Node
            attrs = tuple(node.attrs_dict.items()) if node.attrs_dict else ()
            enode = ENode(
                op=node.op,
                children=tuple(children),
                attrs=attrs,
            )
            eclass_id = self.egraph.add(enode)
            self._node_to_eclass[node_id] = eclass_id

            # 记录形状信息
            if node.output_meta and node.output_meta.shape:
                self._shape_info[eclass_id] = ShapeInfo(
                    shape=list(node.output_meta.shape),
                    dtype=node.output_meta.dtype.value if node.output_meta.dtype else "float32",
                )

        # 获取输出节点的 E-Class ID
        root_id = 0
        if ir.outputs:
            output_id = ir.outputs[0]
            if output_id in self._node_to_eclass:
                root_id = self._node_to_eclass[output_id]

        return root_id, self._shape_info

    @property
    def node_mapping(self) -> Dict[str, int]:
        """获取节点到 E-Class 的映射"""
        return self._node_to_eclass


# =============================================================================
# E-Graph to Subgraph Converter
# =============================================================================

class EGraphToSubgraph:
    """将 E-Graph 表达式转换回 SubgraphIR"""

    def __init__(self, egraph: EGraph, original_ir: SubgraphIR):
        self.egraph = egraph
        self.original_ir = original_ir
        self._counter = 0

    def convert(
        self,
        expr: ExtractedExpr,
        variant_id: str
    ) -> SubgraphIR:
        """转换提取的表达式为 SubgraphIR"""
        nodes: Dict[str, OpNode] = {}

        # 递归构建节点
        output_id = self._build_node(
            expr.root_eclass, expr.enode_choices, nodes)

        # 创建新的 IR
        new_ir = SubgraphIR(
            id=variant_id,
            nodes=nodes,
            inputs=self.original_ir.inputs.copy(),
            outputs=[output_id],
            weights=self.original_ir.weights.copy(),
            metadata={
                **self.original_ir.metadata,
                "transforms": expr.proof,
                "cost_estimate": expr.cost.breakdown if expr.cost else {},
            },
        )

        return new_ir

    def _build_node(
        self,
        eclass_id: int,
        choices: Dict[int, 'ENode'],
        nodes: Dict[str, OpNode]
    ) -> str:
        """递归构建节点"""
        enode = choices.get(eclass_id)

        if enode is None:
            # 没有选择，尝试从 E-Class 获取
            eclass = self.egraph.get_eclass(eclass_id)
            if eclass and eclass.nodes:
                enode = next(iter(eclass.nodes))
            else:
                # 创建占位符
                node_id = f"unknown_{self._counter}"
                self._counter += 1
                nodes[node_id] = OpNode(
                    op="unknown",
                    children=(),
                    attrs=(),
                )
                return node_id

        # 检查是否是输入或权重
        if enode.op == "input":
            for attr in enode.attrs:
                if attr[0] == "name":
                    return attr[1]
            return "input_0"

        if enode.op == "weight":
            weight_name = None
            for attr in enode.attrs:
                if attr[0] == "name":
                    weight_name = attr[1]
                    break

            # 如果权重名称看起来像数字（如 1e-05），创建一个常量节点
            if weight_name and (weight_name[0].isdigit() or weight_name.startswith('-')):
                node_id = f"const_{self._counter}"
                self._counter += 1
                try:
                    const_value = float(weight_name)
                except ValueError:
                    const_value = weight_name
                nodes[node_id] = OpNode(
                    op="const",
                    children=(),
                    attrs=(("value", const_value),),
                )
                return node_id

            # 正常的权重 - 创建权重节点
            node_id = f"w_{self._counter}"
            self._counter += 1
            nodes[node_id] = OpNode(
                op="weight",
                children=(),
                attrs=(("name", weight_name),) if weight_name else (),
            )
            return node_id

        if enode.op == "const":
            # 为常量创建节点
            const_value = None
            for attr in enode.attrs:
                if attr[0] == "value":
                    const_value = attr[1]
                    break

            node_id = f"const_{self._counter}"
            self._counter += 1
            nodes[node_id] = OpNode(
                op="const",
                children=(),
                attrs=(("value", const_value),
                       ) if const_value is not None else (),
            )
            return node_id

        # 递归构建子节点
        children = []
        for child_id in enode.children:
            # 确保使用规范的 E-Class ID
            canonical_child = self.egraph.find(child_id)
            child_name = self._build_node(canonical_child, choices, nodes)
            children.append(child_name)

        # 创建节点
        node_id = f"n_{self._counter}"
        self._counter += 1

        nodes[node_id] = OpNode(
            op=enode.op,
            children=tuple(children),
            attrs=enode.attrs,
        )

        return node_id


# =============================================================================
# Algebraic Optimizer
# =============================================================================

class AlgebraicOptimizer:
    """
    通用代数优化器

    基于 E-Graph 和声明式规则实现自动优化探索
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config

        # 加载规则
        self.rules = get_rule_set(config.rule_set)
        self.rules.extend(config.custom_rules)

        # 初始化代价模型
        self._cost_model = get_cost_model(config.cost_model)

    def optimize(self, ir: SubgraphIR) -> OptimizationResult:
        """
        优化单个子图

        Args:
            ir: 输入的 SubgraphIR

        Returns:
            OptimizationResult 包含所有变体
        """
        start_time = time.time()
        stats = {
            "original_nodes": len(ir.nodes),
            "rules_applied": 0,
            "iterations": 0,
            "eclasses_created": 0,
            "enodes_created": 0,
        }
        applied_rules: List[RuleApplication] = []

        # Step 1: 构建 E-Graph
        egraph = EGraph()
        converter = SubgraphToEGraph(egraph)
        root_id, shape_info = converter.convert(ir)

        if root_id == 0:
            # 转换失败
            return OptimizationResult(
                original_ir=ir,
                variants=[(ir, ExtractedExpr(0, {}, Cost.zero(), []))],
                stats=stats,
                applied_rules=[],
            )

        stats["eclasses_created"] = len(list(egraph.iter_eclasses()))

        # Step 2: 初始化匹配引擎和约束检查器
        match_engine = EMatchEngine(egraph)
        constraint_checker = ConstraintChecker(egraph, shape_info)

        # Step 3: 饱和搜索 - 应用规则直到饱和
        for iteration in range(self.config.max_iterations):
            stats["iterations"] = iteration + 1

            # 检查时间预算
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.config.time_budget_ms:
                stats["stopped_reason"] = "time_budget"
                break

            # 检查大小限制
            num_eclasses = len(list(egraph.iter_eclasses()))
            if num_eclasses > self.config.max_eclasses:
                stats["stopped_reason"] = "max_eclasses"
                break

            new_applications = 0

            # 尝试应用每个规则
            for rule in self.rules:
                matches = match_engine.find_matches(rule)

                for match in matches:
                    # 检查约束条件
                    if rule.condition:
                        if not constraint_checker.check(rule.condition, match.bindings):
                            continue

                    # 应用重写
                    try:
                        rewrite_pattern = rule.rewrite_ast
                        new_id = match_engine.apply_rewrite(
                            rewrite_pattern, match.bindings)

                        # 合并等价类
                        if new_id != match.eclass_id:
                            egraph.merge(match.eclass_id, new_id)
                            new_applications += 1
                            stats["rules_applied"] += 1

                            applied_rules.append(RuleApplication(
                                rule=rule,
                                original_eclass=match.eclass_id,
                                new_eclass=new_id,
                                bindings=match.bindings,
                                proof=f"{rule.name}: {rule.description}",
                            ))
                    except Exception as e:
                        # 应用失败，跳过
                        continue

            # 重建 E-Graph
            egraph.rebuild()
            match_engine.clear_cache()

            if new_applications == 0:
                stats["stopped_reason"] = "saturated"
                break

        stats["eclasses_final"] = len(list(egraph.iter_eclasses()))

        # Step 4: 提取 Top-K 候选
        extractor = KBestExtractor(egraph, self._cost_model)
        candidates = extractor.extract(root_id, k=self.config.top_k)

        # Step 5: 转换回 SubgraphIR
        back_converter = EGraphToSubgraph(egraph, ir)
        variants = []

        for i, expr in enumerate(candidates):
            variant_id = f"{ir.id}_v{i}" if i > 0 else ir.id
            variant_ir = back_converter.convert(expr, variant_id)
            variants.append((variant_ir, expr))

        stats["variants_generated"] = len(variants) - 1  # 不计原始
        stats["processing_time_ms"] = int((time.time() - start_time) * 1000)

        return OptimizationResult(
            original_ir=ir,
            variants=variants,
            stats=stats,
            applied_rules=applied_rules,
        )

    def optimize_json(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        优化 subgraphs.json 文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            统计信息
        """
        from .fx_expander import expand_subgraph_json

        # 读取输入
        with open(input_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        all_subgraphs = []
        total_stats = {
            "original_count": len(items),
            "total_variants": 0,
            "by_rule": {},
            "processing_time_ms": 0,
        }

        start_time = time.time()
        seen_signatures: Set[str] = set()

        for item in items:
            try:
                # 展开高级操作
                if self.config.expand_ops:
                    expanded_item = expand_subgraph_json(
                        item,
                        expand_norms=self.config.expand_norms
                    )
                else:
                    expanded_item = item

                # 解析为 IR
                ir = parse_subgraph_json(expanded_item)

                # 优化
                result = self.optimize(ir)

                # 添加变体
                for variant_ir, expr in result.variants:
                    variant_json = ir_to_json(variant_ir)

                    # 去重
                    if self.config.deduplicate:
                        sig = self._compute_signature(variant_json)
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)

                    # 添加元信息
                    if expr.cost:
                        variant_json["cost_estimate"] = expr.cost.breakdown
                    if self.config.include_proof and expr.proof:
                        variant_json["proof"] = expr.proof

                    all_subgraphs.append(variant_json)

                    if variant_ir.id != ir.id:
                        total_stats["total_variants"] += 1

                # 统计规则应用
                for app in result.applied_rules:
                    rule_name = app.rule.name
                    total_stats["by_rule"][rule_name] = \
                        total_stats["by_rule"].get(rule_name, 0) + 1

            except Exception as e:
                print(f"Warning: Failed to optimize {item.get('id')}: {e}")
                if self.config.include_original:
                    all_subgraphs.append(item)

        total_stats["processing_time_ms"] = int(
            (time.time() - start_time) * 1000)
        total_stats["output_count"] = len(all_subgraphs)

        # 写入输出
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_subgraphs, f, indent=2)

        return total_stats

    def _compute_signature(self, item: Dict[str, Any]) -> str:
        """计算子图签名（用于去重）"""
        import hashlib
        h = hashlib.md5()

        ops = item.get("ops", [])
        for op in ops:
            h.update(str(op.get("op", "")).encode())
            h.update(str(op.get("inputs", [])).encode())

        return h.hexdigest()[:16]


# =============================================================================
# Convenience Functions
# =============================================================================

def optimize_subgraph(
    ir: SubgraphIR,
    rule_set: str = "default",
    top_k: int = 5,
) -> List[SubgraphIR]:
    """
    便捷函数：优化单个子图

    Args:
        ir: 输入 SubgraphIR
        rule_set: 规则集名称
        top_k: 返回 Top-K 变体

    Returns:
        变体列表
    """
    config = OptimizerConfig(
        rule_set=rule_set,
        top_k=top_k,
    )
    optimizer = AlgebraicOptimizer(config)
    result = optimizer.optimize(ir)
    return [v[0] for v in result.variants]


def optimize_json_file(
    input_path: str,
    output_path: str,
    rule_set: str = "default",
) -> Dict[str, Any]:
    """
    便捷函数：优化 JSON 文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        rule_set: 规则集名称

    Returns:
        统计信息
    """
    config = OptimizerConfig(rule_set=rule_set)
    optimizer = AlgebraicOptimizer(config)
    return optimizer.optimize_json(Path(input_path), Path(output_path))
