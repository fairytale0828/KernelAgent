# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
代数重写引擎主入口

提供完整的重写 pipeline:
1. 解析 subgraphs.json
2. 展开高级操作为低级代数操作 (使用 torch.fx)
3. 构建 E-Graph
4. 应用代数规则进行饱和搜索
5. 提取 Top-K 候选
6. 输出 transformed_subgraphs.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ir import SubgraphIR, parse_subgraph_json, ir_to_json, OpNode, TensorMeta
from .egraph import EGraph, ENode, Saturator, SaturationConfig, PatternRewriteRule
from .cost_model import CostModel, IOAwareCostModel, get_cost_model, Cost
from .extractor import Extractor, ExtractedExpr, KBestExtractor
from .rules import RuleSet, Rule, RuleGroup
from .fx_expander import expand_subgraph_json, FXExpander


@dataclass
class RewriteConfig:
    """重写配置"""
    # 规则配置
    rule_set: str = "default"  # "default", "full", "minimal"
    enabled_groups: Optional[List[str]] = None
    disabled_groups: Optional[List[str]] = None

    # 展开配置
    expand_ops: bool = True  # 是否展开高级操作
    expand_norms: bool = True  # 是否展开归一化操作

    # 饱和配置
    max_iterations: int = 30
    max_eclasses: int = 10000
    max_enodes: int = 100000
    time_budget_ms: int = 10000

    # 提取配置
    top_k: int = 5
    cost_model: str = "io_aware"  # "io_aware", "flops", "balanced"

    # 输出配置
    include_original: bool = True
    include_proof: bool = True
    deduplicate: bool = True


@dataclass
class RewriteResult:
    """重写结果"""
    original_count: int
    total_variants: int
    subgraphs: List[Dict[str, Any]]
    stats: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({
            "original_count": self.original_count,
            "total_variants": self.total_variants,
            "stats": self.stats,
        }, indent=2)


class SubgraphToEGraph:
    """
    将 SubgraphIR 转换为 E-Graph
    """

    def __init__(self, egraph: EGraph):
        self.egraph = egraph
        self.node_to_eclass: Dict[str, int] = {}

    def convert(self, ir: SubgraphIR) -> int:
        """
        转换子图为 E-Graph，返回根 EClass ID
        """
        # 添加输入节点
        for inp_name, inp_meta in ir.inputs:
            enode = ENode(
                op="input",
                children=(),
                attrs=(("name", inp_name), ("shape", inp_meta.shape)),
            )
            self.node_to_eclass[inp_name] = self.egraph.add(enode)

        # 按拓扑顺序添加节点
        for node_id in ir.topological_order():
            node = ir.nodes[node_id]

            # 获取子节点的 EClass ID
            child_eclasses = tuple(
                self.node_to_eclass.get(c, 0) for c in node.children
            )

            # 创建 ENode
            enode = ENode(
                op=node.op,
                children=child_eclasses,
                attrs=node.attrs,
            )

            self.node_to_eclass[node_id] = self.egraph.add(enode)

        # 返回输出节点的 EClass ID
        if ir.outputs:
            return self.node_to_eclass.get(ir.outputs[0], 0)
        return 0


class EGraphToSubgraph:
    """
    将 E-Graph 表达式转换回 SubgraphIR
    """

    def __init__(self, egraph: EGraph, original_ir: SubgraphIR):
        self.egraph = egraph
        self.original_ir = original_ir

    def convert(self, expr: ExtractedExpr, variant_id: str) -> SubgraphIR:
        """
        将提取的表达式转换为 SubgraphIR
        """
        nodes: Dict[str, OpNode] = {}
        eclass_to_node_id: Dict[int, str] = {}

        # 按拓扑顺序处理
        order = self._topological_sort(expr.enode_choices, expr.root_eclass)

        for i, eclass_id in enumerate(order):
            enode = expr.enode_choices.get(eclass_id)
            if enode is None:
                continue

            if enode.op == "input":
                # 输入节点不需要创建 OpNode
                inp_name = enode.get_attr("name", f"input_{i}")
                eclass_to_node_id[eclass_id] = inp_name
                continue

            node_id = f"n_{i}"
            eclass_to_node_id[eclass_id] = node_id

            # 获取子节点 ID
            children = tuple(
                eclass_to_node_id.get(self.egraph.find(c), f"unknown_{c}")
                for c in enode.children
            )

            # 创建 OpNode
            op_node = OpNode(
                op=enode.op,
                children=children,
                attrs=enode.attrs,
                output_meta=None,  # 可以从原始 IR 推导
            )
            nodes[node_id] = op_node

        # 确定输出
        outputs = [eclass_to_node_id.get(expr.root_eclass, "")]

        return SubgraphIR(
            id=variant_id,
            nodes=nodes,
            inputs=self.original_ir.inputs,
            outputs=outputs,
            weights=self.original_ir.weights,
            metadata={
                **self.original_ir.metadata,
                "transforms": expr.proof,
                "cost_estimate": expr.cost.breakdown,
                "variant_of": self.original_ir.id,
            },
        )

    def _topological_sort(
        self,
        choices: Dict[int, ENode],
        root: int
    ) -> List[int]:
        """拓扑排序"""
        visited = set()
        order = []

        def visit(eclass_id: int):
            eclass_id = self.egraph.find(eclass_id)
            if eclass_id in visited:
                return
            visited.add(eclass_id)

            enode = choices.get(eclass_id)
            if enode:
                for child_id in enode.children:
                    visit(child_id)

            order.append(eclass_id)

        visit(root)
        return order


class RuleCompiler:
    """
    将声明式规则编译为可执行的重写规则
    """

    def __init__(self):
        self._pattern_cache: Dict[str, Any] = {}

    def compile(self, rule: Rule) -> PatternRewriteRule:
        """
        编译规则为 PatternRewriteRule

        注意：这是一个简化实现，实际需要完整的模式解析器
        """
        # 解析 LHS 模式
        lhs_pattern = self._parse_pattern(rule.lhs_pattern)

        # 创建 RHS 构建器
        rhs_builder = self._create_rhs_builder(rule.rhs_pattern)

        # 创建条件检查器
        condition = None
        if rule.condition:
            def condition(
                eg, bindings, cond=rule.condition): return cond.check(bindings)

        return PatternRewriteRule(
            name=rule.name,
            lhs=lhs_pattern,
            rhs_builder=rhs_builder,
            condition=condition,
            priority=rule.priority,
        )

    def _parse_pattern(self, pattern_str: str) -> Any:
        """解析模式字符串"""
        # 简化实现：返回模式字符串本身
        # 实际需要解析为 Pattern 对象
        return pattern_str

    def _create_rhs_builder(
        self,
        rhs_pattern: str
    ) -> Callable[[EGraph, Dict[str, int]], int]:
        """创建 RHS 构建器"""
        def builder(egraph: EGraph, bindings: Dict[str, int]) -> int:
            # 简化实现：直接返回绑定的第一个值
            # 实际需要根据 RHS 模式构建新的表达式
            if bindings:
                return next(iter(bindings.values()))
            return 0
        return builder


class AlgebraicRewriter:
    """
    代数重写引擎主类

    Usage:
        rewriter = AlgebraicRewriter(config)
        result = rewriter.rewrite(input_path, output_path)
    """

    def __init__(self, config: Optional[RewriteConfig] = None):
        self.config = config or RewriteConfig()
        self._rule_set = self._build_rule_set()
        self._cost_model = get_cost_model(self.config.cost_model)

    def _build_rule_set(self) -> RuleSet:
        """构建规则集"""
        if self.config.rule_set == "full":
            rs = RuleSet.full()
        elif self.config.rule_set == "minimal":
            rs = RuleSet("minimal")
            rs.enable_group(RuleGroup.G1_BASIC_ALGEBRA)
        else:
            rs = RuleSet.default()

        # 处理启用/禁用的组
        if self.config.enabled_groups:
            for g_name in self.config.enabled_groups:
                try:
                    g = RuleGroup[g_name]
                    rs.enable_group(g)
                except KeyError:
                    pass

        if self.config.disabled_groups:
            for g_name in self.config.disabled_groups:
                try:
                    g = RuleGroup[g_name]
                    rs.disable_group(g)
                except KeyError:
                    pass

        return rs

    def rewrite_subgraph(
        self,
        ir: SubgraphIR
    ) -> List[Tuple[SubgraphIR, ExtractedExpr]]:
        """
        重写单个子图

        Returns:
            (变体 IR, 提取表达式) 列表
        """
        # 创建 E-Graph
        egraph = EGraph()

        # 转换子图到 E-Graph
        converter = SubgraphToEGraph(egraph)
        root_eclass = converter.convert(ir)

        if root_eclass == 0:
            return [(ir, ExtractedExpr(0, {}, Cost.zero(), []))]

        # 应用内置的代数变换规则
        self._apply_builtin_rules(egraph, ir)

        # 配置饱和搜索
        sat_config = SaturationConfig(
            max_iterations=self.config.max_iterations,
            max_eclasses=self.config.max_eclasses,
            max_enodes=self.config.max_enodes,
            time_budget_ms=self.config.time_budget_ms,
        )

        # 运行饱和搜索
        saturator = Saturator(egraph, sat_config)
        # 注意：这里需要将声明式规则转换为可执行规则
        # 简化实现中，我们使用内置的变换
        saturator.run()

        # 提取 Top-K 候选
        extractor = KBestExtractor(egraph, self._cost_model)
        candidates = extractor.extract(root_eclass, k=self.config.top_k)

        # 转换回 SubgraphIR
        results = []
        back_converter = EGraphToSubgraph(egraph, ir)

        for i, expr in enumerate(candidates):
            variant_id = f"{ir.id}_v{i}" if i > 0 else ir.id
            variant_ir = back_converter.convert(expr, variant_id)
            results.append((variant_ir, expr))

        return results

    def _apply_builtin_rules(self, egraph: EGraph, ir: SubgraphIR) -> None:
        """
        应用内置的代数变换规则

        这些规则直接操作 E-Graph，不依赖模式匹配
        """
        # 收集所有需要应用的变换
        transforms_applied = []

        # 遍历所有等价类，应用简单的代数变换
        # 注意：不使用 merge，而是记录等价关系，让提取器选择
        for eclass in list(egraph.iter_eclasses()):
            for enode in list(eclass.nodes):
                # 交换律：添加交换后的版本到同一 EClass
                if enode.op in ("add", "mul", "max", "min"):
                    if len(enode.children) == 2:
                        swapped = ENode(
                            op=enode.op,
                            children=(enode.children[1], enode.children[0]),
                            attrs=enode.attrs,
                        )
                        # 直接添加到同一 eclass
                        eclass.nodes.add(swapped)
                        transforms_applied.append(f"commute:{enode.op}")

                # RMSNorm + MatMul 融合：创建变体
                if enode.op == "matmul":
                    variant = self._create_rmsnorm_matmul_variant(
                        egraph, eclass.id, enode)
                    if variant:
                        # 添加变体到同一 eclass
                        eclass.nodes.add(variant)
                        transforms_applied.append("rmsnorm_matmul_pushdown")

                # MatMul + Bias 融合
                if enode.op == "add":
                    variant = self._create_matmul_bias_variant(
                        egraph, eclass.id, enode)
                    if variant:
                        eclass.nodes.add(variant)
                        transforms_applied.append("matmul_bias_fusion")

        # 记录应用的变换
        if transforms_applied:
            egraph.stats["transforms"] = transforms_applied

    def _apply_associativity(
        self,
        egraph: EGraph,
        eclass_id: int,
        enode: ENode
    ) -> None:
        """应用结合律"""
        if len(enode.children) != 2:
            return

        left_id, right_id = enode.children
        left_eclass = egraph.get_eclass(left_id)

        if left_eclass is None:
            return

        # 检查左子节点是否是相同操作
        for left_enode in left_eclass.nodes:
            if left_enode.op == enode.op and len(left_enode.children) == 2:
                # (a op b) op c -> a op (b op c)
                a, b = left_enode.children
                c = right_id

                # 创建 b op c
                bc = ENode(op=enode.op, children=(b, c), attrs=enode.attrs)
                bc_id = egraph.add(bc)

                # 创建 a op (b op c)
                new_expr = ENode(op=enode.op, children=(
                    a, bc_id), attrs=enode.attrs)
                new_id = egraph.add(new_expr)

                egraph.merge(eclass_id, new_id)
                break

    def _create_matmul_bias_variant(
        self,
        egraph: EGraph,
        eclass_id: int,
        add_node: ENode
    ) -> Optional[ENode]:
        """创建 MatMul + Bias 融合变体"""
        if len(add_node.children) != 2:
            return None

        for i, child_id in enumerate(add_node.children):
            child_eclass = egraph.get_eclass(child_id)
            if child_eclass is None:
                continue

            for child_enode in child_eclass.nodes:
                if child_enode.op == "matmul":
                    # 找到 matmul + bias 模式
                    bias_id = add_node.children[1 - i]

                    # 创建融合节点
                    return ENode(
                        op="matmul_bias",
                        children=child_enode.children + (bias_id,),
                        attrs=child_enode.attrs,
                    )
        return None

    def _create_rmsnorm_matmul_variant(
        self,
        egraph: EGraph,
        eclass_id: int,
        matmul_node: ENode
    ) -> Optional[ENode]:
        """
        创建 RMSNorm + MatMul 融合变体

        变换: matmul(mul(x, rsqrt), W) -> mul(matmul(x, W), rsqrt)
        这样可以避免创建中间张量 (x * rsqrt)
        """
        if len(matmul_node.children) < 2:
            return None

        input_id = matmul_node.children[0]
        weight_id = matmul_node.children[1]
        input_eclass = egraph.get_eclass(input_id)

        if input_eclass is None:
            return None

        for input_enode in input_eclass.nodes:
            # 检查是否是 mul(x, rsqrt(...))
            if input_enode.op == "mul" and len(input_enode.children) == 2:
                # 检查两个子节点，找出哪个是 rsqrt
                for j, child_id in enumerate(input_enode.children):
                    child_eclass = egraph.get_eclass(child_id)
                    if child_eclass:
                        for child_enode in child_eclass.nodes:
                            if child_enode.op == "rsqrt":
                                x_id = input_enode.children[1 - j]
                                rsqrt_id = child_id

                                # 创建变体: mul(matmul(x, W), rsqrt)
                                # 这是一个复合表达式，用特殊标记
                                return ENode(
                                    op="rmsnorm_matmul_fused",
                                    children=(x_id, weight_id, rsqrt_id),
                                    attrs=(
                                        ("transform", "rmsnorm_pushdown"),
                                        ("original_pattern",
                                         "matmul(mul(x, rsqrt), W)"),
                                        ("optimized_pattern",
                                         "mul(matmul(x, W), rsqrt)"),
                                    ),
                                )

            # 检查是否是 div(x, norm)
            if input_enode.op == "div" and len(input_enode.children) == 2:
                x_id, norm_id = input_enode.children

                # 创建变体: div(matmul(x, W), norm)
                return ENode(
                    op="rmsnorm_matmul_fused",
                    children=(x_id, weight_id, norm_id),
                    attrs=(
                        ("transform", "rmsnorm_div_pushdown"),
                        ("original_pattern", "matmul(div(x, norm), W)"),
                        ("optimized_pattern", "div(matmul(x, W), norm)"),
                    ),
                )

        return None

    def rewrite(
        self,
        input_path: Path | str,
        output_path: Path | str
    ) -> RewriteResult:
        """
        重写 subgraphs.json

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            重写结果
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        start_time = time.time()

        # 读取输入
        with input_path.open("r", encoding="utf-8") as f:
            original_items = json.load(f)

        if not isinstance(original_items, list):
            raise ValueError("subgraphs.json must be a JSON array")

        all_subgraphs: List[Dict[str, Any]] = []
        stats = {
            "original_count": len(original_items),
            "total_variants": 0,
            "by_transform": {},
            "processing_time_ms": 0,
            "egraph_stats": {},
        }

        seen_signatures: set = set()

        for item in original_items:
            try:
                # Step 1: 展开高级操作为低级代数操作
                if self.config.expand_ops:
                    expanded_item = expand_subgraph_json(
                        item,
                        expand_norms=self.config.expand_norms
                    )
                    stats["expansion_applied"] = True

                    # 记录展开信息
                    orig_ops = len(item.get("ops", []))
                    new_ops = len(expanded_item.get("ops", []))
                    if new_ops > orig_ops:
                        stats.setdefault("expansions", []).append({
                            "id": item.get("id"),
                            "original_ops": orig_ops,
                            "expanded_ops": new_ops
                        })
                else:
                    expanded_item = item

                # Step 2: 解析子图
                ir = parse_subgraph_json(expanded_item)

                # 保留原始
                if self.config.include_original:
                    original_json = ir_to_json(ir)
                    sig = self._compute_signature(original_json)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        all_subgraphs.append(original_json)

                # Step 3: 重写
                variants = self.rewrite_subgraph(ir)

                # 添加变体
                for variant_ir, expr in variants:
                    if variant_ir.id == ir.id:
                        continue  # 跳过原始

                    variant_json = ir_to_json(variant_ir)

                    # 去重
                    if self.config.deduplicate:
                        sig = self._compute_signature(variant_json)
                        if sig in seen_signatures:
                            continue
                        seen_signatures.add(sig)

                    # 添加代价估计
                    variant_json["cost_estimate"] = expr.cost.breakdown

                    # 添加证明
                    if self.config.include_proof:
                        variant_json["proof"] = expr.proof

                    all_subgraphs.append(variant_json)
                    stats["total_variants"] += 1

                    # 统计变换类型
                    for transform in variant_ir.metadata.get("transforms", []):
                        transform_type = transform.split(
                            ":")[0] if ":" in transform else "unknown"
                        stats["by_transform"][transform_type] = \
                            stats["by_transform"].get(transform_type, 0) + 1

            except Exception as e:
                print(
                    f"Warning: Failed to process subgraph {item.get('id')}: {e}")
                # 保留原始
                if self.config.include_original:
                    all_subgraphs.append(item)
                continue

        stats["processing_time_ms"] = int((time.time() - start_time) * 1000)

        # 写入输出
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_subgraphs, f, indent=2)

        return RewriteResult(
            original_count=stats["original_count"],
            total_variants=stats["total_variants"],
            subgraphs=all_subgraphs,
            stats=stats,
        )

    def _compute_signature(self, item: Dict[str, Any]) -> str:
        """计算子图签名用于去重"""
        # 基于 ops 和形状计算签名
        ops = item.get("ops", [])
        ops_str = json.dumps(ops, sort_keys=True)
        input_shape = item.get("input_shape", [])
        output_shape = item.get("output_shape", [])

        import hashlib
        h = hashlib.sha256()
        h.update(ops_str.encode())
        h.update(str(input_shape).encode())
        h.update(str(output_shape).encode())

        return h.hexdigest()[:32]


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="E-Graph 代数重写引擎"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入 subgraphs.json 路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出路径"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="每个子图输出 Top-K 候选 (默认: 5)"
    )
    parser.add_argument(
        "--cost-model",
        choices=["io_aware", "flops", "balanced"],
        default="io_aware",
        help="代价模型 (默认: io_aware)"
    )
    parser.add_argument(
        "--rule-set",
        choices=["default", "full", "minimal"],
        default="default",
        help="规则集 (默认: default, 启用 G1-G4)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="饱和搜索最大迭代次数"
    )
    parser.add_argument(
        "--time-budget-ms",
        type=int,
        default=10000,
        help="时间预算 (毫秒)"
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="不包含原始子图"
    )
    parser.add_argument(
        "--no-proof",
        action="store_true",
        help="不包含变换证明"
    )

    args = parser.parse_args()

    config = RewriteConfig(
        rule_set=args.rule_set,
        top_k=args.top_k,
        cost_model=args.cost_model,
        max_iterations=args.max_iterations,
        time_budget_ms=args.time_budget_ms,
        include_original=not args.no_original,
        include_proof=not args.no_proof,
    )

    rewriter = AlgebraicRewriter(config)
    result = rewriter.rewrite(args.input, args.output)

    print(f"✓ 代数重写完成")
    print(f"  原始子图: {result.original_count}")
    print(f"  生成变体: {result.total_variants}")
    print(f"  总输出: {len(result.subgraphs)}")
    print(f"  处理时间: {result.stats['processing_time_ms']}ms")

    if result.stats.get("by_transform"):
        print(f"  按变换类型:")
        for transform, count in result.stats["by_transform"].items():
            print(f"    - {transform}: {count}")


if __name__ == "__main__":
    main()
