# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
E-Graph (Equality Graph) 核心实现

E-Graph 是一种高效的数据结构，用于表示和操作等价类。
它支持：
1. 高效的等价类合并 (Union-Find)
2. 基于模式的规则匹配
3. 饱和搜索 (Equality Saturation)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List,
    Optional, Set, Tuple, TypeVar
)

from .ir import OpNode, OpProperty


@dataclass
class ENode:
    """
    E-Node: E-Graph 中的节点

    每个 ENode 代表一个具体的操作，包含：
    - op: 操作类型
    - children: 子节点的 EClass ID 列表
    - attrs: 操作属性
    """
    op: str
    children: Tuple[int, ...]  # EClass IDs
    attrs: Tuple[Tuple[str, Any], ...]

    def __hash__(self) -> int:
        # attrs 中的值可能不可哈希，需要特殊处理
        attrs_hashable = tuple(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in self.attrs
        )
        return hash((self.op, self.children, attrs_hashable))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ENode):
            return False
        return (self.op == other.op and
                self.children == other.children and
                self.attrs == other.attrs)

    @property
    def attrs_dict(self) -> Dict[str, Any]:
        return dict(self.attrs)

    def get_attr(self, key: str, default: Any = None) -> Any:
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    def canonicalize(self, find: Callable[[int], int]) -> "ENode":
        """使用 find 函数规范化子节点引用"""
        new_children = tuple(find(c) for c in self.children)
        return ENode(self.op, new_children, self.attrs)

    @classmethod
    def from_op_node(cls, op_node: OpNode, child_eclasses: Tuple[int, ...]) -> "ENode":
        """从 OpNode 创建 ENode"""
        return cls(
            op=op_node.op,
            children=child_eclasses,
            attrs=op_node.attrs,
        )


@dataclass
class EClass:
    """
    E-Class: 等价类

    包含所有等价的 ENode 表示
    """
    id: int
    nodes: Set[ENode] = field(default_factory=set)
    parents: Set[Tuple[ENode, int]] = field(
        default_factory=set)  # (parent_enode, parent_eclass_id)

    def __hash__(self) -> int:
        return hash(self.id)


class EGraph:
    """
    E-Graph: 等价图

    核心数据结构，支持：
    1. 添加表达式
    2. 合并等价类
    3. 模式匹配
    4. 饱和搜索
    """

    def __init__(self):
        self._eclasses: Dict[int, EClass] = {}
        self._hashcons: Dict[ENode, int] = {}  # ENode -> EClass ID
        self._union_find: Dict[int, int] = {}  # Union-Find parent
        self._next_id: int = 0
        self._pending: List[Tuple[int, int]] = []  # 待处理的合并
        self._dirty: Set[int] = set()  # 需要重建的 EClass

        # 统计信息
        self.stats = {
            "adds": 0,
            "merges": 0,
            "rebuilds": 0,
        }

    @property
    def num_eclasses(self) -> int:
        """返回等价类数量"""
        return len(set(self.find(i) for i in self._eclasses.keys()))

    @property
    def num_enodes(self) -> int:
        """返回 ENode 总数"""
        return sum(len(ec.nodes) for ec in self._eclasses.values())

    def find(self, id: int) -> int:
        """Union-Find: 查找根节点 (带路径压缩)"""
        root = id
        while self._union_find.get(root, root) != root:
            root = self._union_find[root]

        # 路径压缩
        while id != root:
            parent = self._union_find.get(id, id)
            self._union_find[id] = root
            id = parent

        return root

    def _new_eclass(self) -> EClass:
        """创建新的等价类"""
        id = self._next_id
        self._next_id += 1
        eclass = EClass(id=id)
        self._eclasses[id] = eclass
        self._union_find[id] = id
        return eclass

    def add(self, enode: ENode) -> int:
        """
        添加 ENode 到 E-Graph

        Returns:
            EClass ID
        """
        # 规范化子节点引用
        enode = enode.canonicalize(self.find)

        # 检查是否已存在
        if enode in self._hashcons:
            return self.find(self._hashcons[enode])

        # 创建新的 EClass
        eclass = self._new_eclass()
        eclass.nodes.add(enode)
        self._hashcons[enode] = eclass.id

        # 更新父节点引用
        for child_id in enode.children:
            child_eclass = self._eclasses.get(self.find(child_id))
            if child_eclass:
                child_eclass.parents.add((enode, eclass.id))

        self.stats["adds"] += 1
        return eclass.id

    def merge(self, id1: int, id2: int) -> int:
        """
        合并两个等价类

        Returns:
            合并后的 EClass ID
        """
        id1 = self.find(id1)
        id2 = self.find(id2)

        if id1 == id2:
            return id1

        # 总是将较小的合并到较大的
        ec1 = self._eclasses[id1]
        ec2 = self._eclasses[id2]

        if len(ec1.nodes) < len(ec2.nodes):
            id1, id2 = id2, id1
            ec1, ec2 = ec2, ec1

        # 更新 Union-Find
        self._union_find[id2] = id1

        # 合并节点和父节点
        ec1.nodes.update(ec2.nodes)
        ec1.parents.update(ec2.parents)

        # 标记需要重建
        self._dirty.add(id1)
        for _, parent_id in ec1.parents:
            self._dirty.add(self.find(parent_id))

        self.stats["merges"] += 1
        return id1

    def rebuild(self) -> None:
        """
        重建 E-Graph，处理所有待定的规范化
        """
        while self._dirty:
            dirty = list(self._dirty)
            self._dirty.clear()

            for eclass_id in dirty:
                canonical_id = self.find(eclass_id)
                eclass = self._eclasses.get(canonical_id)
                if not eclass:
                    continue

                # 重新规范化所有节点
                new_nodes = set()
                for enode in list(eclass.nodes):  # 复制以避免修改时迭代
                    canonical = enode.canonicalize(self.find)

                    # 检查是否与现有节点冲突
                    if canonical in self._hashcons:
                        existing_id = self.find(self._hashcons[canonical])
                        if existing_id != canonical_id:
                            # 合并等价类，但保留当前节点
                            self.merge(canonical_id, existing_id)
                            # 更新 canonical_id 为合并后的根
                            canonical_id = self.find(canonical_id)
                        new_nodes.add(canonical)
                    else:
                        self._hashcons[canonical] = canonical_id
                        new_nodes.add(canonical)

                # 更新节点集合
                final_eclass = self._eclasses.get(canonical_id)
                if final_eclass:
                    final_eclass.nodes.update(new_nodes)

            self.stats["rebuilds"] += 1

    def get_eclass(self, id: int) -> Optional[EClass]:
        """获取等价类"""
        return self._eclasses.get(self.find(id))

    def iter_eclasses(self) -> Iterator[EClass]:
        """迭代所有等价类 (去重)"""
        seen = set()
        for id in self._eclasses.keys():
            root = self.find(id)
            if root not in seen:
                seen.add(root)
                yield self._eclasses[root]

    def lookup(self, enode: ENode) -> Optional[int]:
        """查找 ENode 对应的 EClass ID"""
        enode = enode.canonicalize(self.find)
        if enode in self._hashcons:
            return self.find(self._hashcons[enode])
        return None

    def add_expr(self, op: str, children: Tuple[int, ...],
                 attrs: Optional[Dict[str, Any]] = None) -> int:
        """便捷方法：添加表达式"""
        attrs_tuple = tuple((attrs or {}).items())
        enode = ENode(op=op, children=children, attrs=attrs_tuple)
        return self.add(enode)

    def ematch(self, pattern: "Pattern") -> List[Dict[str, int]]:
        """
        E-Matching: 在 E-Graph 中匹配模式

        Returns:
            匹配结果列表，每个结果是变量名到 EClass ID 的映射
        """
        return pattern.match(self)


@dataclass
class PatternVar:
    """模式变量"""
    name: str

    def __hash__(self):
        return hash(("var", self.name))


@dataclass
class PatternNode:
    """模式节点"""
    op: str
    children: Tuple["Pattern", ...]
    attrs: Optional[Dict[str, Any]] = None

    def __hash__(self):
        return hash(("node", self.op, self.children))


Pattern = PatternVar | PatternNode


class PatternMatcher:
    """模式匹配器"""

    def __init__(self, egraph: EGraph):
        self.egraph = egraph

    def match(self, pattern: Pattern, eclass_id: int) -> List[Dict[str, int]]:
        """
        匹配模式到指定的等价类

        Returns:
            所有可能的变量绑定列表
        """
        eclass = self.egraph.get_eclass(eclass_id)
        if eclass is None:
            return []

        if isinstance(pattern, PatternVar):
            # 变量匹配任何 EClass
            return [{pattern.name: eclass_id}]

        results = []
        for enode in eclass.nodes:
            if enode.op != pattern.op:
                continue
            if len(enode.children) != len(pattern.children):
                continue

            # 检查属性匹配
            if pattern.attrs:
                enode_attrs = enode.attrs_dict
                if not all(enode_attrs.get(k) == v for k, v in pattern.attrs.items()):
                    continue

            # 递归匹配子节点
            child_matches = self._match_children(
                pattern.children, enode.children)
            results.extend(child_matches)

        return results

    def _match_children(
        self,
        patterns: Tuple[Pattern, ...],
        eclass_ids: Tuple[int, ...]
    ) -> List[Dict[str, int]]:
        """匹配子节点列表"""
        if not patterns:
            return [{}]

        first_pattern = patterns[0]
        first_id = eclass_ids[0]

        first_matches = self.match(first_pattern, first_id)
        if not first_matches:
            return []

        rest_matches = self._match_children(patterns[1:], eclass_ids[1:])
        if not rest_matches:
            return []

        # 组合匹配结果
        results = []
        for fm in first_matches:
            for rm in rest_matches:
                # 检查变量绑定是否一致
                combined = {**fm}
                conflict = False
                for k, v in rm.items():
                    if k in combined and combined[k] != v:
                        conflict = True
                        break
                    combined[k] = v
                if not conflict:
                    results.append(combined)

        return results

    def match_all(self, pattern: Pattern) -> List[Tuple[int, Dict[str, int]]]:
        """
        在整个 E-Graph 中匹配模式

        Returns:
            (EClass ID, 变量绑定) 列表
        """
        results = []
        for eclass in self.egraph.iter_eclasses():
            matches = self.match(pattern, eclass.id)
            for m in matches:
                results.append((eclass.id, m))
        return results


@dataclass
class SaturationConfig:
    """饱和搜索配置"""
    max_iterations: int = 30
    max_eclasses: int = 10000
    max_enodes: int = 100000
    time_budget_ms: int = 10000
    node_limit_per_iter: int = 5000


class Saturator:
    """
    饱和搜索引擎

    应用规则直到达到不动点或超出预算
    """

    def __init__(self, egraph: EGraph, config: Optional[SaturationConfig] = None):
        self.egraph = egraph
        self.config = config or SaturationConfig()
        self.rules: List["RewriteRule"] = []

        # 统计
        self.stats = {
            "iterations": 0,
            "rules_applied": 0,
            "stopped_reason": None,
        }

    def add_rule(self, rule: "RewriteRule") -> None:
        """添加重写规则"""
        self.rules.append(rule)

    def run(self) -> None:
        """
        运行饱和搜索
        """
        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            self.stats["iterations"] = iteration + 1

            # 检查预算
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.config.time_budget_ms:
                self.stats["stopped_reason"] = "time_budget"
                break

            if self.egraph.num_eclasses > self.config.max_eclasses:
                self.stats["stopped_reason"] = "max_eclasses"
                break

            if self.egraph.num_enodes > self.config.max_enodes:
                self.stats["stopped_reason"] = "max_enodes"
                break

            # 应用所有规则
            changes = 0
            for rule in self.rules:
                rule_changes = rule.apply(self.egraph)
                changes += rule_changes
                self.stats["rules_applied"] += rule_changes

            # 重建
            self.egraph.rebuild()

            # 检查不动点
            if changes == 0:
                self.stats["stopped_reason"] = "saturation"
                break
        else:
            self.stats["stopped_reason"] = "max_iterations"


class RewriteRule:
    """
    重写规则基类

    子类需要实现 apply 方法
    """

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority

    def apply(self, egraph: EGraph) -> int:
        """
        应用规则到 E-Graph

        Returns:
            应用的变换数量
        """
        raise NotImplementedError


class PatternRewriteRule(RewriteRule):
    """
    基于模式的重写规则

    lhs -> rhs 形式的规则
    """

    def __init__(
        self,
        name: str,
        lhs: Pattern,
        rhs_builder: Callable[[EGraph, Dict[str, int]], int],
        condition: Optional[Callable[[EGraph, Dict[str, int]], bool]] = None,
        priority: int = 0,
    ):
        super().__init__(name, priority)
        self.lhs = lhs
        self.rhs_builder = rhs_builder
        self.condition = condition

    def apply(self, egraph: EGraph) -> int:
        """应用模式重写规则"""
        matcher = PatternMatcher(egraph)
        matches = matcher.match_all(self.lhs)

        changes = 0
        for eclass_id, bindings in matches:
            # 检查条件
            if self.condition and not self.condition(egraph, bindings):
                continue

            # 构建 RHS
            try:
                rhs_id = self.rhs_builder(egraph, bindings)

                # 合并等价类
                if rhs_id != egraph.find(eclass_id):
                    egraph.merge(eclass_id, rhs_id)
                    changes += 1
            except Exception:
                continue

        return changes
