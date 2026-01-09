# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
规则系统基础设施
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..egraph import EGraph, ENode, Pattern


class RuleGroup(Enum):
    """规则分组"""
    G1_BASIC_ALGEBRA = auto()      # 基础代数
    G2_STRUCTURAL = auto()          # 结构简化
    G3_REDUCTION_LINEAR = auto()    # 归约线性
    G4_MATMUL = auto()              # MatMul 相关
    G5_SOFTMAX = auto()             # Softmax 展开/折叠
    G6_DISTRIBUTION = auto()        # 分配律


@dataclass
class RuleCondition:
    """规则应用条件"""
    # 形状条件
    shape_compatible: bool = True
    # 维度条件
    axis_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    # dtype 条件
    dtype_condition: Optional[Callable[[str, str], bool]] = None
    # 自定义条件
    custom: Optional[Callable[[Any], bool]] = None
    
    def check(self, context: Dict[str, Any]) -> bool:
        """检查所有条件"""
        if self.axis_condition and not self.axis_condition(context):
            return False
        if self.custom and not self.custom(context):
            return False
        return True


@dataclass
class CostHint:
    """代价提示"""
    flops_delta: float = 0.0      # FLOPs 变化 (正=增加)
    io_delta: float = 0.0         # Global IO 变化
    temp_delta: float = 0.0       # 临时张量变化
    kernel_delta: int = 0         # Kernel 数量变化
    
    @property
    def is_beneficial(self) -> bool:
        """是否有益 (启发式)"""
        # IO 减少通常更重要
        if self.io_delta < 0:
            return True
        if self.temp_delta < 0 and self.flops_delta <= 0:
            return True
        if self.kernel_delta < 0:
            return True
        return False


@dataclass
class Rule:
    """
    代数规则
    
    定义一个等价变换规则
    """
    name: str
    group: RuleGroup
    lhs_pattern: str  # 模式字符串表示
    rhs_pattern: str  # 变换后的模式
    condition: Optional[RuleCondition] = None
    cost_hint: Optional[CostHint] = None
    bidirectional: bool = False  # 是否双向
    priority: int = 0  # 优先级 (越高越先应用)
    soundness: str = ""  # 正确性说明
    
    def __hash__(self):
        return hash(self.name)


class RuleBuilder:
    """规则构建器 - 流式 API"""
    
    def __init__(self, name: str):
        self._name = name
        self._group = RuleGroup.G1_BASIC_ALGEBRA
        self._lhs = ""
        self._rhs = ""
        self._condition: Optional[RuleCondition] = None
        self._cost_hint: Optional[CostHint] = None
        self._bidirectional = False
        self._priority = 0
        self._soundness = ""
    
    def group(self, g: RuleGroup) -> "RuleBuilder":
        self._group = g
        return self
    
    def pattern(self, lhs: str, rhs: str) -> "RuleBuilder":
        self._lhs = lhs
        self._rhs = rhs
        return self
    
    def condition(self, cond: RuleCondition) -> "RuleBuilder":
        self._condition = cond
        return self
    
    def cost(self, hint: CostHint) -> "RuleBuilder":
        self._cost_hint = hint
        return self
    
    def bidirectional(self, b: bool = True) -> "RuleBuilder":
        self._bidirectional = b
        return self
    
    def priority(self, p: int) -> "RuleBuilder":
        self._priority = p
        return self
    
    def soundness(self, s: str) -> "RuleBuilder":
        self._soundness = s
        return self
    
    def build(self) -> Rule:
        return Rule(
            name=self._name,
            group=self._group,
            lhs_pattern=self._lhs,
            rhs_pattern=self._rhs,
            condition=self._condition,
            cost_hint=self._cost_hint,
            bidirectional=self._bidirectional,
            priority=self._priority,
            soundness=self._soundness,
        )


class RuleSet:
    """
    规则集合
    
    管理和组织代数规则
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._rules: Dict[str, Rule] = {}
        self._by_group: Dict[RuleGroup, List[Rule]] = {g: [] for g in RuleGroup}
        self._enabled_groups: Set[RuleGroup] = set()
    
    def add(self, rule: Rule) -> "RuleSet":
        """添加规则"""
        self._rules[rule.name] = rule
        self._by_group[rule.group].append(rule)
        return self
    
    def enable_group(self, group: RuleGroup) -> "RuleSet":
        """启用规则组"""
        self._enabled_groups.add(group)
        return self
    
    def disable_group(self, group: RuleGroup) -> "RuleSet":
        """禁用规则组"""
        self._enabled_groups.discard(group)
        return self
    
    def get_enabled_rules(self) -> List[Rule]:
        """获取所有启用的规则"""
        rules = []
        for group in self._enabled_groups:
            rules.extend(self._by_group[group])
        # 按优先级排序
        rules.sort(key=lambda r: -r.priority)
        return rules
    
    def get_rules_by_group(self, group: RuleGroup) -> List[Rule]:
        """获取指定组的规则"""
        return self._by_group.get(group, [])
    
    @classmethod
    def default(cls) -> "RuleSet":
        """创建默认规则集 (G1-G4 启用)"""
        from .basic_algebra import BasicAlgebraRules
        from .structural import StructuralRules
        from .reduction import ReductionLinearRules
        from .matmul import MatmulRules
        
        rs = cls("default")
        
        # 添加各组规则
        for rule in BasicAlgebraRules.all():
            rs.add(rule)
        for rule in StructuralRules.all():
            rs.add(rule)
        for rule in ReductionLinearRules.all():
            rs.add(rule)
        for rule in MatmulRules.all():
            rs.add(rule)
        
        # 默认启用 G1-G4
        rs.enable_group(RuleGroup.G1_BASIC_ALGEBRA)
        rs.enable_group(RuleGroup.G2_STRUCTURAL)
        rs.enable_group(RuleGroup.G3_REDUCTION_LINEAR)
        rs.enable_group(RuleGroup.G4_MATMUL)
        
        return rs
    
    @classmethod
    def full(cls) -> "RuleSet":
        """创建完整规则集 (所有规则启用)"""
        from .basic_algebra import BasicAlgebraRules
        from .structural import StructuralRules
        from .reduction import ReductionLinearRules
        from .matmul import MatmulRules
        from .softmax import SoftmaxRules
        from .distribution import DistributionRules
        
        rs = cls("full")
        
        for rule in BasicAlgebraRules.all():
            rs.add(rule)
        for rule in StructuralRules.all():
            rs.add(rule)
        for rule in ReductionLinearRules.all():
            rs.add(rule)
        for rule in MatmulRules.all():
            rs.add(rule)
        for rule in SoftmaxRules.all():
            rs.add(rule)
        for rule in DistributionRules.all():
            rs.add(rule)
        
        # 启用所有组
        for g in RuleGroup:
            rs.enable_group(g)
        
        return rs
    
    def __len__(self) -> int:
        return len(self._rules)
    
    def __iter__(self):
        return iter(self.get_enabled_rules())
