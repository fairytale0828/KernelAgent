# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
通用模式匹配系统

提供声明式规则定义和 E-Matching 引擎
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum


# =============================================================================
# Pattern AST - 模式的抽象语法树表示
# =============================================================================

class Pattern:
    """模式基类"""
    pass


@dataclass
class VarPattern(Pattern):
    """
    变量模式: ?x, ?y, ?scale

    匹配任意 E-Class，并绑定到变量名
    """
    name: str

    def __repr__(self):
        return f"?{self.name}"


@dataclass
class OpPattern(Pattern):
    """
    操作模式: op(子模式...)

    匹配指定操作类型的 E-Node
    """
    op: str
    children: List[Pattern] = field(default_factory=list)

    def __repr__(self):
        if not self.children:
            return self.op
        children_str = ", ".join(str(c) for c in self.children)
        return f"{self.op}({children_str})"


@dataclass
class WildcardPattern(Pattern):
    """
    通配符模式: *

    匹配任意子树（用于部分匹配）
    """

    def __repr__(self):
        return "*"


@dataclass
class ConstPattern(Pattern):
    """
    常量模式: 123, "string", 1e-5

    匹配特定常量值
    """
    value: Any

    def __repr__(self):
        return repr(self.value)


# =============================================================================
# Pattern Parser - 从字符串解析模式
# =============================================================================

class PatternParser:
    """
    模式解析器

    将字符串格式的模式解析为 Pattern AST

    语法:
        pattern := var | op | const | wildcard
        var := '?' identifier
        op := identifier '(' pattern (',' pattern)* ')'
        const := number | string
        wildcard := '*'
        identifier := [a-zA-Z_][a-zA-Z0-9_]*

    示例:
        "?x"                          -> VarPattern("x")
        "matmul(?x, ?w)"             -> OpPattern("matmul", [VarPattern("x"), VarPattern("w")])
        "mul(matmul(?x, ?w), ?s)"    -> OpPattern("mul", [OpPattern("matmul", ...), VarPattern("s")])
    """

    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> Pattern:
        """解析完整模式"""
        result = self._parse_pattern()
        self._skip_whitespace()
        if self.pos < len(self.text):
            raise ValueError(
                f"Unexpected character at position {self.pos}: '{self.text[self.pos]}'")
        return result

    def _parse_pattern(self) -> Pattern:
        """解析单个模式"""
        self._skip_whitespace()

        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of pattern")

        char = self.text[self.pos]

        # 变量: ?name
        if char == '?':
            return self._parse_var()

        # 通配符: *
        if char == '*':
            self.pos += 1
            return WildcardPattern()

        # 数字常量
        if char.isdigit() or (char == '-' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
            return self._parse_number()

        # 字符串常量
        if char in ('"', "'"):
            return self._parse_string()

        # 操作或标识符
        if char.isalpha() or char == '_':
            return self._parse_op_or_const()

        raise ValueError(
            f"Unexpected character at position {self.pos}: '{char}'")

    def _parse_var(self) -> VarPattern:
        """解析变量 ?name"""
        assert self.text[self.pos] == '?'
        self.pos += 1

        name = self._parse_identifier()
        if not name:
            raise ValueError(
                f"Expected variable name after '?' at position {self.pos}")

        return VarPattern(name)

    def _parse_identifier(self) -> str:
        """解析标识符"""
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return self.text[start:self.pos]

    def _parse_number(self) -> ConstPattern:
        """解析数字常量"""
        start = self.pos

        # 可选负号
        if self.text[self.pos] == '-':
            self.pos += 1

        # 整数部分
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1

        # 小数部分
        if self.pos < len(self.text) and self.text[self.pos] == '.':
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1

        # 科学计数法
        if self.pos < len(self.text) and self.text[self.pos] in ('e', 'E'):
            self.pos += 1
            if self.pos < len(self.text) and self.text[self.pos] in ('+', '-'):
                self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1

        num_str = self.text[start:self.pos]
        try:
            if '.' in num_str or 'e' in num_str.lower():
                return ConstPattern(float(num_str))
            else:
                return ConstPattern(int(num_str))
        except ValueError:
            raise ValueError(f"Invalid number: {num_str}")

    def _parse_string(self) -> ConstPattern:
        """解析字符串常量"""
        quote = self.text[self.pos]
        self.pos += 1
        start = self.pos

        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == '\\':
                self.pos += 2
            else:
                self.pos += 1

        if self.pos >= len(self.text):
            raise ValueError("Unterminated string")

        value = self.text[start:self.pos]
        self.pos += 1  # 跳过结束引号
        return ConstPattern(value)

    def _parse_op_or_const(self) -> Pattern:
        """解析操作或常量标识符"""
        name = self._parse_identifier()
        self._skip_whitespace()

        # 检查是否有参数列表
        if self.pos < len(self.text) and self.text[self.pos] == '(':
            return self._parse_op_args(name)
        else:
            # 无参数的操作（叶子节点）
            return OpPattern(name, [])

    def _parse_op_args(self, op_name: str) -> OpPattern:
        """解析操作参数列表"""
        assert self.text[self.pos] == '('
        self.pos += 1

        children = []
        self._skip_whitespace()

        # 空参数列表
        if self.pos < len(self.text) and self.text[self.pos] == ')':
            self.pos += 1
            return OpPattern(op_name, [])

        # 解析参数
        while True:
            children.append(self._parse_pattern())
            self._skip_whitespace()

            if self.pos >= len(self.text):
                raise ValueError(
                    "Unexpected end of pattern, expected ')' or ','")

            if self.text[self.pos] == ')':
                self.pos += 1
                break
            elif self.text[self.pos] == ',':
                self.pos += 1
                self._skip_whitespace()
            else:
                raise ValueError(
                    f"Expected ')' or ',' at position {self.pos}, got '{self.text[self.pos]}'")

        return OpPattern(op_name, children)

    def _skip_whitespace(self):
        """跳过空白字符"""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1


def parse_pattern(text: str) -> Pattern:
    """解析模式字符串"""
    return PatternParser(text).parse()


# =============================================================================
# Rewrite Rule - 重写规则定义
# =============================================================================

class RuleCategory(Enum):
    """规则分类"""
    ALGEBRA = "algebra"           # 基础代数规则（交换律、结合律等）
    FUSION = "fusion"             # 融合规则
    DECOMPOSE = "decompose"       # 分解规则
    SIMPLIFY = "simplify"         # 简化规则
    LAYOUT = "layout"             # 数据布局规则
    CUSTOM = "custom"             # 自定义规则


@dataclass
class RewriteRule:
    """
    重写规则

    定义一个代数变换规则，包含：
    - 匹配模式 (pattern)
    - 重写模式 (rewrite)
    - 约束条件 (condition)
    - 元信息
    """
    name: str                                    # 规则名称
    pattern: str                                 # 匹配模式字符串
    rewrite: str                                 # 重写模式字符串
    condition: Optional[str] = None              # 约束条件（可选）
    bidirectional: bool = False                  # 是否双向规则
    priority: int = 0                            # 优先级（越高越先应用）
    category: RuleCategory = RuleCategory.ALGEBRA
    description: str = ""                        # 描述

    # 解析后的模式（惰性解析）
    _pattern_ast: Optional[Pattern] = field(default=None, repr=False)
    _rewrite_ast: Optional[Pattern] = field(default=None, repr=False)

    @property
    def pattern_ast(self) -> Pattern:
        """获取解析后的匹配模式"""
        if self._pattern_ast is None:
            self._pattern_ast = parse_pattern(self.pattern)
        return self._pattern_ast

    @property
    def rewrite_ast(self) -> Pattern:
        """获取解析后的重写模式"""
        if self._rewrite_ast is None:
            self._rewrite_ast = parse_pattern(self.rewrite)
        return self._rewrite_ast

    def reverse(self) -> 'RewriteRule':
        """生成反向规则"""
        return RewriteRule(
            name=f"{self.name}_rev",
            pattern=self.rewrite,
            rewrite=self.pattern,
            condition=self.condition,
            bidirectional=False,
            priority=self.priority - 1,
            category=self.category,
            description=f"Reverse of {self.name}",
        )


# =============================================================================
# E-Matching Engine - E-Graph 模式匹配引擎
# =============================================================================

@dataclass
class Match:
    """匹配结果"""
    rule: RewriteRule                    # 匹配的规则
    eclass_id: int                       # 匹配位置的 E-Class ID
    bindings: Dict[str, int]             # 变量绑定 {变量名: E-Class ID}

    def __repr__(self):
        bindings_str = ", ".join(f"?{k}={v}" for k, v in self.bindings.items())
        return f"Match({self.rule.name}, eclass={self.eclass_id}, {{{bindings_str}}})"


class EMatchEngine:
    """
    E-Matching 引擎

    在 E-Graph 中查找所有匹配给定模式的位置
    """

    def __init__(self, egraph: 'EGraph'):
        self.egraph = egraph
        self._match_cache: Dict[Tuple[int, str], List[Dict[str, int]]] = {}

    def find_matches(self, rule: RewriteRule) -> List[Match]:
        """查找规则的所有匹配"""
        pattern = rule.pattern_ast
        matches = []

        for eclass in self.egraph.iter_eclasses():
            bindings_list = self._match_eclass(pattern, eclass.id, {})
            for bindings in bindings_list:
                matches.append(Match(
                    rule=rule,
                    eclass_id=eclass.id,
                    bindings=bindings,
                ))

        return matches

    def _match_eclass(
        self,
        pattern: Pattern,
        eclass_id: int,
        bindings: Dict[str, int]
    ) -> List[Dict[str, int]]:
        """
        在 E-Class 中匹配模式

        返回所有可能的变量绑定列表
        """
        # 获取规范 E-Class ID
        canonical_id = self.egraph.find(eclass_id)

        if isinstance(pattern, VarPattern):
            return self._match_var(pattern, canonical_id, bindings)

        elif isinstance(pattern, OpPattern):
            return self._match_op(pattern, canonical_id, bindings)

        elif isinstance(pattern, WildcardPattern):
            # 通配符匹配任意 E-Class
            return [bindings.copy()]

        elif isinstance(pattern, ConstPattern):
            return self._match_const(pattern, canonical_id, bindings)

        return []

    def _match_var(
        self,
        pattern: VarPattern,
        eclass_id: int,
        bindings: Dict[str, int]
    ) -> List[Dict[str, int]]:
        """匹配变量模式"""
        if pattern.name in bindings:
            # 变量已绑定：检查是否在同一 E-Class
            bound_id = self.egraph.find(bindings[pattern.name])
            if bound_id == eclass_id:
                return [bindings.copy()]
            return []
        else:
            # 变量未绑定：创建新绑定
            new_bindings = bindings.copy()
            new_bindings[pattern.name] = eclass_id
            return [new_bindings]

    def _match_op(
        self,
        pattern: OpPattern,
        eclass_id: int,
        bindings: Dict[str, int]
    ) -> List[Dict[str, int]]:
        """匹配操作模式"""
        eclass = self.egraph.get_eclass(eclass_id)
        if eclass is None:
            return []

        results = []

        for enode in eclass.nodes:
            # 检查操作类型
            if enode.op != pattern.op:
                continue

            # 检查子节点数量
            if len(enode.children) != len(pattern.children):
                continue

            # 递归匹配子模式
            child_results = [bindings.copy()]

            for child_pattern, child_id in zip(pattern.children, enode.children):
                new_results = []
                for b in child_results:
                    matches = self._match_eclass(child_pattern, child_id, b)
                    new_results.extend(matches)
                child_results = new_results

                if not child_results:
                    break

            results.extend(child_results)

        return results

    def _match_const(
        self,
        pattern: ConstPattern,
        eclass_id: int,
        bindings: Dict[str, int]
    ) -> List[Dict[str, int]]:
        """匹配常量模式"""
        eclass = self.egraph.get_eclass(eclass_id)
        if eclass is None:
            return []

        for enode in eclass.nodes:
            # 检查是否是常量节点
            if enode.op in ("const", "constant"):
                # 从属性中获取值
                for attr in enode.attrs:
                    if attr[0] == "value" and attr[1] == pattern.value:
                        return [bindings.copy()]

        return []

    def apply_rewrite(
        self,
        pattern: Pattern,
        bindings: Dict[str, int]
    ) -> int:
        """
        应用重写模式，创建新的 E-Node

        返回新创建的 E-Class ID
        """
        from .egraph import ENode

        if isinstance(pattern, VarPattern):
            # 变量：返回绑定的 E-Class
            if pattern.name not in bindings:
                raise ValueError(
                    f"Unbound variable in rewrite: ?{pattern.name}")
            return bindings[pattern.name]

        elif isinstance(pattern, OpPattern):
            # 操作：递归创建子节点
            children = tuple(
                self.apply_rewrite(child, bindings)
                for child in pattern.children
            )
            enode = ENode(op=pattern.op, children=children, attrs=())
            return self.egraph.add(enode)

        elif isinstance(pattern, ConstPattern):
            # 常量：创建常量节点
            from .egraph import ENode
            enode = ENode(
                op="const",
                children=(),
                attrs=(("value", pattern.value),),
            )
            return self.egraph.add(enode)

        else:
            raise ValueError(
                f"Cannot apply rewrite for pattern type: {type(pattern)}")

    def clear_cache(self):
        """清空匹配缓存"""
        self._match_cache.clear()


# =============================================================================
# Constraint System - 约束检查系统
# =============================================================================

@dataclass
class ShapeInfo:
    """形状信息"""
    shape: List[int]
    dtype: str = "float32"


class ConstraintChecker:
    """
    约束检查器

    检查变换是否满足约束条件
    """

    def __init__(
        self,
        egraph: 'EGraph',
        shape_info: Optional[Dict[int, ShapeInfo]] = None
    ):
        self.egraph = egraph
        self.shape_info = shape_info or {}

        # 注册约束函数
        self._constraints: Dict[str, Callable] = {
            "is_scalar": self._is_scalar,
            "is_vector": self._is_vector,
            "is_matrix": self._is_matrix,
            "is_broadcastable": self._is_broadcastable,
            "same_shape": self._same_shape,
            "same_dtype": self._same_dtype,
            "is_contiguous": self._is_contiguous,
            "shape_compatible": self._shape_compatible,
            "is_bias_shape": self._is_bias_shape,
            "true": lambda *args: True,
            "false": lambda *args: False,
        }

    def check(
        self,
        condition: str,
        bindings: Dict[str, int]
    ) -> bool:
        """
        检查约束条件

        Args:
            condition: 约束表达式，如 "is_broadcastable(?s, ?out)"
            bindings: 变量绑定

        Returns:
            是否满足约束
        """
        if not condition:
            return True

        # 解析约束表达式
        try:
            return self._eval_condition(condition, bindings)
        except Exception as e:
            # 约束检查失败，保守地返回 False
            return False

    def _eval_condition(
        self,
        condition: str,
        bindings: Dict[str, int]
    ) -> bool:
        """评估约束条件"""
        condition = condition.strip()

        # 处理逻辑运算符
        if " and " in condition:
            parts = condition.split(" and ")
            return all(self._eval_condition(p.strip(), bindings) for p in parts)

        if " or " in condition:
            parts = condition.split(" or ")
            return any(self._eval_condition(p.strip(), bindings) for p in parts)

        if condition.startswith("not "):
            return not self._eval_condition(condition[4:].strip(), bindings)

        # 解析函数调用: func_name(arg1, arg2, ...)
        match = re.match(r'(\w+)\s*\((.*)\)', condition)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            # 解析参数
            args = self._parse_args(args_str, bindings)

            # 调用约束函数
            if func_name in self._constraints:
                return self._constraints[func_name](*args)
            else:
                # 未知约束，保守地返回 True
                return True

        # 无法解析，返回 True
        return True

    def _parse_args(
        self,
        args_str: str,
        bindings: Dict[str, int]
    ) -> List[Any]:
        """解析参数列表"""
        args = []
        current = ""
        depth = 0

        for char in args_str + ",":
            if char == ',' and depth == 0:
                arg = current.strip()
                if arg:
                    args.append(self._resolve_arg(arg, bindings))
                current = ""
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current += char

        return args

    def _resolve_arg(
        self,
        arg: str,
        bindings: Dict[str, int]
    ) -> Any:
        """解析单个参数"""
        arg = arg.strip()

        # 变量引用: ?x
        if arg.startswith('?'):
            var_name = arg[1:]
            if var_name in bindings:
                return bindings[var_name]
            raise ValueError(f"Unbound variable: {arg}")

        # 数字
        try:
            if '.' in arg:
                return float(arg)
            return int(arg)
        except ValueError:
            pass

        # 字符串
        if (arg.startswith('"') and arg.endswith('"')) or \
           (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]

        # 其他：作为字符串返回
        return arg

    def register_constraint(
        self,
        name: str,
        func: Callable
    ):
        """注册自定义约束函数"""
        self._constraints[name] = func

    # ========== 内置约束函数 ==========

    def _get_shape(self, eclass_id: int) -> Optional[List[int]]:
        """获取 E-Class 的形状"""
        if eclass_id in self.shape_info:
            return self.shape_info[eclass_id].shape
        return None

    def _is_scalar(self, eclass_id: int) -> bool:
        """检查是否是标量"""
        shape = self._get_shape(eclass_id)
        if shape is None:
            return True  # 未知形状，保守地返回 True
        return len(shape) == 0 or shape == [1] or all(s == 1 for s in shape)

    def _is_vector(self, eclass_id: int) -> bool:
        """检查是否是向量"""
        shape = self._get_shape(eclass_id)
        if shape is None:
            return True
        return len(shape) == 1

    def _is_matrix(self, eclass_id: int) -> bool:
        """检查是否是矩阵"""
        shape = self._get_shape(eclass_id)
        if shape is None:
            return True
        return len(shape) == 2

    def _is_broadcastable(self, src_id: int, dst_id: int) -> bool:
        """检查是否可广播"""
        src_shape = self._get_shape(src_id)
        dst_shape = self._get_shape(dst_id)

        if src_shape is None or dst_shape is None:
            return True  # 未知形状，保守地返回 True

        return self._can_broadcast(src_shape, dst_shape)

    def _can_broadcast(self, src: List[int], dst: List[int]) -> bool:
        """检查广播兼容性"""
        # 对齐维度
        src = [1] * (len(dst) - len(src)) + src

        for s, d in zip(src, dst):
            if s != 1 and s != d:
                return False

        return True

    def _same_shape(self, id1: int, id2: int) -> bool:
        """检查形状是否相同"""
        shape1 = self._get_shape(id1)
        shape2 = self._get_shape(id2)

        if shape1 is None or shape2 is None:
            return True

        return shape1 == shape2

    def _same_dtype(self, id1: int, id2: int) -> bool:
        """检查数据类型是否相同"""
        info1 = self.shape_info.get(id1)
        info2 = self.shape_info.get(id2)

        if info1 is None or info2 is None:
            return True

        return info1.dtype == info2.dtype

    def _is_contiguous(self, eclass_id: int) -> bool:
        """检查是否连续存储"""
        # 默认返回 True（需要更多信息才能确定）
        return True

    def _shape_compatible(self, id1: int, id2: int) -> bool:
        """检查形状是否兼容（可广播或相同）"""
        return self._is_broadcastable(id1, id2) or self._same_shape(id1, id2)

    def _is_bias_shape(self, bias_id: int, output_id: int) -> bool:
        """检查是否是有效的 bias 形状"""
        bias_shape = self._get_shape(bias_id)
        output_shape = self._get_shape(output_id)

        if bias_shape is None or output_shape is None:
            return True

        # bias 应该是 1D 且与输出最后一维匹配
        if len(bias_shape) == 1 and len(output_shape) >= 1:
            return bias_shape[0] == output_shape[-1]

        return self._is_broadcastable(bias_id, output_id)


# =============================================================================
# Rule Application Result
# =============================================================================

@dataclass
class RuleApplication:
    """规则应用结果"""
    rule: RewriteRule
    original_eclass: int
    new_eclass: int
    bindings: Dict[str, int]
    proof: str = ""

    def __repr__(self):
        return f"Applied {self.rule.name}: eclass {self.original_eclass} -> {self.new_eclass}"
