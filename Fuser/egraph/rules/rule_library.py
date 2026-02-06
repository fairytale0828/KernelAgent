"""
统一代数规则库

从各个规则模块导入规则，并提供统一的访问接口。
支持两种规则格式：
1. Rule (来自 base.py) - 使用 RuleBuilder 构建
2. RewriteRule (用于 E-Matching) - 声明式定义

本模块自动将 Rule 转换为 RewriteRule 以供 E-Matching 引擎使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# 导入各规则模块
from .base import Rule, RuleGroup, CostHint
from .basic_algebra import BasicAlgebraRules
from .structural import StructuralRules
from .reduction import ReductionLinearRules
from .matmul import MatmulRules
from .softmax import SoftmaxRules
from .distribution import DistributionRules


# =============================================================================
# RewriteRule - E-Matching 引擎使用的规则格式
# =============================================================================

class RuleCategory(Enum):
    """规则分类"""
    ALGEBRA = auto()      # 代数规则（交换律、结合律等）
    FUSION = auto()       # 融合规则（MatMul+Bias 等）
    DECOMPOSE = auto()    # 分解规则（展开复合操作）
    SIMPLIFY = auto()     # 简化规则（消除冗余）
    STRUCTURAL = auto()   # 结构规则（reshape、transpose 等）


@dataclass
class RewriteRule:
    """
    重写规则 - 用于 E-Matching 引擎

    Attributes:
        name: 规则名称
        pattern: 匹配模式，如 "add(?a, ?b)"
        rewrite: 重写模式，如 "add(?b, ?a)"
        condition: 约束条件，如 "is_broadcastable(?a, ?b)"
        bidirectional: 是否双向规则
        priority: 优先级（越高越先应用）
        category: 规则分类
        description: 规则描述
    """
    name: str
    pattern: str
    rewrite: str
    condition: Optional[str] = None
    bidirectional: bool = False
    priority: int = 0
    category: RuleCategory = RuleCategory.ALGEBRA
    description: str = ""

    # 缓存解析后的模式 AST
    _pattern_ast: Any = field(default=None, repr=False, compare=False)
    _rewrite_ast: Any = field(default=None, repr=False, compare=False)

    @property
    def pattern_ast(self):
        """获取解析后的模式 AST"""
        if self._pattern_ast is None:
            from ..pattern_match import parse_pattern
            object.__setattr__(self, '_pattern_ast',
                               parse_pattern(self.pattern))
        return self._pattern_ast

    @property
    def rewrite_ast(self):
        """获取解析后的重写 AST"""
        if self._rewrite_ast is None:
            from ..pattern_match import parse_pattern
            object.__setattr__(self, '_rewrite_ast',
                               parse_pattern(self.rewrite))
        return self._rewrite_ast

    def reverse(self) -> "RewriteRule":
        """创建反向规则"""
        return RewriteRule(
            name=f"{self.name}_rev",
            pattern=self.rewrite,
            rewrite=self.pattern,
            condition=self.condition,
            bidirectional=False,  # 反向规则不再是双向的
            priority=self.priority - 1,  # 稍低优先级
            category=self.category,
            description=f"{self.description} (反向)",
        )


# =============================================================================
# 规则转换：Rule -> RewriteRule
# =============================================================================

def _rule_group_to_category(group: RuleGroup) -> RuleCategory:
    """将 RuleGroup 转换为 RuleCategory"""
    mapping = {
        RuleGroup.G1_BASIC_ALGEBRA: RuleCategory.ALGEBRA,
        RuleGroup.G2_STRUCTURAL: RuleCategory.STRUCTURAL,
        RuleGroup.G3_REDUCTION_LINEAR: RuleCategory.ALGEBRA,
        RuleGroup.G4_MATMUL: RuleCategory.FUSION,
        RuleGroup.G5_SOFTMAX: RuleCategory.DECOMPOSE,
        RuleGroup.G6_DISTRIBUTION: RuleCategory.ALGEBRA,
    }
    return mapping.get(group, RuleCategory.ALGEBRA)


def _convert_rule_to_rewrite_rule(rule: Rule) -> RewriteRule:
    """将 Rule 转换为 RewriteRule"""
    # 转换模式格式：a -> ?a
    def convert_pattern(pattern: str) -> str:
        import re
        # 将单字母变量转换为 ?变量 格式
        # 例如: add(a, b) -> add(?a, ?b)
        # 但保留操作名和常量
        result = []
        i = 0
        while i < len(pattern):
            char = pattern[i]
            # 检查是否是标识符开始
            if char.isalpha() or char == '_':
                # 读取完整标识符
                start = i
                while i < len(pattern) and (pattern[i].isalnum() or pattern[i] == '_'):
                    i += 1
                ident = pattern[start:i]

                # 判断是否是变量（单个小写字母或常见变量名）
                # 操作名通常是全小写多字母，如 matmul, add, mul
                is_var = (
                    len(ident) == 1 and ident.islower() or
                    ident in ('X', 'Y', 'W', 'A', 'B', 'C') or
                    ident in ('gamma', 'beta', 'norm_x', 'rsqrt_rms')
                )

                if is_var:
                    result.append(f"?{ident}")
                else:
                    result.append(ident)
            else:
                result.append(char)
                i += 1

        return ''.join(result)

    return RewriteRule(
        name=rule.name,
        pattern=convert_pattern(rule.lhs_pattern),
        rewrite=convert_pattern(rule.rhs_pattern),
        condition=None,  # 条件需要手动转换
        bidirectional=rule.bidirectional,
        priority=rule.priority,
        category=_rule_group_to_category(rule.group),
        description=rule.soundness,
    )


# =============================================================================
# 从各模块收集规则
# =============================================================================

def _collect_rules_from_modules() -> List[RewriteRule]:
    """从各规则模块收集规则"""
    all_rules: List[RewriteRule] = []

    # 收集各模块的规则
    rule_sources = [
        BasicAlgebraRules.all(),
        StructuralRules.all(),
        ReductionLinearRules.all(),
        MatmulRules.all(),
        SoftmaxRules.all(),
        DistributionRules.all(),
    ]

    for rules in rule_sources:
        for rule in rules:
            rewrite_rule = _convert_rule_to_rewrite_rule(rule)
            all_rules.append(rewrite_rule)

    return all_rules


# =============================================================================
# 额外的手动定义规则（补充模块中没有的规则）
# =============================================================================

EXTRA_RULES: List[RewriteRule] = [
    # ========== Late Scaling 规则 ==========
    RewriteRule(
        name="late_scaling_mul",
        pattern="matmul(mul(?x, ?s), ?w)",
        rewrite="mul(matmul(?x, ?w), ?s)",
        condition="is_broadcastable(?s, ?x)",
        priority=10,
        category=RuleCategory.FUSION,
        description="将乘法缩放推迟到 matmul 之后: matmul(x*s, W) = matmul(x, W)*s",
    ),
    RewriteRule(
        name="late_scaling_div",
        pattern="matmul(div(?x, ?s), ?w)",
        rewrite="div(matmul(?x, ?w), ?s)",
        condition="is_broadcastable(?s, ?x)",
        priority=10,
        category=RuleCategory.FUSION,
        description="将除法缩放推迟到 matmul 之后: matmul(x/s, W) = matmul(x, W)/s",
    ),

    # ========== RMSNorm 规则 ==========
    RewriteRule(
        name="rmsnorm_expand",
        pattern="rms_norm(?x, ?w, ?eps)",
        rewrite="mul(mul(?x, rsqrt(add(reduce_mean(mul(?x, ?x)), ?eps))), ?w)",
        category=RuleCategory.DECOMPOSE,
        description="展开 RMSNorm: x * rsqrt(mean(x^2) + eps) * w",
    ),

    # ========== LoRA 规则 ==========
    RewriteRule(
        name="lora_fusion",
        pattern="add(matmul(?x, ?w), matmul(matmul(?x, ?b), ?a))",
        rewrite="matmul(?x, add(?w, matmul(?b, ?a)))",
        priority=20,
        category=RuleCategory.FUSION,
        description="LoRA 权重预融合: Wx + (xB)A = x(W + BA)",
    ),

    # ========== Attention 规则 ==========
    RewriteRule(
        name="attention_pattern",
        pattern="matmul(softmax(div(matmul(?q, transpose(?k)), ?scale)), ?v)",
        rewrite="attention(?q, ?k, ?v, ?scale)",
        priority=30,
        category=RuleCategory.FUSION,
        description="识别标准 Attention 模式",
    ),
    RewriteRule(
        name="flash_attention",
        pattern="attention(?q, ?k, ?v, ?scale)",
        rewrite="flash_attention(?q, ?k, ?v, ?scale)",
        priority=35,
        category=RuleCategory.FUSION,
        description="转换为 Flash Attention",
    ),

    # ========== MatMul + Bias 融合 ==========
    RewriteRule(
        name="matmul_bias_fusion",
        pattern="add(matmul(?x, ?w), ?b)",
        rewrite="matmul_bias(?x, ?w, ?b)",
        condition="is_bias_shape(?b, ?x)",
        priority=20,
        category=RuleCategory.FUSION,
        description="融合 matmul 和 bias: matmul(x, W) + b = matmul_bias(x, W, b)",
    ),

    # ========== MatMul + Activation 融合 ==========
    RewriteRule(
        name="matmul_relu_fusion",
        pattern="relu(matmul(?x, ?w))",
        rewrite="matmul_relu(?x, ?w)",
        priority=15,
        category=RuleCategory.FUSION,
        description="融合 matmul 和 ReLU",
    ),
    RewriteRule(
        name="matmul_gelu_fusion",
        pattern="gelu(matmul(?x, ?w))",
        rewrite="matmul_gelu(?x, ?w)",
        priority=15,
        category=RuleCategory.FUSION,
        description="融合 matmul 和 GELU",
    ),
    RewriteRule(
        name="matmul_bias_relu_fusion",
        pattern="relu(add(matmul(?x, ?w), ?b))",
        rewrite="matmul_bias_relu(?x, ?w, ?b)",
        priority=25,
        category=RuleCategory.FUSION,
        description="融合 matmul + bias + ReLU",
    ),
    RewriteRule(
        name="matmul_bias_gelu_fusion",
        pattern="gelu(add(matmul(?x, ?w), ?b))",
        rewrite="matmul_bias_gelu(?x, ?w, ?b)",
        priority=25,
        category=RuleCategory.FUSION,
        description="融合 matmul + bias + GELU",
    ),
    RewriteRule(
        name="matmul_silu_fusion",
        pattern="silu(matmul(?x, ?w))",
        rewrite="matmul_silu(?x, ?w)",
        priority=15,
        category=RuleCategory.FUSION,
        description="融合 matmul 和 SiLU",
    ),
    RewriteRule(
        name="matmul_bias_silu_fusion",
        pattern="silu(add(matmul(?x, ?w), ?b))",
        rewrite="matmul_bias_silu(?x, ?w, ?b)",
        priority=25,
        category=RuleCategory.FUSION,
        description="融合 matmul + bias + SiLU",
    ),

    # ========== MatMul + Activation + Softmax 融合 ==========
    RewriteRule(
        name="matmul_gelu_softmax_fusion",
        pattern="softmax(gelu(matmul(?x, ?w)))",
        rewrite="matmul_gelu_softmax(?x, ?w)",
        priority=30,
        category=RuleCategory.FUSION,
        description="融合 matmul + GELU + softmax",
    ),
    RewriteRule(
        name="matmul_bias_gelu_softmax_fusion",
        pattern="softmax(gelu(add(matmul(?x, ?w), ?b)))",
        rewrite="matmul_bias_gelu_softmax(?x, ?w, ?b)",
        priority=35,
        category=RuleCategory.FUSION,
        description="融合 matmul + bias + GELU + softmax",
    ),
    RewriteRule(
        name="matmul_relu_softmax_fusion",
        pattern="softmax(relu(matmul(?x, ?w)))",
        rewrite="matmul_relu_softmax(?x, ?w)",
        priority=30,
        category=RuleCategory.FUSION,
        description="融合 matmul + ReLU + softmax",
    ),
    RewriteRule(
        name="matmul_bias_relu_softmax_fusion",
        pattern="softmax(relu(add(matmul(?x, ?w), ?b)))",
        rewrite="matmul_bias_relu_softmax(?x, ?w, ?b)",
        priority=35,
        category=RuleCategory.FUSION,
        description="融合 matmul + bias + ReLU + softmax",
    ),
    RewriteRule(
        name="linear_gelu_softmax_fusion",
        pattern="softmax(gelu(matmul_bias(?x, ?w, ?b)))",
        rewrite="linear_gelu_softmax(?x, ?w, ?b)",
        priority=35,
        category=RuleCategory.FUSION,
        description="融合 linear + GELU + softmax",
    ),

    # ========== Online Softmax 规则 ==========
    # Online Softmax 是一种单次遍历计算 softmax 的优化技术
    RewriteRule(
        name="softmax_to_online",
        pattern="softmax(?x)",
        rewrite="online_softmax(?x)",
        priority=20,
        category=RuleCategory.FUSION,
        description="转换为 Online Softmax（单次遍历实现）",
    ),
    RewriteRule(
        name="softmax_expanded_to_online",
        pattern="div(exp(sub(?x, reduce_max(?x))), reduce_sum(exp(sub(?x, reduce_max(?x)))))",
        rewrite="online_softmax(?x)",
        priority=25,
        category=RuleCategory.FUSION,
        description="将展开的 softmax 转换为 Online Softmax",
    ),
    RewriteRule(
        name="safe_softmax_to_online",
        pattern="div(exp(sub(?x, ?max)), reduce_sum(exp(sub(?x, ?max))))",
        rewrite="online_softmax(?x)",
        priority=25,
        category=RuleCategory.FUSION,
        description="将数值稳定的 softmax 转换为 Online Softmax",
    ),

    # ========== Softmax + 后续操作融合 ==========
    # RewriteRule(
    #     name="softmax_matmul_fusion",
    #     pattern="matmul(softmax(?x), ?v)",
    #     rewrite="softmax_matmul(?x, ?v)",
    #     priority=25,
    #     category=RuleCategory.FUSION,
    #     description="融合 softmax 和后续 matmul（用于 Attention）",
    # ),
    RewriteRule(
        name="online_softmax_matmul_fusion",
        pattern="matmul(online_softmax(?x), ?v)",
        rewrite="online_softmax_matmul(?x, ?v)",
        priority=30,
        category=RuleCategory.FUSION,
        description="融合 Online Softmax 和后续 matmul",
    ),

    # ========== GELU 分解和融合 ==========
    RewriteRule(
        name="gelu_expand_tanh",
        pattern="gelu(?x)",
        rewrite="mul(mul(?x, const_0.5), add(const_1, tanh(mul(mul(const_0.7978845608, add(?x, mul(const_0.044715, pow(?x, const_3)))), const_1))))",
        category=RuleCategory.DECOMPOSE,
        description="GELU 的 tanh 近似展开",
    ),
    RewriteRule(
        name="gelu_expand_erf",
        pattern="gelu(?x)",
        rewrite="mul(mul(?x, const_0.5), add(const_1, erf(div(?x, const_1.4142135623))))",
        category=RuleCategory.DECOMPOSE,
        description="GELU 的精确 erf 展开",
    ),

    # ========== SiLU (Swish) 分解 ==========
    RewriteRule(
        name="silu_expand",
        pattern="silu(?x)",
        rewrite="mul(?x, sigmoid(?x))",
        category=RuleCategory.DECOMPOSE,
        description="SiLU 展开: silu(x) = x * sigmoid(x)",
    ),
    RewriteRule(
        name="silu_fold",
        pattern="mul(?x, sigmoid(?x))",
        rewrite="silu(?x)",
        priority=15,
        category=RuleCategory.FUSION,
        description="SiLU 折叠: x * sigmoid(x) = silu(x)",
    ),

    # ========== LayerNorm 规则 ==========
    RewriteRule(
        name="layernorm_expand",
        pattern="layer_norm(?x, ?w, ?b, ?eps)",
        rewrite="add(mul(div(sub(?x, reduce_mean(?x)), sqrt(add(reduce_var(?x), ?eps))), ?w), ?b)",
        category=RuleCategory.DECOMPOSE,
        description="展开 LayerNorm",
    ),
    RewriteRule(
        name="layernorm_matmul_fusion",
        pattern="matmul(layer_norm(?x, ?w, ?b, ?eps), ?wm)",
        rewrite="layernorm_matmul(?x, ?w, ?b, ?eps, ?wm)",
        priority=25,
        category=RuleCategory.FUSION,
        description="融合 LayerNorm 和 MatMul",
    ),

    # ========== Dropout 规则 ==========
    RewriteRule(
        name="dropout_inference",
        pattern="dropout(?x, ?p)",
        rewrite="?x",
        priority=20,
        category=RuleCategory.SIMPLIFY,
        description="推理时 dropout 可以移除",
    ),
    RewriteRule(
        name="dropout_matmul_fusion",
        pattern="matmul(dropout(?x, ?p), ?w)",
        rewrite="dropout_matmul(?x, ?p, ?w)",
        priority=15,
        category=RuleCategory.FUSION,
        description="融合 dropout 和 matmul",
    ),

    # ========== 残差连接规则 ==========
    RewriteRule(
        name="residual_add_fusion",
        pattern="add(?x, layer_norm(add(?x, ?y), ?w, ?b, ?eps))",
        rewrite="residual_layernorm(?x, ?y, ?w, ?b, ?eps)",
        priority=25,
        category=RuleCategory.FUSION,
        description="融合残差连接和 LayerNorm",
    ),
    RewriteRule(
        name="residual_matmul_fusion",
        pattern="add(?x, matmul(?x, ?w))",
        rewrite="residual_matmul(?x, ?w)",
        priority=20,
        category=RuleCategory.FUSION,
        description="融合残差连接和 MatMul",
    ),

    # ========== 简化规则 ==========
    RewriteRule(
        name="add_zero",
        pattern="add(?x, const_0)",
        rewrite="?x",
        category=RuleCategory.SIMPLIFY,
        description="加零恒等: x + 0 = x",
    ),
    RewriteRule(
        name="mul_one",
        pattern="mul(?x, const_1)",
        rewrite="?x",
        category=RuleCategory.SIMPLIFY,
        description="乘一恒等: x * 1 = x",
    ),
    RewriteRule(
        name="mul_zero",
        pattern="mul(?x, const_0)",
        rewrite="const_0",
        category=RuleCategory.SIMPLIFY,
        description="乘零: x * 0 = 0",
    ),
    RewriteRule(
        name="double_neg",
        pattern="neg(neg(?x))",
        rewrite="?x",
        category=RuleCategory.SIMPLIFY,
        description="双重取负: --x = x",
    ),
    RewriteRule(
        name="double_transpose",
        pattern="transpose(transpose(?x))",
        rewrite="?x",
        category=RuleCategory.SIMPLIFY,
        description="双重转置: x.T.T = x",
    ),
]


# =============================================================================
# 规则集合 API
# =============================================================================

# 缓存
_ALL_RULES_CACHE: Optional[List[RewriteRule]] = None


def get_all_rules() -> List[RewriteRule]:
    """获取所有规则"""
    global _ALL_RULES_CACHE

    if _ALL_RULES_CACHE is None:
        all_rules = []

        # 从模块收集规则
        all_rules.extend(_collect_rules_from_modules())

        # 添加额外规则
        all_rules.extend(EXTRA_RULES)

        # 添加双向规则的反向版本
        reverse_rules = []
        for rule in all_rules:
            if rule.bidirectional:
                reverse_rules.append(rule.reverse())
        all_rules.extend(reverse_rules)

        # 去重（按名称）
        seen_names: Set[str] = set()
        unique_rules = []
        for rule in all_rules:
            if rule.name not in seen_names:
                seen_names.add(rule.name)
                unique_rules.append(rule)

        # 按优先级排序
        unique_rules.sort(key=lambda r: -r.priority)

        _ALL_RULES_CACHE = unique_rules

    return _ALL_RULES_CACHE


def get_rules_by_category(category: RuleCategory) -> List[RewriteRule]:
    """按分类获取规则"""
    return [r for r in get_all_rules() if r.category == category]


def get_fusion_rules() -> List[RewriteRule]:
    """获取融合规则"""
    return get_rules_by_category(RuleCategory.FUSION)


def get_algebra_rules() -> List[RewriteRule]:
    """获取代数规则"""
    return get_rules_by_category(RuleCategory.ALGEBRA)


def get_simplify_rules() -> List[RewriteRule]:
    """获取简化规则"""
    return get_rules_by_category(RuleCategory.SIMPLIFY)


# =============================================================================
# 规则集预设
# =============================================================================

def _deduplicate_rules(rules: List[RewriteRule]) -> List[RewriteRule]:
    """去重规则列表"""
    seen: Set[str] = set()
    result = []
    for r in rules:
        if r.name not in seen:
            seen.add(r.name)
            result.append(r)
    return result


def _build_rule_sets() -> Dict[str, List[RewriteRule]]:
    """构建规则集预设"""
    all_rules = get_all_rules()

    # 基础代数规则（交换律、结合律）
    algebra_basic = [r for r in all_rules if r.category ==
                     RuleCategory.ALGEBRA][:8]

    # Late Scaling 规则
    late_scaling = [
        r for r in all_rules if "late_scaling" in r.name or "scaling" in r.name.lower()]

    # RMSNorm 相关规则
    rmsnorm = [r for r in all_rules if "rms" in r.name.lower()
               or "norm" in r.name.lower()]

    # Attention 相关规则
    attention = [r for r in all_rules if "attention" in r.name.lower()
                 or "softmax" in r.name.lower()]

    # 融合规则
    fusion_rules = get_fusion_rules()

    # MatMul + Bias 融合规则（常用）
    matmul_bias_rules = [r for r in all_rules if "matmul_bias" in r.name]

    # MatMul + Activation 融合规则
    matmul_activation_rules = [r for r in all_rules if
                               "matmul_gelu" in r.name or
                               "matmul_relu" in r.name or
                               "matmul_silu" in r.name or
                               "linear_gelu" in r.name]

    # Online Softmax 相关规则
    online_softmax_rules = [r for r in all_rules if "online" in r.name.lower()]

    # Softmax 融合规则
    softmax_fusion_rules = [r for r in all_rules if
                            "softmax" in r.name.lower() and
                            r.category == RuleCategory.FUSION]

    # GELU/SiLU 相关规则
    activation_rules = [r for r in all_rules if
                        "gelu" in r.name.lower() or
                        "silu" in r.name.lower() or
                        "relu" in r.name.lower()]

    return {
        "minimal": _deduplicate_rules(algebra_basic[:4]),
        "default": _deduplicate_rules(
            algebra_basic +
            late_scaling +
            matmul_bias_rules +
            matmul_activation_rules +
            softmax_fusion_rules[:5] +
            fusion_rules[:5]
        ),
        "full": all_rules,
        "fusion_only": _deduplicate_rules(fusion_rules),
        "late_scaling": _deduplicate_rules(late_scaling + algebra_basic[:4] + matmul_bias_rules),
        "rmsnorm": _deduplicate_rules(rmsnorm + late_scaling + algebra_basic[:4] + matmul_bias_rules),
        "attention": _deduplicate_rules(attention + algebra_basic[:4] + online_softmax_rules),
        "simplify": _deduplicate_rules(get_simplify_rules() + algebra_basic[:4]),
        "online_softmax": _deduplicate_rules(
            online_softmax_rules +
            softmax_fusion_rules +
            algebra_basic[:4]
        ),
        "activation_fusion": _deduplicate_rules(
            activation_rules +
            matmul_activation_rules +
            matmul_bias_rules +
            algebra_basic[:4]
        ),
        "mlp": _deduplicate_rules(
            matmul_bias_rules +
            matmul_activation_rules +
            softmax_fusion_rules +
            late_scaling +
            algebra_basic[:4]
        ),
    }


# 规则集缓存
_RULE_SETS: Optional[Dict[str, List[RewriteRule]]] = None


def get_rule_set(name: str) -> List[RewriteRule]:
    """获取预设规则集"""
    global _RULE_SETS

    if _RULE_SETS is None:
        _RULE_SETS = _build_rule_sets()

    if name not in _RULE_SETS:
        available = list(_RULE_SETS.keys())
        raise ValueError(f"Unknown rule set: {name}. Available: {available}")

    return _RULE_SETS[name]


def list_rule_sets() -> List[str]:
    """列出所有可用的规则集"""
    global _RULE_SETS

    if _RULE_SETS is None:
        _RULE_SETS = _build_rule_sets()

    return list(_RULE_SETS.keys())
