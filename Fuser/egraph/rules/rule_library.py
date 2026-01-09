# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
代数规则库

包含常用的代数变换规则，支持自动优化探索
"""

from typing import List
from ..pattern_match import RewriteRule, RuleCategory


# =============================================================================
# 基础代数规则 - 交换律、结合律、分配律
# =============================================================================

ALGEBRA_RULES: List[RewriteRule] = [
    # ========== 交换律 (Commutativity) ==========
    RewriteRule(
        name="commute_add",
        pattern="add(?a, ?b)",
        rewrite="add(?b, ?a)",
        bidirectional=True,
        category=RuleCategory.ALGEBRA,
        description="加法交换律: a + b = b + a",
    ),
    RewriteRule(
        name="commute_mul",
        pattern="mul(?a, ?b)",
        rewrite="mul(?b, ?a)",
        bidirectional=True,
        category=RuleCategory.ALGEBRA,
        description="乘法交换律: a * b = b * a",
    ),
    RewriteRule(
        name="commute_max",
        pattern="max(?a, ?b)",
        rewrite="max(?b, ?a)",
        bidirectional=True,
        category=RuleCategory.ALGEBRA,
        description="最大值交换律",
    ),
    RewriteRule(
        name="commute_min",
        pattern="min(?a, ?b)",
        rewrite="min(?b, ?a)",
        bidirectional=True,
        category=RuleCategory.ALGEBRA,
        description="最小值交换律",
    ),
    
    # ========== 结合律 (Associativity) ==========
    RewriteRule(
        name="assoc_add_left",
        pattern="add(add(?a, ?b), ?c)",
        rewrite="add(?a, add(?b, ?c))",
        category=RuleCategory.ALGEBRA,
        description="加法结合律: (a + b) + c = a + (b + c)",
    ),
    RewriteRule(
        name="assoc_add_right",
        pattern="add(?a, add(?b, ?c))",
        rewrite="add(add(?a, ?b), ?c)",
        category=RuleCategory.ALGEBRA,
        description="加法结合律（反向）",
    ),
    RewriteRule(
        name="assoc_mul_left",
        pattern="mul(mul(?a, ?b), ?c)",
        rewrite="mul(?a, mul(?b, ?c))",
        category=RuleCategory.ALGEBRA,
        description="乘法结合律: (a * b) * c = a * (b * c)",
    ),
    RewriteRule(
        name="assoc_mul_right",
        pattern="mul(?a, mul(?b, ?c))",
        rewrite="mul(mul(?a, ?b), ?c)",
        category=RuleCategory.ALGEBRA,
        description="乘法结合律（反向）",
    ),
    
    # ========== 分配律 (Distributivity) ==========
    RewriteRule(
        name="distribute_mul_add_left",
        pattern="mul(?a, add(?b, ?c))",
        rewrite="add(mul(?a, ?b), mul(?a, ?c))",
        category=RuleCategory.ALGEBRA,
        description="乘法对加法分配律: a * (b + c) = a*b + a*c",
    ),
    RewriteRule(
        name="distribute_mul_add_right",
        pattern="mul(add(?a, ?b), ?c)",
        rewrite="add(mul(?a, ?c), mul(?b, ?c))",
        category=RuleCategory.ALGEBRA,
        description="乘法对加法分配律: (a + b) * c = a*c + b*c",
    ),
    RewriteRule(
        name="factor_mul_add",
        pattern="add(mul(?a, ?b), mul(?a, ?c))",
        rewrite="mul(?a, add(?b, ?c))",
        category=RuleCategory.ALGEBRA,
        description="提取公因子: a*b + a*c = a*(b+c)",
    ),
]


# =============================================================================
# Late Scaling 规则 - 将缩放推迟到 MatMul 之后
# =============================================================================

LATE_SCALING_RULES: List[RewriteRule] = [
    # ========== 基础 Late Scaling ==========
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
    
    # ========== 嵌套 Late Scaling (RMSNorm 场景) ==========
    RewriteRule(
        name="late_scaling_nested_mul",
        pattern="matmul(mul(mul(?x, ?s1), ?s2), ?w)",
        rewrite="mul(mul(matmul(?x, ?w), ?s1), ?s2)",
        condition="is_broadcastable(?s1, ?x) and is_broadcastable(?s2, ?x)",
        priority=15,
        category=RuleCategory.FUSION,
        description="嵌套乘法的 late scaling: matmul((x*s1)*s2, W) = (matmul(x,W)*s1)*s2",
    ),
    
    # ========== 合并缩放因子 ==========
    RewriteRule(
        name="merge_scaling_factors",
        pattern="mul(mul(?x, ?s1), ?s2)",
        rewrite="mul(?x, mul(?s1, ?s2))",
        condition="is_broadcastable(?s1, ?s2)",
        priority=5,
        category=RuleCategory.SIMPLIFY,
        description="合并缩放因子: (x*s1)*s2 = x*(s1*s2)",
    ),
    
    # ========== 带 bias 的 Late Scaling ==========
    RewriteRule(
        name="late_scaling_with_bias",
        pattern="add(matmul(mul(?x, ?s), ?w), ?b)",
        rewrite="add(mul(matmul(?x, ?w), ?s), ?b)",
        condition="is_broadcastable(?s, ?x)",
        priority=12,
        category=RuleCategory.FUSION,
        description="带 bias 的 late scaling",
    ),
]


# =============================================================================
# MatMul 融合规则
# =============================================================================

MATMUL_FUSION_RULES: List[RewriteRule] = [
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
    
    # ========== MatMul + Bias + Activation 融合 ==========
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
]


# =============================================================================
# RMSNorm 相关规则
# =============================================================================

RMSNORM_RULES: List[RewriteRule] = [
    # ========== RMSNorm 分解 ==========
    RewriteRule(
        name="rmsnorm_expand",
        pattern="rms_norm(?x, ?w, ?eps)",
        rewrite="mul(mul(?x, rsqrt(add(reduce_mean(mul(?x, ?x)), ?eps))), ?w)",
        category=RuleCategory.DECOMPOSE,
        description="展开 RMSNorm: x * rsqrt(mean(x^2) + eps) * w",
    ),
    
    # ========== RMSNorm + MatMul 融合 ==========
    RewriteRule(
        name="rmsnorm_matmul_fuse",
        pattern="matmul(mul(mul(?x, rsqrt(add(reduce_mean(mul(?x, ?x)), ?eps))), ?w), ?wm)",
        rewrite="rmsnorm_matmul_fused(?x, ?w, ?wm, ?eps)",
        priority=30,
        category=RuleCategory.FUSION,
        description="融合 RMSNorm 和 MatMul",
    ),
    
    # ========== RMSNorm Late Scaling ==========
    RewriteRule(
        name="rmsnorm_late_scaling",
        pattern="matmul(mul(?x, mul(rsqrt(?var), ?w)), ?wm)",
        rewrite="mul(matmul(?x, ?wm), mul(rsqrt(?var), ?w))",
        priority=25,
        category=RuleCategory.FUSION,
        description="RMSNorm 的 late scaling 优化",
    ),
]


# =============================================================================
# LoRA 相关规则
# =============================================================================

LORA_RULES: List[RewriteRule] = [
    # ========== LoRA 融合 ==========
    RewriteRule(
        name="lora_fusion",
        pattern="add(matmul(?x, ?w), matmul(matmul(?x, ?b), ?a))",
        rewrite="matmul(?x, add(?w, matmul(?b, ?a)))",
        priority=20,
        category=RuleCategory.FUSION,
        description="LoRA 权重预融合: Wx + (xB)A = x(W + BA)",
    ),
    
    # ========== LoRA 分解（用于分析） ==========
    RewriteRule(
        name="lora_decompose",
        pattern="matmul(?x, add(?w, matmul(?b, ?a)))",
        rewrite="add(matmul(?x, ?w), matmul(matmul(?x, ?b), ?a))",
        category=RuleCategory.DECOMPOSE,
        description="LoRA 分解（反向）",
    ),
]


# =============================================================================
# Softmax 相关规则
# =============================================================================

SOFTMAX_RULES: List[RewriteRule] = [
    # ========== Softmax 分解 ==========
    RewriteRule(
        name="softmax_expand",
        pattern="softmax(?x)",
        rewrite="div(exp(sub(?x, reduce_max(?x))), reduce_sum(exp(sub(?x, reduce_max(?x)))))",
        category=RuleCategory.DECOMPOSE,
        description="展开 softmax 为数值稳定形式",
    ),
    
    # ========== Safe Softmax (数值稳定) ==========
    RewriteRule(
        name="softmax_safe",
        pattern="div(exp(?x), reduce_sum(exp(?x)))",
        rewrite="div(exp(sub(?x, reduce_max(?x))), reduce_sum(exp(sub(?x, reduce_max(?x)))))",
        priority=15,
        category=RuleCategory.SIMPLIFY,
        description="转换为数值稳定的 softmax",
    ),
]


# =============================================================================
# Attention 相关规则
# =============================================================================

ATTENTION_RULES: List[RewriteRule] = [
    # ========== Attention 模式识别 ==========
    RewriteRule(
        name="attention_pattern",
        pattern="matmul(softmax(div(matmul(?q, transpose(?k)), ?scale)), ?v)",
        rewrite="attention(?q, ?k, ?v, ?scale)",
        priority=30,
        category=RuleCategory.FUSION,
        description="识别标准 Attention 模式",
    ),
    
    # ========== Flash Attention ==========
    RewriteRule(
        name="flash_attention",
        pattern="attention(?q, ?k, ?v, ?scale)",
        rewrite="flash_attention(?q, ?k, ?v, ?scale)",
        priority=35,
        category=RuleCategory.FUSION,
        description="转换为 Flash Attention",
    ),
    
    # ========== Attention 分解 ==========
    RewriteRule(
        name="attention_decompose",
        pattern="attention(?q, ?k, ?v, ?scale)",
        rewrite="matmul(softmax(div(matmul(?q, transpose(?k)), ?scale)), ?v)",
        category=RuleCategory.DECOMPOSE,
        description="分解 Attention 为基础操作",
    ),
]


# =============================================================================
# 简化规则
# =============================================================================

SIMPLIFY_RULES: List[RewriteRule] = [
    # ========== 恒等式 ==========
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
    
    # ========== 双重操作消除 ==========
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
    
    # ========== rsqrt 相关 ==========
    RewriteRule(
        name="rsqrt_mul_rsqrt",
        pattern="mul(rsqrt(?x), rsqrt(?x))",
        rewrite="div(const_1, ?x)",
        category=RuleCategory.SIMPLIFY,
        description="rsqrt(x) * rsqrt(x) = 1/x",
    ),
]


# =============================================================================
# 规则集合
# =============================================================================

def get_all_rules() -> List[RewriteRule]:
    """获取所有规则"""
    all_rules = []
    all_rules.extend(ALGEBRA_RULES)
    all_rules.extend(LATE_SCALING_RULES)
    all_rules.extend(MATMUL_FUSION_RULES)
    all_rules.extend(RMSNORM_RULES)
    all_rules.extend(LORA_RULES)
    all_rules.extend(SOFTMAX_RULES)
    all_rules.extend(ATTENTION_RULES)
    all_rules.extend(SIMPLIFY_RULES)
    
    # 添加双向规则的反向版本
    reverse_rules = []
    for rule in all_rules:
        if rule.bidirectional:
            reverse_rules.append(rule.reverse())
    all_rules.extend(reverse_rules)
    
    # 按优先级排序
    all_rules.sort(key=lambda r: -r.priority)
    
    return all_rules


def get_rules_by_category(category: RuleCategory) -> List[RewriteRule]:
    """按分类获取规则"""
    return [r for r in get_all_rules() if r.category == category]


def get_fusion_rules() -> List[RewriteRule]:
    """获取融合规则"""
    return get_rules_by_category(RuleCategory.FUSION)


def get_algebra_rules() -> List[RewriteRule]:
    """获取代数规则"""
    return get_rules_by_category(RuleCategory.ALGEBRA)


# =============================================================================
# 规则集预设
# =============================================================================

RULE_SETS = {
    "minimal": ALGEBRA_RULES[:4],  # 只有交换律
    "default": ALGEBRA_RULES + LATE_SCALING_RULES + MATMUL_FUSION_RULES[:2],
    "full": get_all_rules(),
    "fusion_only": get_fusion_rules(),
    "late_scaling": LATE_SCALING_RULES,
    "rmsnorm": RMSNORM_RULES + LATE_SCALING_RULES,
    "attention": ATTENTION_RULES + SOFTMAX_RULES,
}


def get_rule_set(name: str) -> List[RewriteRule]:
    """获取预设规则集"""
    if name not in RULE_SETS:
        raise ValueError(f"Unknown rule set: {name}. Available: {list(RULE_SETS.keys())}")
    return RULE_SETS[name]
