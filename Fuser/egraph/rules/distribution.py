# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
G6: 分配律规则 (谨慎使用)

这些规则可能增加 FLOPs，需要强预算约束。
仅在特定优化场景下启用。
"""

from typing import List

from .base import Rule, RuleBuilder, RuleGroup, CostHint, RuleCondition


class DistributionRules:
    """分配律规则集"""
    
    @staticmethod
    def all() -> List[Rule]:
        """返回所有 G6 规则"""
        return [
            # === 乘法分配律 ===
            RuleBuilder("mul_add_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("mul(add(a, b), c)", "add(mul(a, c), mul(b, c))")
                .cost(CostHint(flops_delta=1))  # 增加一次乘法
                .condition(RuleCondition(
                    custom=lambda ctx: ctx.get("flops_growth_ratio", 2.0) <= 1.5
                ))
                .priority(3)
                .soundness("乘法分配律: (a+b)*c = a*c + b*c")
                .build(),
            
            RuleBuilder("mul_add_factor")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("add(mul(a, c), mul(b, c))", "mul(add(a, b), c)")
                .cost(CostHint(flops_delta=-1))  # 减少一次乘法
                .priority(12)
                .soundness("乘法提取: a*c + b*c = (a+b)*c")
                .build(),
            
            # === 除法分配律 ===
            RuleBuilder("div_add_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("div(add(a, b), c)", "add(div(a, c), div(b, c))")
                .cost(CostHint(flops_delta=1))
                .priority(3)
                .soundness("除法分配律: (a+b)/c = a/c + b/c")
                .build(),
            
            RuleBuilder("div_add_factor")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("add(div(a, c), div(b, c))", "div(add(a, b), c)")
                .cost(CostHint(flops_delta=-1))
                .priority(12)
                .soundness("除法提取: a/c + b/c = (a+b)/c")
                .build(),
            
            # === 幂运算分配律 ===
            RuleBuilder("pow_mul_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("pow(mul(a, b), n)", "mul(pow(a, n), pow(b, n))")
                .cost(CostHint(flops_delta=1))
                .priority(3)
                .soundness("幂运算分配: (a*b)^n = a^n * b^n")
                .build(),
            
            RuleBuilder("pow_mul_factor")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("mul(pow(a, n), pow(b, n))", "pow(mul(a, b), n)")
                .cost(CostHint(flops_delta=-1))
                .priority(12)
                .soundness("幂运算提取: a^n * b^n = (a*b)^n")
                .build(),
            
            # === Exp 分配律 ===
            RuleBuilder("exp_add_to_mul")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("exp(add(a, b))", "mul(exp(a), exp(b))")
                .cost(CostHint(flops_delta=1))
                .priority(5)
                .soundness("指数分配: exp(a+b) = exp(a) * exp(b)")
                .build(),
            
            RuleBuilder("mul_exp_to_exp_add")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("mul(exp(a), exp(b))", "exp(add(a, b))")
                .cost(CostHint(flops_delta=-1))
                .priority(10)
                .soundness("指数合并: exp(a) * exp(b) = exp(a+b)")
                .build(),
            
            # === Log 分配律 ===
            RuleBuilder("log_mul_to_add")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("log(mul(a, b))", "add(log(a), log(b))")
                .cost(CostHint(flops_delta=1))
                .priority(5)
                .soundness("对数分配: log(a*b) = log(a) + log(b)")
                .build(),
            
            RuleBuilder("add_log_to_log_mul")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("add(log(a), log(b))", "log(mul(a, b))")
                .cost(CostHint(flops_delta=-1))
                .priority(10)
                .soundness("对数合并: log(a) + log(b) = log(a*b)")
                .build(),
            
            RuleBuilder("log_div_to_sub")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("log(div(a, b))", "sub(log(a), log(b))")
                .cost(CostHint(flops_delta=1))
                .priority(5)
                .soundness("对数除法: log(a/b) = log(a) - log(b)")
                .build(),
            
            RuleBuilder("log_pow_to_mul")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("log(pow(a, n))", "mul(n, log(a))")
                .cost(CostHint(flops_delta=0))
                .priority(8)
                .soundness("对数幂: log(a^n) = n*log(a)")
                .build(),
            
            # === 平方根分配律 ===
            RuleBuilder("sqrt_mul_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("sqrt(mul(a, b))", "mul(sqrt(a), sqrt(b))")
                .cost(CostHint(flops_delta=1))
                .condition(RuleCondition(
                    custom=lambda ctx: ctx.get("a_nonneg", False) and ctx.get("b_nonneg", False)
                ))
                .priority(3)
                .soundness("平方根分配: sqrt(a*b) = sqrt(a)*sqrt(b) (a,b >= 0)")
                .build(),
            
            RuleBuilder("sqrt_div_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("sqrt(div(a, b))", "div(sqrt(a), sqrt(b))")
                .cost(CostHint(flops_delta=1))
                .condition(RuleCondition(
                    custom=lambda ctx: ctx.get("a_nonneg", False) and ctx.get("b_positive", False)
                ))
                .priority(3)
                .soundness("平方根除法: sqrt(a/b) = sqrt(a)/sqrt(b) (a >= 0, b > 0)")
                .build(),
            
            # === 条件分配律 ===
            RuleBuilder("where_add_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("add(where(c, a1, a2), where(c, b1, b2))",
                        "where(c, add(a1, b1), add(a2, b2))")
                .cost(CostHint(io_delta=-1))
                .priority(8)
                .soundness("条件分配: where(c,a1,a2) + where(c,b1,b2) = where(c, a1+b1, a2+b2)")
                .build(),
            
            RuleBuilder("where_mul_distribute")
                .group(RuleGroup.G6_DISTRIBUTION)
                .pattern("mul(where(c, a1, a2), where(c, b1, b2))",
                        "where(c, mul(a1, b1), mul(a2, b2))")
                .cost(CostHint(io_delta=-1))
                .priority(8)
                .soundness("条件分配: where(c,a1,a2) * where(c,b1,b2) = where(c, a1*b1, a2*b2)")
                .build(),
        ]
