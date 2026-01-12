# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
G1: 基础代数规则

包含：
- 交换律
- 结合律
- 单位元/零元
- 减法规范化
"""

from typing import List

from .base import Rule, RuleBuilder, RuleGroup, CostHint


class BasicAlgebraRules:
    """基础代数规则集"""
    
    @staticmethod
    def all() -> List[Rule]:
        """返回所有 G1 规则"""
        return [
            # === 交换律 ===
            RuleBuilder("add_comm")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("add(a, b)", "add(b, a)")
                .bidirectional(True)
                .priority(10)
                .soundness("交换律: a + b = b + a")
                .build(),
            
            RuleBuilder("mul_comm")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("mul(a, b)", "mul(b, a)")
                .bidirectional(True)
                .priority(10)
                .soundness("交换律: a * b = b * a")
                .build(),
            
            RuleBuilder("max_comm")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("max(a, b)", "max(b, a)")
                .bidirectional(True)
                .priority(10)
                .soundness("交换律: max(a, b) = max(b, a)")
                .build(),
            
            RuleBuilder("min_comm")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("min(a, b)", "min(b, a)")
                .bidirectional(True)
                .priority(10)
                .soundness("交换律: min(a, b) = min(b, a)")
                .build(),
            
            # === 结合律 ===
            RuleBuilder("add_assoc_left")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("add(add(a, b), c)", "add(a, add(b, c))")
                .bidirectional(True)
                .priority(8)
                .soundness("结合律: (a + b) + c = a + (b + c)")
                .build(),
            
            RuleBuilder("mul_assoc_left")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("mul(mul(a, b), c)", "mul(a, mul(b, c))")
                .bidirectional(True)
                .priority(8)
                .soundness("结合律: (a * b) * c = a * (b * c)")
                .build(),
            
            # === 幂等性 ===
            RuleBuilder("relu_relu")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("relu(relu(a))", "relu(a)")
                .cost(CostHint(flops_delta=-1, io_delta=-1))
                .priority(20)
                .soundness("幂等性: relu(relu(x)) = relu(x)")
                .build(),
            
            RuleBuilder("abs_abs")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("abs(abs(a))", "abs(a)")
                .cost(CostHint(flops_delta=-1, io_delta=-1))
                .priority(20)
                .soundness("幂等性: abs(abs(x)) = abs(x)")
                .build(),
            
            # === 指数/对数 ===
            RuleBuilder("exp_log")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("exp(log(a))", "a")
                .cost(CostHint(flops_delta=-2, io_delta=-2))
                .priority(20)
                .soundness("逆运算: exp(log(x)) = x (x > 0)")
                .build(),
            
            RuleBuilder("log_exp")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("log(exp(a))", "a")
                .cost(CostHint(flops_delta=-2, io_delta=-2))
                .priority(20)
                .soundness("逆运算: log(exp(x)) = x")
                .build(),
            
            # === 平方根 ===
            RuleBuilder("sqrt_square")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("sqrt(mul(a, a))", "abs(a)")
                .cost(CostHint(flops_delta=-1))
                .priority(15)
                .soundness("sqrt(a^2) = |a|")
                .build(),
            
            RuleBuilder("rsqrt_to_div_sqrt")
                .group(RuleGroup.G1_BASIC_ALGEBRA)
                .pattern("rsqrt(a)", "div(one, sqrt(a))")
                .priority(5)
                .soundness("rsqrt(x) = 1/sqrt(x)")
                .build(),
        ]
