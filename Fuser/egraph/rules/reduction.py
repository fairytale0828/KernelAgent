"""
G3: 归约线性规则

包含：
- sum/mean 与点算子交换
- max/min 与仿射变换
- 归约分配律
"""

from typing import List

from .base import Rule, RuleBuilder, RuleGroup, CostHint, RuleCondition


class ReductionLinearRules:
    """归约线性规则集"""
    
    @staticmethod
    def all() -> List[Rule]:
        """返回所有 G3 规则"""
        return [
            # === Sum 线性 ===
            RuleBuilder("sum_add_distribute")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("sum(add(a, b), ax)", "add(sum(a, ax), sum(b, ax))")
                .bidirectional(True)
                .priority(10)
                .soundness("sum 对 add 的分配律: sum(a+b) = sum(a) + sum(b)")
                .build(),
            
            RuleBuilder("sum_mul_factor_out")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("sum(mul(c, a), ax)", "mul(c, sum(a, ax))")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("c_independent_of_ax", True)
                ))
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("常数因子提出: sum(c*a) = c*sum(a) (c 不依赖归约轴)")
                .build(),
            
            RuleBuilder("sum_sub_distribute")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("sum(sub(a, b), ax)", "sub(sum(a, ax), sum(b, ax))")
                .bidirectional(True)
                .priority(10)
                .soundness("sum 对 sub 的分配律: sum(a-b) = sum(a) - sum(b)")
                .build(),
            
            # === Mean 线性 ===
            RuleBuilder("mean_add_distribute")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("mean(add(a, b), ax)", "add(mean(a, ax), mean(b, ax))")
                .bidirectional(True)
                .priority(10)
                .soundness("mean 对 add 的分配律")
                .build(),
            
            RuleBuilder("mean_mul_factor_out")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("mean(mul(c, a), ax)", "mul(c, mean(a, ax))")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("c_independent_of_ax", True)
                ))
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("常数因子提出: mean(c*a) = c*mean(a)")
                .build(),
            
            # === Max/Min 与仿射 ===
            RuleBuilder("max_add_constant")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("max(add(a, c), ax)", "add(max(a, ax), c)")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("c_independent_of_ax", True)
                ))
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("max(a+c) = max(a)+c (c 不依赖归约轴)")
                .build(),
            
            RuleBuilder("min_add_constant")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("min(add(a, c), ax)", "add(min(a, ax), c)")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("c_independent_of_ax", True)
                ))
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("min(a+c) = min(a)+c (c 不依赖归约轴)")
                .build(),
            
            RuleBuilder("max_mul_positive")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("max(mul(c, a), ax)", "mul(c, max(a, ax))")
                .condition(RuleCondition(
                    custom=lambda ctx: ctx.get("c_positive", False) and ctx.get("c_independent_of_ax", True)
                ))
                .priority(8)
                .soundness("max(c*a) = c*max(a) (c > 0 且不依赖归约轴)")
                .build(),
            
            # === 归约与布局 ===
            RuleBuilder("sum_transpose")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("sum(transpose(a, p), ax)", "transpose(sum(a, inv_ax), q)")
                .priority(5)
                .soundness("sum 与 transpose 交换 (调整归约轴)")
                .build(),
            
            RuleBuilder("sum_reshape")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("sum(reshape(a, s), ax)", "reshape(sum(a, mapped_ax), new_s)")
                .priority(5)
                .soundness("sum 与 reshape 交换 (需要映射归约轴)")
                .build(),
            
            # === 归约融合 ===
            RuleBuilder("sum_sum_merge")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("sum(sum(a, ax1), ax2)", "sum(a, merged_axes)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("连续的 sum 可合并")
                .build(),
            
            RuleBuilder("mean_to_sum_div")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("mean(a, ax)", "div(sum(a, ax), size_of_ax)")
                .priority(5)
                .soundness("mean 定义展开: mean(a) = sum(a) / n")
                .build(),
            
            # === Softmax 相关归约 ===
            RuleBuilder("logsumexp_stability")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("log(sum(exp(a), ax))", "add(max(a, ax), log(sum(exp(sub(a, max(a, ax))), ax)))")
                .priority(8)
                .soundness("logsumexp 数值稳定形式")
                .build(),
            
            # === Variance/Std ===
            RuleBuilder("var_expand")
                .group(RuleGroup.G3_REDUCTION_LINEAR)
                .pattern("var(a, ax)", "mean(mul(sub(a, mean(a, ax)), sub(a, mean(a, ax))), ax)")
                .priority(5)
                .soundness("方差定义展开: var(a) = mean((a - mean(a))^2)")
                .build(),
        ]
