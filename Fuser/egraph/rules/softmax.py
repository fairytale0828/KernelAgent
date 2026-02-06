"""
G5: Softmax 定义展开/折叠规则

包含：
- Softmax 展开为稳定形式
- Softmax 模式折叠
- LogSoftmax 优化
- Flash Attention 相关模式
"""

from typing import List

from .base import Rule, RuleBuilder, RuleGroup, CostHint, RuleCondition


class SoftmaxRules:
    """Softmax 相关规则集"""
    
    @staticmethod
    def all() -> List[Rule]:
        """返回所有 G5 规则"""
        return [
            # === Softmax 定义展开 (稳定形式) ===
            RuleBuilder("softmax_expand_stable")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("softmax(x, ax)", 
                        "div(exp(sub(x, max(x, ax))), sum(exp(sub(x, max(x, ax))), ax))")
                .priority(5)
                .soundness("Softmax 稳定形式展开: softmax(x) = exp(x-max(x)) / sum(exp(x-max(x)))")
                .build(),
            
            # === Softmax 折叠 ===
            RuleBuilder("softmax_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("div(exp(sub(x, max(x, ax))), sum(exp(sub(x, max(x, ax))), ax))",
                        "softmax(x, ax)")
                .cost(CostHint(io_delta=-2, kernel_delta=-3))
                .priority(15)
                .soundness("Softmax 模式折叠")
                .build(),
            
            # === LogSoftmax ===
            RuleBuilder("log_softmax_expand")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("log_softmax(x, ax)",
                        "sub(sub(x, max(x, ax)), log(sum(exp(sub(x, max(x, ax))), ax)))")
                .priority(5)
                .soundness("LogSoftmax 展开")
                .build(),
            
            RuleBuilder("log_softmax_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("log(softmax(x, ax))", "log_softmax(x, ax)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("log(softmax(x)) = log_softmax(x)")
                .build(),
            
            RuleBuilder("log_softmax_from_parts")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("sub(sub(x, max(x, ax)), log(sum(exp(sub(x, max(x, ax))), ax)))",
                        "log_softmax(x, ax)")
                .cost(CostHint(io_delta=-3, kernel_delta=-4))
                .priority(15)
                .soundness("LogSoftmax 模式折叠")
                .build(),
            
            # === Softmax 与缩放 ===
            RuleBuilder("softmax_scale")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("softmax(mul(x, s), ax)", "softmax(x, ax)")
                .condition(RuleCondition(
                    custom=lambda ctx: False  # 这个规则实际上不成立，仅作示例
                ))
                .priority(1)
                .soundness("注意：softmax(x*s) != softmax(x)，除非 s=1")
                .build(),
            
            RuleBuilder("softmax_temperature")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("softmax(div(x, T), ax)", "softmax_temp(x, T, ax)")
                .priority(10)
                .soundness("Temperature scaling: softmax(x/T)")
                .build(),
            
            # === Flash Attention 模式 ===
            RuleBuilder("attention_pattern_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("matmul(softmax(div(matmul(Q, transpose(K)), sqrt_dk), ax), V)",
                        "scaled_dot_product_attention(Q, K, V, sqrt_dk)")
                .cost(CostHint(io_delta=-3, kernel_delta=-4, temp_delta=-2))
                .priority(20)
                .soundness("Scaled Dot-Product Attention 模式折叠")
                .build(),
            
            RuleBuilder("attention_with_mask_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("matmul(softmax(add(div(matmul(Q, transpose(K)), sqrt_dk), mask), ax), V)",
                        "scaled_dot_product_attention_mask(Q, K, V, sqrt_dk, mask)")
                .cost(CostHint(io_delta=-4, kernel_delta=-5, temp_delta=-3))
                .priority(20)
                .soundness("带 Mask 的 Attention 模式折叠")
                .build(),
            
            # === Multi-Head Attention ===
            RuleBuilder("mha_qkv_split_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("attention(split(linear(x, Wqkv), 3))",
                        "multi_head_attention(x, Wqkv)")
                .cost(CostHint(io_delta=-2, kernel_delta=-3))
                .priority(18)
                .soundness("MHA QKV 投影融合")
                .build(),
            
            # === Softmax 数值稳定性 ===
            RuleBuilder("softmax_add_stability")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("softmax(x, ax)", "softmax(sub(x, max(x, ax)), ax)")
                .bidirectional(True)
                .priority(8)
                .soundness("Softmax 数值稳定性: softmax(x) = softmax(x - max(x))")
                .build(),
            
            # === Cross Entropy ===
            RuleBuilder("cross_entropy_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("neg(sum(mul(y, log_softmax(x, ax)), ax))",
                        "cross_entropy(x, y, ax)")
                .cost(CostHint(io_delta=-2, kernel_delta=-3))
                .priority(15)
                .soundness("Cross Entropy 模式折叠")
                .build(),
            
            RuleBuilder("nll_loss_fold")
                .group(RuleGroup.G5_SOFTMAX)
                .pattern("neg(gather(log_softmax(x, ax), idx))",
                        "nll_loss(x, idx, ax)")
                .cost(CostHint(io_delta=-1, kernel_delta=-2))
                .priority(15)
                .soundness("NLL Loss 模式折叠")
                .build(),
        ]
