# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
G4: MatMul 推入推出规则

包含：
- Bias/残差吸收
- 缩放推入推出
- MatMul 分配律
- BatchMatMul 优化
"""

from typing import List

from .base import Rule, RuleBuilder, RuleGroup, CostHint, RuleCondition


class MatmulRules:
    """MatMul 相关规则集"""
    
    @staticmethod
    def all() -> List[Rule]:
        """返回所有 G4 规则"""
        return [
            # === Bias 吸收 ===
            RuleBuilder("matmul_add_bias")
                .group(RuleGroup.G4_MATMUL)
                .pattern("add(matmul(X, W), b)", "matmul_bias(X, W, b)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("MatMul + Bias 融合: Y = XW + b")
                .build(),
            
            RuleBuilder("linear_bias_fuse")
                .group(RuleGroup.G4_MATMUL)
                .pattern("add(linear(X, W), b)", "linear(X, W, b)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("Linear + Bias 融合")
                .build(),
            
            # === 残差吸收 ===
            RuleBuilder("matmul_add_residual")
                .group(RuleGroup.G4_MATMUL)
                .pattern("add(add(matmul(X, W), b), r)", "add(matmul(X, W), add(b, r))")
                .bidirectional(True)
                .priority(10)
                .soundness("残差重组: (XW + b) + r = XW + (b + r)")
                .build(),
            
            # === 列缩放推入 ===
            RuleBuilder("matmul_scale_col_push_in")
                .group(RuleGroup.G4_MATMUL)
                .pattern("mul(matmul(X, W), s)", "matmul(X, mul(W, s))")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("s_broadcasts_on_N", True)
                ))
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("列缩放推入: (XW) * s = X * (W * s) (s 广播在 N 维)")
                .build(),
            
            # === 行缩放推入 ===
            RuleBuilder("matmul_scale_row_push_in")
                .group(RuleGroup.G4_MATMUL)
                .pattern("mul(matmul(X, W), s)", "matmul(mul(X, s), W)")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("s_broadcasts_on_M", True)
                ))
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("行缩放推入: (XW) * s = (X * s) * W (s 广播在 M 维)")
                .build(),
            
            # === 缩放推出 ===
            RuleBuilder("matmul_scale_push_out")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(mul(X, s), W)", "mul(matmul(X, W), s)")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("s_independent_of_K", True)
                ))
                .priority(8)
                .soundness("缩放推出: (X*s)W = (XW)*s (s 不依赖收缩维)")
                .build(),
            
            # === RMSNorm + MatMul 融合 (核心规则) ===
            RuleBuilder("rmsnorm_matmul_div_pushdown")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(div(X, rms), W)", "div(matmul(X, W), rms)")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("rms_broadcasts_compatible", True)
                ))
                .cost(CostHint(io_delta=-1, temp_delta=-1))
                .priority(18)
                .soundness("RMSNorm 除法后推: (X/rms)W = (XW)/rms")
                .build(),
            
            RuleBuilder("rmsnorm_matmul_mul_pushdown")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(mul(X, rsqrt_rms), W)", "mul(matmul(X, W), rsqrt_rms)")
                .condition(RuleCondition(
                    axis_condition=lambda ctx: ctx.get("rsqrt_broadcasts_compatible", True)
                ))
                .cost(CostHint(io_delta=-1, temp_delta=-1))
                .priority(18)
                .soundness("RMSNorm rsqrt 后推: (X*rsqrt)W = (XW)*rsqrt")
                .build(),
            
            # === LayerNorm + MatMul ===
            RuleBuilder("layernorm_matmul_affine_push")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(add(mul(norm_x, gamma), beta), W)", 
                        "add(mul(matmul(norm_x, W), gamma), matmul_vec(beta, W))")
                .priority(10)
                .soundness("LayerNorm 仿射变换与 MatMul 交换")
                .build(),
            
            # === MatMul 分配律 (谨慎使用) ===
            RuleBuilder("matmul_add_distribute_right")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(A, add(B, C))", "add(matmul(A, B), matmul(A, C))")
                .cost(CostHint(flops_delta=1))  # 增加 FLOPs
                .priority(3)  # 低优先级
                .soundness("MatMul 右分配律: A(B+C) = AB + AC")
                .build(),
            
            RuleBuilder("matmul_add_distribute_left")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(add(A, B), C)", "add(matmul(A, C), matmul(B, C))")
                .cost(CostHint(flops_delta=1))
                .priority(3)
                .soundness("MatMul 左分配律: (A+B)C = AC + BC")
                .build(),
            
            # === LoRA 融合 (核心规则) ===
            RuleBuilder("lora_fusion")
                .group(RuleGroup.G4_MATMUL)
                .pattern("add(matmul(X, W), matmul(matmul(X, B), A))",
                        "matmul(X, add(W, matmul(B, A)))")
                .cost(CostHint(io_delta=-1, temp_delta=-1))
                .priority(15)
                .soundness("LoRA 融合: XW + XBA = X(W + BA)")
                .build(),
            
            RuleBuilder("lora_concat_fusion")
                .group(RuleGroup.G4_MATMUL)
                .pattern("add(matmul(X, W), matmul(matmul(X, B), A))",
                        "matmul_lora(X, W, B, A)")
                .cost(CostHint(io_delta=-2, temp_delta=-2, kernel_delta=-2))
                .priority(16)
                .soundness("LoRA concat 融合: XW + XBA -> concat kernel")
                .build(),
            
            # === Transpose 与 MatMul ===
            RuleBuilder("matmul_transpose_right")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(A, transpose(B))", "matmul_nt(A, B)")
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("MatMul 右转置吸收: A @ B^T")
                .build(),
            
            RuleBuilder("matmul_transpose_left")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(transpose(A), B)", "matmul_tn(A, B)")
                .cost(CostHint(io_delta=-1))
                .priority(12)
                .soundness("MatMul 左转置吸收: A^T @ B")
                .build(),
            
            RuleBuilder("matmul_transpose_both")
                .group(RuleGroup.G4_MATMUL)
                .pattern("matmul(transpose(A), transpose(B))", "transpose(matmul(B, A))")
                .priority(10)
                .soundness("A^T @ B^T = (BA)^T")
                .build(),
            
            # === BatchMatMul ===
            RuleBuilder("bmm_unbatched_right")
                .group(RuleGroup.G4_MATMUL)
                .pattern("bmm(A, broadcast(B))", "matmul_broadcast(A, B)")
                .cost(CostHint(io_delta=-1))
                .priority(10)
                .soundness("BMM 右操作数无 batch 维度时优化")
                .build(),
            
            # === Einsum 规范化 ===
            RuleBuilder("einsum_to_matmul")
                .group(RuleGroup.G4_MATMUL)
                .pattern("einsum('ij,jk->ik', A, B)", "matmul(A, B)")
                .priority(15)
                .soundness("Einsum 到 MatMul 规范化")
                .build(),
            
            RuleBuilder("einsum_to_bmm")
                .group(RuleGroup.G4_MATMUL)
                .pattern("einsum('bij,bjk->bik', A, B)", "bmm(A, B)")
                .priority(15)
                .soundness("Einsum 到 BMM 规范化")
                .build(),
        ]
