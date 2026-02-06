"""
G2: 结构简化规则

包含：
- reshape/transpose 链压缩
- cast 链压缩
- broadcast 简化
- 无用操作消除
"""

from typing import List

from .base import Rule, RuleBuilder, RuleGroup, CostHint


class StructuralRules:
    """结构简化规则集"""
    
    @staticmethod
    def all() -> List[Rule]:
        """返回所有 G2 规则"""
        return [
            # === Reshape 压缩 ===
            RuleBuilder("reshape_reshape")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("reshape(reshape(a, s1), s2)", "reshape(a, s2)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("reshape 链压缩 (合法时)")
                .build(),
            
            RuleBuilder("reshape_identity")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("reshape(a, same_shape)", "a")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(20)
                .soundness("reshape 到相同形状是恒等变换")
                .build(),
            
            # === Transpose 压缩 ===
            RuleBuilder("transpose_transpose_inv")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("transpose(transpose(a, p), inv_p)", "a")
                .cost(CostHint(io_delta=-2, kernel_delta=-2))
                .priority(15)
                .soundness("transpose 与其逆的组合是恒等变换")
                .build(),
            
            RuleBuilder("transpose_identity")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("transpose(a, identity_perm)", "a")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(20)
                .soundness("恒等置换的 transpose 是恒等变换")
                .build(),
            
            # === Permute 压缩 ===
            RuleBuilder("permute_permute")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("permute(permute(a, p1), p2)", "permute(a, compose(p1, p2))")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("permute 链压缩")
                .build(),
            
            # === Cast 压缩 ===
            RuleBuilder("cast_cast")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("cast(cast(a, t1), t2)", "cast(a, t2)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("cast 链压缩 (合法时)")
                .build(),
            
            RuleBuilder("cast_identity")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("cast(a, same_dtype)", "a")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(20)
                .soundness("cast 到相同类型是恒等变换")
                .build(),
            
            # === Broadcast 简化 ===
            RuleBuilder("broadcast_broadcast")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("broadcast(broadcast(a, s1), s2)", "broadcast(a, s2)")
                .cost(CostHint(io_delta=-1))
                .priority(15)
                .soundness("broadcast 链压缩")
                .build(),
            
            RuleBuilder("broadcast_identity")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("broadcast(a, same_shape)", "a")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(20)
                .soundness("broadcast 到相同形状是恒等变换")
                .build(),
            
            # === View/Contiguous ===
            RuleBuilder("view_contiguous")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("view(contiguous(a), s)", "view(a, s)")
                .cost(CostHint(io_delta=-1))
                .priority(10)
                .soundness("view 前的 contiguous 可能冗余")
                .build(),
            
            RuleBuilder("contiguous_contiguous")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("contiguous(contiguous(a))", "contiguous(a)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("连续的 contiguous 是冗余的")
                .build(),
            
            # === Squeeze/Unsqueeze ===
            RuleBuilder("squeeze_unsqueeze")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("squeeze(unsqueeze(a, d), d)", "a")
                .cost(CostHint(io_delta=-2, kernel_delta=-2))
                .priority(15)
                .soundness("squeeze 和 unsqueeze 在同一维度上互逆")
                .build(),
            
            RuleBuilder("unsqueeze_squeeze")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("unsqueeze(squeeze(a, d), d)", "a")
                .cost(CostHint(io_delta=-2, kernel_delta=-2))
                .priority(15)
                .soundness("unsqueeze 和 squeeze 在同一维度上互逆")
                .build(),
            
            # === Flatten ===
            RuleBuilder("flatten_flatten")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("flatten(flatten(a, s1, e1), s2, e2)", "flatten(a, s1, e2)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(10)
                .soundness("连续的 flatten 可合并")
                .build(),
            
            # === Clone/Copy 消除 ===
            RuleBuilder("clone_clone")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("clone(clone(a))", "clone(a)")
                .cost(CostHint(io_delta=-1, kernel_delta=-1))
                .priority(15)
                .soundness("连续的 clone 是冗余的")
                .build(),
            
            # === Detach ===
            RuleBuilder("detach_detach")
                .group(RuleGroup.G2_STRUCTURAL)
                .pattern("detach(detach(a))", "detach(a)")
                .cost(CostHint(kernel_delta=-1))
                .priority(15)
                .soundness("连续的 detach 是冗余的")
                .build(),
        ]
