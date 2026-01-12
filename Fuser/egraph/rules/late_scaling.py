# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
Late Scaling 优化规则

核心思想：将归一化操作推迟到 MatMul 之后执行，减少中间张量

变换示例：
  原始: matmul(x * scale, W) 
  优化: matmul(x, W) * scale

这样可以避免创建 x * scale 这个中间张量
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict, Any
from enum import Enum


class ScalingPattern(Enum):
    """缩放模式类型"""
    MUL_RSQRT = "mul_rsqrt"           # x * rsqrt(...)
    MUL_SCALE = "mul_scale"           # x * scale
    DIV_NORM = "div_norm"             # x / norm
    MUL_WEIGHT_RSQRT = "mul_weight_rsqrt"  # x * weight * rsqrt


@dataclass
class ScalingFactor:
    """缩放因子信息"""
    factor_id: int          # E-Class ID of the scaling factor
    factor_type: str        # 'rsqrt', 'weight', 'combined'
    broadcastable: bool     # 是否可广播到 matmul 输出
    shape: Optional[List[int]] = None
    
    # 组成部分（如果是组合因子）
    components: Optional[List['ScalingFactor']] = None


@dataclass  
class LateScalingPattern:
    """Late Scaling 模式匹配结果"""
    original_expr_id: int      # 原始表达式的 E-Class ID
    x_id: int                  # 输入 x 的 E-Class ID
    weight_id: int             # 权重 W 的 E-Class ID
    scaling_factor: ScalingFactor  # 缩放因子
    bias_id: Optional[int]     # bias 的 E-Class ID (如果有)
    pattern_type: ScalingPattern
    
    # 变换信息
    transform_name: str = "late_scaling"
    proof: str = ""


class LateScalingMatcher:
    """
    Late Scaling 模式匹配器
    
    支持的模式：
    1. matmul(mul(x, rsqrt), W) -> mul(matmul(x, W), rsqrt)
    2. matmul(mul(mul(x, rsqrt), weight), W) -> mul(matmul(x, W), combined_scale)
    3. add(matmul(mul(x, scale), W), bias) -> add(mul(matmul(x, W), scale), bias)
    """
    
    def __init__(self, egraph: 'EGraph', max_depth: int = 5):
        self.egraph = egraph
        self.max_depth = max_depth
        self._cache: Dict[int, Optional[LateScalingPattern]] = {}
    
    def find_patterns(self) -> List[LateScalingPattern]:
        """查找所有 Late Scaling 模式"""
        patterns = []
        
        for eclass in self.egraph.iter_eclasses():
            for enode in eclass.nodes:
                # 检查 matmul 节点
                if enode.op == "matmul":
                    pattern = self._match_matmul_scaling(eclass.id, enode)
                    if pattern:
                        patterns.append(pattern)
                
                # 检查 add(matmul(...), bias) 模式
                if enode.op == "add":
                    pattern = self._match_add_matmul_scaling(eclass.id, enode)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _match_matmul_scaling(
        self, 
        eclass_id: int, 
        matmul_node: 'ENode'
    ) -> Optional[LateScalingPattern]:
        """匹配 matmul(scaled_x, W) 模式"""
        if len(matmul_node.children) < 2:
            return None
        
        input_id = matmul_node.children[0]
        weight_id = matmul_node.children[1]
        
        # 递归查找缩放因子
        scaling_result = self._find_scaling_in_expr(input_id, depth=0)
        
        if scaling_result is None:
            return None
        
        x_id, scaling_factor, pattern_type = scaling_result
        
        return LateScalingPattern(
            original_expr_id=eclass_id,
            x_id=x_id,
            weight_id=weight_id,
            scaling_factor=scaling_factor,
            bias_id=None,
            pattern_type=pattern_type,
            proof=f"matmul(mul(x, {scaling_factor.factor_type}), W) -> mul(matmul(x, W), {scaling_factor.factor_type})"
        )
    
    def _match_add_matmul_scaling(
        self, 
        eclass_id: int, 
        add_node: 'ENode'
    ) -> Optional[LateScalingPattern]:
        """匹配 add(matmul(scaled_x, W), bias) 模式"""
        if len(add_node.children) != 2:
            return None
        
        # 检查哪个子节点是 matmul
        for i, child_id in enumerate(add_node.children):
            child_eclass = self.egraph.get_eclass(child_id)
            if child_eclass is None:
                continue
            
            for child_enode in child_eclass.nodes:
                if child_enode.op == "matmul":
                    bias_id = add_node.children[1 - i]
                    
                    # 检查 matmul 的输入是否有缩放
                    pattern = self._match_matmul_scaling(child_id, child_enode)
                    if pattern:
                        pattern.bias_id = bias_id
                        pattern.original_expr_id = eclass_id
                        pattern.proof = f"add(matmul(mul(x, scale), W), bias) -> add(mul(matmul(x, W), scale), bias)"
                        return pattern
        
        return None
    
    def _find_scaling_in_expr(
        self, 
        expr_id: int, 
        depth: int
    ) -> Optional[Tuple[int, ScalingFactor, ScalingPattern]]:
        """
        递归查找表达式中的缩放因子
        
        返回: (原始x的ID, 缩放因子, 模式类型)
        """
        if depth > self.max_depth:
            return None
        
        # 检查缓存
        if expr_id in self._cache:
            cached = self._cache[expr_id]
            if cached:
                return (cached.x_id, cached.scaling_factor, cached.pattern_type)
            return None
        
        eclass = self.egraph.get_eclass(expr_id)
        if eclass is None:
            return None
        
        for enode in eclass.nodes:
            # 模式 1: mul(x, rsqrt(...))
            if enode.op == "mul" and len(enode.children) == 2:
                result = self._check_mul_scaling(enode, depth)
                if result:
                    return result
            
            # 模式 2: div(x, norm)
            if enode.op == "div" and len(enode.children) == 2:
                result = self._check_div_scaling(enode)
                if result:
                    return result
        
        return None
    
    def _check_mul_scaling(
        self, 
        mul_node: 'ENode', 
        depth: int
    ) -> Optional[Tuple[int, ScalingFactor, ScalingPattern]]:
        """检查 mul 节点是否是缩放模式"""
        child0, child1 = mul_node.children
        
        # 检查每个子节点是否是缩放因子
        for i, child_id in enumerate(mul_node.children):
            other_id = mul_node.children[1 - i]
            
            child_eclass = self.egraph.get_eclass(child_id)
            if child_eclass is None:
                continue
            
            for child_enode in child_eclass.nodes:
                # 直接是 rsqrt
                if child_enode.op == "rsqrt":
                    return (
                        other_id,
                        ScalingFactor(
                            factor_id=child_id,
                            factor_type="rsqrt",
                            broadcastable=True,
                        ),
                        ScalingPattern.MUL_RSQRT
                    )
                
                # 是 weight（向量）
                if child_enode.op == "weight":
                    # 继续检查 other_id 是否也有缩放
                    nested = self._find_scaling_in_expr(other_id, depth + 1)
                    if nested:
                        x_id, inner_factor, _ = nested
                        # 组合缩放因子
                        combined = ScalingFactor(
                            factor_id=-1,  # 组合因子没有单一 ID
                            factor_type="combined",
                            broadcastable=True,
                            components=[
                                inner_factor,
                                ScalingFactor(
                                    factor_id=child_id,
                                    factor_type="weight",
                                    broadcastable=True,
                                )
                            ]
                        )
                        return (x_id, combined, ScalingPattern.MUL_WEIGHT_RSQRT)
        
        return None
    
    def _check_div_scaling(
        self, 
        div_node: 'ENode'
    ) -> Optional[Tuple[int, ScalingFactor, ScalingPattern]]:
        """检查 div 节点是否是缩放模式"""
        x_id, norm_id = div_node.children
        
        # div(x, norm) 等价于 mul(x, 1/norm)
        return (
            x_id,
            ScalingFactor(
                factor_id=norm_id,
                factor_type="norm_reciprocal",
                broadcastable=True,
            ),
            ScalingPattern.DIV_NORM
        )


class LateScalingTransformer:
    """
    Late Scaling 变换器
    
    将匹配的模式转换为优化后的表达式
    """
    
    def __init__(self, egraph: 'EGraph'):
        self.egraph = egraph
    
    def apply_transform(
        self, 
        pattern: LateScalingPattern
    ) -> Optional[int]:
        """
        应用 Late Scaling 变换
        
        返回: 新表达式的 E-Class ID
        """
        from ..egraph import ENode
        
        # Step 1: 创建 matmul(x, W)
        matmul_node = ENode(
            op="matmul",
            children=(pattern.x_id, pattern.weight_id),
            attrs=(),
        )
        matmul_id = self.egraph.add(matmul_node)
        
        # Step 2: 应用缩放
        scaled_id = self._apply_scaling(matmul_id, pattern.scaling_factor)
        
        # Step 3: 如果有 bias，添加 add
        if pattern.bias_id is not None:
            add_node = ENode(
                op="add",
                children=(scaled_id, pattern.bias_id),
                attrs=(),
            )
            result_id = self.egraph.add(add_node)
        else:
            result_id = scaled_id
        
        # Step 4: 合并到原始 E-Class（标记为等价）
        self.egraph.merge(pattern.original_expr_id, result_id)
        
        return result_id
    
    def _apply_scaling(
        self, 
        expr_id: int, 
        factor: ScalingFactor
    ) -> int:
        """应用缩放因子"""
        from ..egraph import ENode
        
        if factor.factor_type == "combined" and factor.components:
            # 组合因子：依次应用
            result_id = expr_id
            for component in factor.components:
                result_id = self._apply_scaling(result_id, component)
            return result_id
        
        elif factor.factor_type == "norm_reciprocal":
            # div 模式转换为 div
            div_node = ENode(
                op="div",
                children=(expr_id, factor.factor_id),
                attrs=(),
            )
            return self.egraph.add(div_node)
        
        else:
            # mul 模式
            mul_node = ENode(
                op="mul",
                children=(expr_id, factor.factor_id),
                attrs=(),
            )
            return self.egraph.add(mul_node)


def find_and_apply_late_scaling(egraph: 'EGraph') -> List[LateScalingPattern]:
    """
    查找并应用所有 Late Scaling 变换
    
    返回: 应用的变换列表
    """
    matcher = LateScalingMatcher(egraph)
    patterns = matcher.find_patterns()
    
    transformer = LateScalingTransformer(egraph)
    
    applied = []
    for pattern in patterns:
        result_id = transformer.apply_transform(pattern)
        if result_id is not None:
            applied.append(pattern)
    
    return applied
