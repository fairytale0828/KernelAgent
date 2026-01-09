# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
代数规则库

规则分组：
- G1: 基础代数 (交换律、结合律、单位元)
- G2: 结构简化 (reshape/transpose 压缩)
- G3: 归约线性 (sum/mean 与点算子交换)
- G4: MatMul 推入推出 (缩放、bias 吸收)
- G5: Softmax 定义展开/折叠
- G6: 分配律 (谨慎使用)
"""

from .base import Rule, RuleSet, RuleGroup
from .basic_algebra import BasicAlgebraRules
from .structural import StructuralRules
from .reduction import ReductionLinearRules
from .matmul import MatmulRules
from .softmax import SoftmaxRules
from .distribution import DistributionRules

__all__ = [
    "Rule",
    "RuleSet", 
    "RuleGroup",
    "BasicAlgebraRules",
    "StructuralRules",
    "ReductionLinearRules",
    "MatmulRules",
    "SoftmaxRules",
    "DistributionRules",
]
