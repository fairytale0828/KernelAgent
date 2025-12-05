# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""算子类别识别器 - 基于LLM的智能推断"""

import json
import logging
import re
from typing import Optional, List, Dict, Any
from enum import Enum


class OpType(str, Enum):
    """算子类别枚举"""
    ELEMENTWISE = "elementwise"  # 逐元素操作
    REDUCE = "reduce"  # 归约操作
    MATMUL = "matmul"  # 矩阵乘法
    CONV2D = "conv2d"  # 2D卷积
    CONV1D = "conv1d"  # 1D卷积
    CONV3D = "conv3d"  # 3D卷积
    POOLING = "pooling"  # 池化操作
    NORMALIZATION = "normalization"  # 归一化操作
    ATTENTION = "attention"  # 注意力机制
    EMBEDDING = "embedding"  # 嵌入操作
    TRANSPOSE = "transpose"  # 转置操作
    GATHER_SCATTER = "gather_scatter"  # 聚集/分散操作
    SORT = "sort"  # 排序操作
    COMPOSITE = "composite"  # 复合操作
    UNKNOWN = "unknown"  # 未知类型


class OpClassifier:
    """算子类别识别器 - 基于LLM的智能推断"""

    def __init__(self, provider=None, model_name: str = "gpt-4"):
        """
        初始化算子分类器

        Args:
            provider: LLM provider实例
            model_name: 使用的模型名称
        """
        self.provider = provider
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_operation_keywords()

    def _setup_operation_keywords(self):
        """设置操作类型关键词映射"""
        self.operation_keywords = {
            "matmul": ["matmul", "matrix multiplication", "gemm", "dot product", "mm", "bmm", "@", "矩阵乘"],
            "conv2d": ["conv2d", "convolution 2d", "2d conv", "卷积"],
            "conv1d": ["conv1d", "convolution 1d", "1d conv"],
            "conv3d": ["conv3d", "convolution 3d", "3d conv"],
            "elementwise": ["elementwise", "element-wise", "add", "multiply", "mul", "relu", "sigmoid", "tanh", "逐元素"],
            "reduce": ["reduce", "reduction", "sum", "mean", "max", "min", "归约"],
            "normalization": ["norm", "batchnorm", "layernorm", "groupnorm", "instancenorm", "归一化"],
            "attention": ["attention", "self-attention", "multi-head", "注意力"],
            "pooling": ["pool", "maxpool", "avgpool", "pooling", "池化"],
            "transpose": ["transpose", "permute", "reshape", "转置"],
            "embedding": ["embedding", "embed", "lookup", "嵌入"],
            "gather_scatter": ["gather", "scatter", "index", "聚集", "分散"],
            "sort": ["sort", "argsort", "topk", "排序"],
            "composite": ["composite", "multiple", "combination", "复合"]
        }

    def classify_operator(
        self, 
        problem_description: str, 
        pytorch_code: Optional[str] = None,
        operation_name: Optional[str] = None
    ) -> OpType:
        """
        识别算子类别

        Args:
            problem_description: 问题描述
            pytorch_code: PyTorch算子代码（可选）
            operation_name: 操作名称（可选）

        Returns:
            OpType: 识别出的算子类别
        """
        # 首先尝试基于规则的快速分类
        rule_based_type = self._rule_based_classify(problem_description, pytorch_code, operation_name)
        if rule_based_type != OpType.UNKNOWN:
            self.logger.info(f"Rule-based classification: {rule_based_type.value}")
            return rule_based_type

        # 如果规则无法识别，使用LLM
        if self.provider:
            try:
                llm_result = self._llm_classify(problem_description, pytorch_code, operation_name)
                llm_type = llm_result.get("primary_operation", OpType.UNKNOWN)
                confidence = llm_result.get("confidence", 0.0)
                self.logger.info(
                    f"LLM-based classification: {llm_type.value} "
                    f"(confidence: {confidence:.2f})"
                )
                return llm_type
            except Exception as e:
                self.logger.warning(f"LLM classification failed: {e}")

        # 默认返回UNKNOWN
        self.logger.warning("Unable to classify operator, returning UNKNOWN")
        return OpType.UNKNOWN

    def _rule_based_classify(
        self, 
        problem_description: str, 
        pytorch_code: Optional[str] = None,
        operation_name: Optional[str] = None
    ) -> OpType:
        """基于规则的快速分类"""
        text = problem_description.lower()
        if pytorch_code:
            text += " " + pytorch_code.lower()
        if operation_name:
            text += " " + operation_name.lower()

        # 计算每种操作类型的匹配分数
        scores = {}
        for op_type, keywords in self.operation_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[op_type] = score

        # 如果没有匹配，返回UNKNOWN
        if not scores:
            return OpType.UNKNOWN

        # 选择得分最高的操作类型
        best_op = max(scores, key=scores.get)
        best_score = scores[best_op]

        # 特殊处理：归一化操作优先级高于归约
        if "normalization" in scores and scores["normalization"] > 0:
            if any(kw in text for kw in ["layernorm", "batchnorm", "groupnorm", "instancenorm"]):
                return OpType.NORMALIZATION

        # 如果得分足够高（至少2个关键词匹配），返回结果
        if best_score >= 2:
            try:
                return OpType(best_op)
            except ValueError:
                return OpType.UNKNOWN

        # 单个关键词匹配，需要更严格的验证
        if best_score == 1:
            # 对于某些明确的操作类型，单个关键词也足够
            high_confidence_ops = ["matmul", "attention", "embedding", "conv2d", "conv3d"]
            if best_op in high_confidence_ops:
                try:
                    return OpType(best_op)
                except ValueError:
                    return OpType.UNKNOWN

        return OpType.UNKNOWN

    def _llm_classify(
        self, 
        problem_description: str, 
        pytorch_code: Optional[str] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用LLM进行分类"""
        prompt = self._build_classification_prompt(problem_description, pytorch_code, operation_name)

        messages = [{"role": "user", "content": prompt}]
        response = self.provider.get_response(self.model_name, messages, max_tokens=1000)

        # 解析LLM响应
        return self._parse_llm_response(response.content)

    def _build_classification_prompt(
        self, 
        problem_description: str, 
        pytorch_code: Optional[str] = None,
        operation_name: Optional[str] = None
    ) -> str:
        """构建分类提示"""
        op_types_desc = "\n".join([
            f"- {op.value}: {self._get_op_type_description(op)}" 
            for op in OpType if op not in [OpType.UNKNOWN, OpType.COMPOSITE]
        ])

        prompt = f"""你是一个专业的GPU kernel操作类型分析专家。请分析以下信息，推断其核心操作类型。

## 操作名称
{operation_name or "Unknown"}

## 问题描述
{problem_description}
"""

        if pytorch_code:
            prompt += f"""

## PyTorch代码
```python
{pytorch_code}
```
"""

        prompt += f"""

## 支持的操作类型
{op_types_desc}
- composite: 复合操作（多个基础操作的组合）

## 分析要求
1. 仔细分析问题描述和代码
2. 识别主要的计算操作
3. 如果包含多个操作，判断是否为复合操作
4. 选择最符合的操作类型

请返回JSON格式的结果：
```json
{{
    "primary_operation": "主要操作类型",
    "secondary_operations": ["次要操作1", "次要操作2"],
    "is_composite": true/false,
    "confidence": 0.95,
    "reasoning": "推理过程说明"
}}
```
"""
        return prompt

    def _get_op_type_description(self, op_type: OpType) -> str:
        """获取算子类型描述"""
        descriptions = {
            OpType.ELEMENTWISE: "逐元素操作，如加法、乘法、激活函数等",
            OpType.REDUCE: "归约操作，如sum、mean、max、min、softmax等",
            OpType.MATMUL: "矩阵乘法操作（包括batch_matmul、gemm等）",
            OpType.CONV2D: "2D卷积操作（包括conv2d、depthwise_conv等）",
            OpType.CONV1D: "1D卷积操作",
            OpType.CONV3D: "3D卷积操作",
            OpType.POOLING: "池化操作，如maxpool、avgpool",
            OpType.NORMALIZATION: "归一化操作，如layernorm、batchnorm",
            OpType.ATTENTION: "注意力机制相关操作",
            OpType.EMBEDDING: "嵌入查找操作",
            OpType.TRANSPOSE: "张量转置或维度重排",
            OpType.GATHER_SCATTER: "索引聚集或分散操作",
            OpType.SORT: "排序相关操作",
            OpType.COMPOSITE: "复合操作（多个基础操作的组合）",
        }
        return descriptions.get(op_type, "未知类型")

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试直接解析JSON
            result = json.loads(response)
            return self._normalize_llm_result(result)
        except json.JSONDecodeError:
            # 尝试提取JSON代码块
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    return self._normalize_llm_result(result)
                except json.JSONDecodeError:
                    pass

            # 如果都失败了，尝试从文本中提取信息
            return self._extract_from_text(response)

    def _normalize_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """规范化LLM返回结果"""
        primary_op = result.get("primary_operation", "unknown")
        
        # 转换为OpType
        try:
            op_type = OpType(primary_op.lower())
        except ValueError:
            op_type = OpType.UNKNOWN

        return {
            "primary_operation": op_type,
            "secondary_operations": result.get("secondary_operations", []),
            "is_composite": result.get("is_composite", False),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", "")
        }

    def _extract_from_text(self, response: str) -> Dict[str, Any]:
        """从文本响应中提取操作类型信息"""
        response_lower = response.lower()

        # 计算每种操作类型的匹配分数
        scores = {}
        for op_type, keywords in self.operation_keywords.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            if score > 0:
                scores[op_type] = score

        # 选择得分最高的操作类型
        if scores:
            primary_operation = max(scores, key=scores.get)
            confidence = min(scores[primary_operation] * 0.2, 0.8)  # 基于匹配数量的置信度
            
            # 转换为OpType
            try:
                op_type = OpType(primary_operation)
            except ValueError:
                op_type = OpType.UNKNOWN
        else:
            op_type = OpType.UNKNOWN
            confidence = 0.1

        return {
            "primary_operation": op_type,
            "secondary_operations": [],
            "is_composite": "composite" in response_lower or "multiple" in response_lower,
            "confidence": confidence,
            "reasoning": "基于关键词匹配的fallback推断"
        }

    def get_operation_hierarchy(self, operation_type: OpType) -> Dict[str, Any]:
        """获取操作类型的层次结构信息"""
        hierarchy = {
            OpType.MATMUL: {
                "category": "compute_intensive",
                "complexity": "medium",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            OpType.CONV2D: {
                "category": "compute_intensive",
                "complexity": "high",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            OpType.CONV1D: {
                "category": "compute_intensive",
                "complexity": "medium",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            OpType.CONV3D: {
                "category": "compute_intensive",
                "complexity": "very_high",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            OpType.ELEMENTWISE: {
                "category": "memory_bound",
                "complexity": "low",
                "memory_pattern": "simple",
                "parallelization": "very_high"
            },
            OpType.REDUCE: {
                "category": "mixed",
                "complexity": "medium",
                "memory_pattern": "irregular",
                "parallelization": "medium"
            },
            OpType.NORMALIZATION: {
                "category": "mixed",
                "complexity": "medium",
                "memory_pattern": "structured",
                "parallelization": "medium"
            },
            OpType.ATTENTION: {
                "category": "compute_intensive",
                "complexity": "high",
                "memory_pattern": "complex",
                "parallelization": "high"
            },
            OpType.POOLING: {
                "category": "memory_bound",
                "complexity": "low",
                "memory_pattern": "structured",
                "parallelization": "high"
            },
            OpType.TRANSPOSE: {
                "category": "memory_bound",
                "complexity": "low",
                "memory_pattern": "irregular",
                "parallelization": "high"
            },
            OpType.EMBEDDING: {
                "category": "memory_bound",
                "complexity": "low",
                "memory_pattern": "irregular",
                "parallelization": "medium"
            },
            OpType.GATHER_SCATTER: {
                "category": "memory_bound",
                "complexity": "medium",
                "memory_pattern": "irregular",
                "parallelization": "medium"
            },
            OpType.SORT: {
                "category": "mixed",
                "complexity": "high",
                "memory_pattern": "irregular",
                "parallelization": "low"
            },
            OpType.COMPOSITE: {
                "category": "mixed",
                "complexity": "high",
                "memory_pattern": "complex",
                "parallelization": "medium"
            }
        }
        return hierarchy.get(operation_type, {
            "category": "unknown",
            "complexity": "unknown",
            "memory_pattern": "unknown",
            "parallelization": "unknown"
        })

    @staticmethod
    def get_all_op_types() -> List[OpType]:
        """获取所有算子类型"""
        return [op for op in OpType if op != OpType.UNKNOWN]
