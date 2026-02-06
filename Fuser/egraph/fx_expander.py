"""
基于 torch.fx 的操作展开器

将高级 PyTorch 操作（如 rms_norm, linear）展开为低级代数操作，
以便 E-Graph 规则系统能够匹配和变换。

核心思路：
1. 从 subgraph.json 的 source.code 提取 PyTorch 代码
2. 使用 torch.fx.symbolic_trace 获取计算图
3. 将 fx.Graph 转换为低级操作序列
4. 输出可供 E-Graph 处理的 IR
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
import json
import hashlib

# 尝试导入 torch，如果不可用则提供 fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    if HAS_TORCH:
        from torch import fx
        from torch.fx import symbolic_trace, Graph, Node
        HAS_FX = True
    else:
        HAS_FX = False
except ImportError:
    HAS_FX = False


@dataclass
class LowLevelOp:
    """低级操作表示"""
    op: str  # 操作类型: add, mul, div, matmul, rsqrt, mean, etc.
    inputs: List[str]  # 输入张量名称
    output: str  # 输出张量名称
    attrs: Dict[str, Any] = field(default_factory=dict)  # 操作属性
    shape: Optional[List[int]] = None  # 输出形状
    dtype: Optional[str] = None  # 数据类型


@dataclass
class ExpandedGraph:
    """展开后的计算图"""
    ops: List[LowLevelOp]
    inputs: List[Tuple[str, List[int]]]  # [(name, shape), ...]
    outputs: List[Tuple[str, List[int]]]
    weights: Dict[str, List[int]]  # {name: shape}
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# FX 操作映射表
# ============================================================================

# torch 函数到低级操作的映射
TORCH_FUNC_MAP = {
    # 基础算术
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "neg": "neg",
    "abs": "abs",
    "pow": "pow",

    # 数学函数
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "exp": "exp",
    "log": "log",
    "tanh": "tanh",
    "sigmoid": "sigmoid",

    # 归约操作
    "sum": "reduce_sum",
    "mean": "reduce_mean",
    "max": "reduce_max",
    "min": "reduce_min",
    "var": "reduce_var",

    # 矩阵操作
    "matmul": "matmul",
    "mm": "matmul",
    "bmm": "bmm",
    "linear": "linear",

    # 激活函数
    "relu": "relu",
    "gelu": "gelu",
    "silu": "silu",
    "softmax": "softmax",

    # 形状操作
    "reshape": "reshape",
    "view": "reshape",
    "transpose": "transpose",
    "permute": "permute",
    "squeeze": "squeeze",
    "unsqueeze": "unsqueeze",
    "flatten": "flatten",
    "expand": "broadcast",
    "broadcast_to": "broadcast",

    # 归一化
    "layer_norm": "layer_norm",
    "batch_norm": "batch_norm",
    "group_norm": "group_norm",
    "rms_norm": "rms_norm",

    # 其他
    "cat": "concat",
    "stack": "stack",
    "split": "split",
    "chunk": "chunk",
    "where": "where",
    "clamp": "clamp",
}

# F.xxx 函数映射
F_FUNC_MAP = {
    "linear": "linear",
    "relu": "relu",
    "gelu": "gelu",
    "silu": "silu",
    "softmax": "softmax",
    "layer_norm": "layer_norm",
    "batch_norm": "batch_norm",
    "group_norm": "group_norm",
    "dropout": "dropout",
    "embedding": "embedding",
}


class FXExpander:
    """
    基于 torch.fx 的操作展开器

    将 PyTorch 模块或函数展开为低级操作序列
    """

    def __init__(self, expand_norms: bool = True, expand_activations: bool = False):
        """
        Args:
            expand_norms: 是否展开归一化操作 (LayerNorm, RMSNorm 等)
            expand_activations: 是否展开激活函数 (GELU, SiLU 等)
        """
        self.expand_norms = expand_norms
        self.expand_activations = expand_activations
        self._node_counter = 0

    def _normalize_code(self, code: str) -> str:
        """
        规范化代码字符串

        处理常见问题:
        - 不一致的缩进
        - 模块定义代码 (nn.Linear(...))
        - 空代码
        """
        if not code:
            return ""

        # 去除首尾空白
        code = code.strip()

        # 检查是否是模块定义代码（不是 forward 函数）
        if code.startswith("self.") and "nn." in code and "=" in code:
            # 这是 __init__ 中的代码，如 "self.linear = nn.Linear(...)"
            # 无法展开，返回空
            return ""

        # 规范化缩进：找到最小非空行的缩进，然后统一去除
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]

        if not non_empty_lines:
            return ""

        # 计算最小缩进
        min_indent = float('inf')
        for line in non_empty_lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)

        if min_indent == float('inf'):
            min_indent = 0

        # 去除公共缩进
        normalized_lines = []
        for line in lines:
            if line.strip():
                # 去除最小缩进
                if len(line) >= min_indent:
                    normalized_lines.append(line[min_indent:])
                else:
                    normalized_lines.append(line.lstrip())
            else:
                normalized_lines.append("")

        return '\n'.join(normalized_lines)

    def expand_from_code(
        self,
        code: str,
        input_shapes: List[List[int]],
        weight_shapes: Optional[Dict[str, List[int]]] = None,
        dtype: str = "float32"
    ) -> ExpandedGraph:
        """
        从 PyTorch 代码字符串展开计算图

        Args:
            code: PyTorch forward 函数代码
            input_shapes: 输入张量形状列表
            weight_shapes: 权重形状字典
            dtype: 数据类型

        Returns:
            展开后的计算图
        """
        if not HAS_FX:
            # Fallback: 使用 AST 解析
            return self._expand_from_ast(code, input_shapes, weight_shapes, dtype)

        try:
            # 尝试使用 fx trace
            return self._expand_with_fx(code, input_shapes, weight_shapes, dtype)
        except Exception as e:
            # Fallback: 使用 AST 解析
            print(f"FX trace failed ({e}), falling back to AST parsing")
            return self._expand_from_ast(code, input_shapes, weight_shapes, dtype)

    def _expand_with_fx(
        self,
        code: str,
        input_shapes: List[List[int]],
        weight_shapes: Optional[Dict[str, List[int]]],
        dtype: str
    ) -> ExpandedGraph:
        """使用 torch.fx 展开"""
        # 构建可执行模块
        module = self._build_module_from_code(code, weight_shapes)

        if module is None:
            raise ValueError("Failed to build module from code")

        # 创建示例输入
        torch_dtype = getattr(torch, dtype, torch.float32)
        example_inputs = [
            torch.randn(*shape, dtype=torch_dtype)
            for shape in input_shapes
        ]

        # Symbolic trace
        try:
            traced = symbolic_trace(module)
        except Exception as e:
            # 某些模块无法 trace，使用 make_fx
            from torch.fx.experimental.proxy_tensor import make_fx

            def forward_fn(*args):
                return module(*args)
            traced = make_fx(forward_fn)(*example_inputs)

        # 转换 fx.Graph 为 ExpandedGraph
        return self._fx_graph_to_expanded(
            traced.graph,
            input_shapes,
            weight_shapes or {},
            dtype
        )

    def _build_module_from_code(
        self,
        code: str,
        weight_shapes: Optional[Dict[str, List[int]]]
    ) -> Optional[nn.Module]:
        """从代码构建 PyTorch 模块

        注意：以下情况无法使用 fx.trace，会返回 None 让 AST 解析器处理：
        1. 代码包含 self.xxx() 子模块调用（如 self.linear(x)）
        2. 代码是模块定义而非 forward 函数
        3. 代码格式不正确
        """
        # 清理代码
        code = textwrap.dedent(code).strip()

        if not code:
            return None

        # 检查是否包含 self.xxx() 子模块调用 - 这类代码无法用 fx trace
        # 因为我们没有实际的子模块实例
        has_self_method_call = self._has_self_method_call(code)
        has_forward_def = "def forward" in code

        if has_self_method_call and not has_forward_def:
            # 有 self.xxx() 调用但不是完整模块定义，无法 trace
            return None

        # 构建完整的模块定义
        module_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5
'''

        # 添加权重参数
        if weight_shapes:
            for name, shape in weight_shapes.items():
                safe_name = name.replace(".", "_")
                module_code += f"        self.{safe_name} = nn.Parameter(torch.randn({shape}))\n"

        # 添加 forward 方法
        if has_forward_def:
            # 已经是完整的 forward 定义，直接添加（注意缩进）
            # 需要确保 forward 方法有正确的类级别缩进
            forward_lines = code.split('\n')
            module_code += "\n"
            for line in forward_lines:
                if line.strip():
                    module_code += "    " + line + "\n"
                else:
                    module_code += "\n"
        else:
            # 代码片段，包装成 forward
            module_code += "\n    def forward(self, x):\n"
            for line in code.split('\n'):
                if line.strip():
                    module_code += "        " + line.strip() + "\n"
            # 确保有返回值
            if "return" not in code:
                module_code += "        return x\n"

        # 执行代码
        local_ns: Dict[str, Any] = {}
        try:
            exec(module_code, local_ns)
            ModuleClass = local_ns.get("DynamicModule")
            if ModuleClass:
                return ModuleClass()
        except Exception as e:
            print(f"Failed to build module: {e}")

        return None

    def _has_self_method_call(self, code: str) -> bool:
        """检查代码是否包含 self.xxx() 形式的方法调用"""
        import re
        # 匹配 self.xxx( 但排除 self.xxx = 赋值
        # 例如: self.linear(x), self.norm(x) 等
        pattern = r'self\.(\w+)\s*\('
        matches = re.findall(pattern, code)

        # 排除常见的属性访问（不是方法调用）
        excluded = {'eps', 'weight', 'bias', 'scale', 'gamma', 'beta'}
        for match in matches:
            if match.lower() not in excluded:
                return True
        return False

    def _fx_graph_to_expanded(
        self,
        graph: "Graph",
        input_shapes: List[List[int]],
        weight_shapes: Dict[str, List[int]],
        dtype: str
    ) -> ExpandedGraph:
        """将 fx.Graph 转换为 ExpandedGraph"""
        ops: List[LowLevelOp] = []
        node_to_name: Dict[str, str] = {}
        inputs: List[Tuple[str, List[int]]] = []
        outputs: List[Tuple[str, List[int]]] = []

        input_idx = 0

        for node in graph.nodes:
            if node.op == "placeholder":
                # 输入节点
                name = f"input_{input_idx}"
                node_to_name[node.name] = name
                if input_idx < len(input_shapes):
                    inputs.append((name, input_shapes[input_idx]))
                input_idx += 1

            elif node.op == "get_attr":
                # 权重/参数
                name = node.target.replace(".", "_")
                node_to_name[node.name] = name

            elif node.op == "call_function":
                # 函数调用
                low_level_ops = self._expand_call_function(
                    node, node_to_name, dtype
                )
                ops.extend(low_level_ops)

            elif node.op == "call_method":
                # 方法调用
                low_level_ops = self._expand_call_method(
                    node, node_to_name, dtype
                )
                ops.extend(low_level_ops)

            elif node.op == "call_module":
                # 模块调用
                low_level_ops = self._expand_call_module(
                    node, node_to_name, dtype
                )
                ops.extend(low_level_ops)

            elif node.op == "output":
                # 输出节点
                for arg in node.args[0] if isinstance(node.args[0], tuple) else [node.args[0]]:
                    if hasattr(arg, 'name'):
                        out_name = node_to_name.get(arg.name, arg.name)
                        outputs.append((out_name, []))  # 形状需要推导

        return ExpandedGraph(
            ops=ops,
            inputs=inputs,
            outputs=outputs,
            weights=weight_shapes,
            metadata={"expansion_method": "fx"}
        )

    def _expand_call_function(
        self,
        node: "Node",
        node_to_name: Dict[str, str],
        dtype: str
    ) -> List[LowLevelOp]:
        """展开函数调用"""
        ops = []
        func = node.target
        func_name = getattr(func, "__name__", str(func))

        # 获取输入名称
        input_names = []
        for arg in node.args:
            if hasattr(arg, 'name'):
                input_names.append(node_to_name.get(arg.name, arg.name))
            else:
                input_names.append(str(arg))

        # 处理关键字参数
        attrs = dict(node.kwargs)

        output_name = f"t_{self._node_counter}"
        self._node_counter += 1
        node_to_name[node.name] = output_name

        # 特殊处理：展开 RMSNorm
        if func_name in ("rms_norm", "_rms_norm") or "rms" in func_name.lower():
            if self.expand_norms:
                ops.extend(self._expand_rms_norm(
                    input_names, output_name, attrs, dtype))
                return ops

        # 特殊处理：展开 LayerNorm
        if func_name == "layer_norm":
            if self.expand_norms:
                ops.extend(self._expand_layer_norm(
                    input_names, output_name, attrs, dtype))
                return ops

        # 特殊处理：展开 linear -> matmul + add
        if func_name == "linear":
            ops.extend(self._expand_linear(
                input_names, output_name, attrs, dtype))
            return ops

        # 映射到低级操作
        low_level_op = TORCH_FUNC_MAP.get(func_name, func_name)

        ops.append(LowLevelOp(
            op=low_level_op,
            inputs=input_names,
            output=output_name,
            attrs=attrs,
            dtype=dtype
        ))

        return ops

    def _expand_call_method(
        self,
        node: "Node",
        node_to_name: Dict[str, str],
        dtype: str
    ) -> List[LowLevelOp]:
        """展开方法调用"""
        ops = []
        method_name = node.target

        # 获取 self 对象
        self_arg = node.args[0] if node.args else None
        self_name = node_to_name.get(self_arg.name, self_arg.name) if hasattr(
            self_arg, 'name') else "self"

        # 获取其他参数
        input_names = [self_name]
        for arg in node.args[1:]:
            if hasattr(arg, 'name'):
                input_names.append(node_to_name.get(arg.name, arg.name))
            else:
                input_names.append(str(arg))

        attrs = dict(node.kwargs)

        output_name = f"t_{self._node_counter}"
        self._node_counter += 1
        node_to_name[node.name] = output_name

        # 映射方法到操作
        op_map = {
            "view": "reshape",
            "reshape": "reshape",
            "transpose": "transpose",
            "permute": "permute",
            "squeeze": "squeeze",
            "unsqueeze": "unsqueeze",
            "expand": "broadcast",
            "contiguous": "identity",  # 可以忽略
            "mean": "reduce_mean",
            "sum": "reduce_sum",
            "max": "reduce_max",
            "min": "reduce_min",
            "__add__": "add",
            "__mul__": "mul",
            "__sub__": "sub",
            "__truediv__": "div",
            "__matmul__": "matmul",
        }

        low_level_op = op_map.get(method_name, method_name)

        if low_level_op == "identity":
            # 直接传递
            node_to_name[node.name] = self_name
            return ops

        ops.append(LowLevelOp(
            op=low_level_op,
            inputs=input_names,
            output=output_name,
            attrs=attrs,
            dtype=dtype
        ))

        return ops

    def _expand_call_module(
        self,
        node: "Node",
        node_to_name: Dict[str, str],
        dtype: str
    ) -> List[LowLevelOp]:
        """展开模块调用"""
        ops = []
        module_name = node.target

        input_names = []
        for arg in node.args:
            if hasattr(arg, 'name'):
                input_names.append(node_to_name.get(arg.name, arg.name))

        output_name = f"t_{self._node_counter}"
        self._node_counter += 1
        node_to_name[node.name] = output_name

        # 根据模块类型展开
        # 这里简化处理，实际需要获取模块实例来确定类型
        ops.append(LowLevelOp(
            op="module_call",
            inputs=input_names,
            output=output_name,
            attrs={"module": module_name},
            dtype=dtype
        ))

        return ops

    def _expand_rms_norm(
        self,
        input_names: List[str],
        output_name: str,
        attrs: Dict[str, Any],
        dtype: str
    ) -> List[LowLevelOp]:
        """
        展开 RMSNorm 为低级操作

        RMSNorm(x) = x * rsqrt(mean(x^2) + eps)

        展开为:
        1. t1 = mul(x, x)           # x^2
        2. t2 = reduce_mean(t1)     # mean(x^2)
        3. t3 = add(t2, eps)        # mean(x^2) + eps
        4. t4 = rsqrt(t3)           # rsqrt(...)
        5. out = mul(x, t4)         # x * rsqrt(...)
        """
        ops = []
        x = input_names[0] if input_names else "input"
        eps = attrs.get("eps", 1e-5)

        # t1 = x * x
        t1 = f"{output_name}_sq"
        ops.append(LowLevelOp(
            op="mul",
            inputs=[x, x],
            output=t1,
            dtype=dtype
        ))

        # t2 = mean(t1, dim=-1, keepdim=True)
        t2 = f"{output_name}_mean"
        ops.append(LowLevelOp(
            op="reduce_mean",
            inputs=[t1],
            output=t2,
            attrs={"dim": -1, "keepdim": True},
            dtype=dtype
        ))

        # t3 = t2 + eps
        t3 = f"{output_name}_eps"
        ops.append(LowLevelOp(
            op="add",
            inputs=[t2, f"const_{eps}"],
            output=t3,
            attrs={"const_value": eps},
            dtype=dtype
        ))

        # t4 = rsqrt(t3)
        t4 = f"{output_name}_rsqrt"
        ops.append(LowLevelOp(
            op="rsqrt",
            inputs=[t3],
            output=t4,
            dtype=dtype
        ))

        # out = x * t4
        ops.append(LowLevelOp(
            op="mul",
            inputs=[x, t4],
            output=output_name,
            dtype=dtype
        ))

        return ops

    def _expand_layer_norm(
        self,
        input_names: List[str],
        output_name: str,
        attrs: Dict[str, Any],
        dtype: str
    ) -> List[LowLevelOp]:
        """
        展开 LayerNorm 为低级操作

        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

        展开为:
        1. t1 = reduce_mean(x)
        2. t2 = sub(x, t1)
        3. t3 = mul(t2, t2)
        4. t4 = reduce_mean(t3)
        5. t5 = add(t4, eps)
        6. t6 = rsqrt(t5)
        7. t7 = mul(t2, t6)
        8. t8 = mul(t7, gamma)
        9. out = add(t8, beta)
        """
        ops = []
        x = input_names[0] if input_names else "input"
        eps = attrs.get("eps", 1e-5)

        # t1 = mean(x)
        t1 = f"{output_name}_mean"
        ops.append(LowLevelOp(
            op="reduce_mean",
            inputs=[x],
            output=t1,
            attrs={"dim": -1, "keepdim": True},
            dtype=dtype
        ))

        # t2 = x - t1
        t2 = f"{output_name}_centered"
        ops.append(LowLevelOp(
            op="sub",
            inputs=[x, t1],
            output=t2,
            dtype=dtype
        ))

        # t3 = t2 * t2
        t3 = f"{output_name}_sq"
        ops.append(LowLevelOp(
            op="mul",
            inputs=[t2, t2],
            output=t3,
            dtype=dtype
        ))

        # t4 = mean(t3)
        t4 = f"{output_name}_var"
        ops.append(LowLevelOp(
            op="reduce_mean",
            inputs=[t3],
            output=t4,
            attrs={"dim": -1, "keepdim": True},
            dtype=dtype
        ))

        # t5 = t4 + eps
        t5 = f"{output_name}_var_eps"
        ops.append(LowLevelOp(
            op="add",
            inputs=[t4, f"const_{eps}"],
            output=t5,
            attrs={"const_value": eps},
            dtype=dtype
        ))

        # t6 = rsqrt(t5)
        t6 = f"{output_name}_rstd"
        ops.append(LowLevelOp(
            op="rsqrt",
            inputs=[t5],
            output=t6,
            dtype=dtype
        ))

        # t7 = t2 * t6 (normalized)
        t7 = f"{output_name}_norm"
        ops.append(LowLevelOp(
            op="mul",
            inputs=[t2, t6],
            output=t7,
            dtype=dtype
        ))

        # 如果有 gamma 和 beta
        if len(input_names) > 1:
            gamma = input_names[1]
            t8 = f"{output_name}_scaled"
            ops.append(LowLevelOp(
                op="mul",
                inputs=[t7, gamma],
                output=t8,
                dtype=dtype
            ))

            if len(input_names) > 2:
                beta = input_names[2]
                ops.append(LowLevelOp(
                    op="add",
                    inputs=[t8, beta],
                    output=output_name,
                    dtype=dtype
                ))
            else:
                # 重命名最后一个输出
                ops[-1].output = output_name
        else:
            # 重命名最后一个输出
            ops[-1].output = output_name

        return ops

    def _expand_linear(
        self,
        input_names: List[str],
        output_name: str,
        attrs: Dict[str, Any],
        dtype: str
    ) -> List[LowLevelOp]:
        """
        展开 linear 操作为 matmul + add

        F.linear(input, weight, bias) -> matmul(input, weight.T) + bias

        注意：在 E-Graph 规则中，我们假设 weight 已经是正确的形状，
        所以这里直接使用 matmul 而不是 matmul + transpose
        """
        ops = []

        # 获取输入
        x = input_names[0] if len(input_names) > 0 else "input"
        weight = input_names[1] if len(input_names) > 1 else "weight"
        bias = input_names[2] if len(input_names) > 2 else None

        if bias and bias not in ("None", "none", ""):
            # 有 bias: matmul + add
            mm_output = f"{output_name}_mm"
            ops.append(LowLevelOp(
                op="matmul",
                inputs=[x, weight],
                output=mm_output,
                dtype=dtype
            ))
            ops.append(LowLevelOp(
                op="add",
                inputs=[mm_output, bias],
                output=output_name,
                dtype=dtype
            ))
        else:
            # 无 bias: 只有 matmul
            ops.append(LowLevelOp(
                op="matmul",
                inputs=[x, weight],
                output=output_name,
                dtype=dtype
            ))

        return ops

    def _expand_from_ast(
        self,
        code: str,
        input_shapes: List[List[int]],
        weight_shapes: Optional[Dict[str, List[int]]],
        dtype: str
    ) -> ExpandedGraph:
        """
        使用 AST 解析展开（fallback 方案）

        当 torch.fx 不可用或 trace 失败时使用

        支持的代码格式:
        - 完整的函数定义: def forward(self, x): ...
        - 代码片段: variance = x.pow(2).mean(...)
        """
        ops: List[LowLevelOp] = []
        inputs: List[Tuple[str, List[int]]] = []
        outputs: List[Tuple[str, List[int]]] = []

        # 清理和规范化代码
        code = self._normalize_code(code)

        if not code or not code.strip():
            # 空代码，返回空图
            return ExpandedGraph(
                ops=[],
                inputs=[(f"input_{i}", shape)
                        for i, shape in enumerate(input_shapes)],
                outputs=[("input_0", [])],
                weights=weight_shapes or {},
                metadata={"expansion_method": "empty"}
            )

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # 尝试作为函数体解析
            try:
                wrapped = f"def forward(self, x):\n{textwrap.indent(code, '    ')}"
                tree = ast.parse(wrapped)
            except SyntaxError:
                # 最后尝试：作为单行表达式
                try:
                    wrapped = f"result = {code}"
                    tree = ast.parse(wrapped)
                except SyntaxError:
                    # 无法解析，返回空图
                    return ExpandedGraph(
                        ops=[],
                        inputs=[(f"input_{i}", shape)
                                for i, shape in enumerate(input_shapes)],
                        outputs=[("input_0", [])],
                        weights=weight_shapes or {},
                        metadata={"expansion_method": "parse_failed"}
                    )

        # 遍历 AST 提取操作
        visitor = _ASTOpExtractor(self)

        # 初始化变量映射
        visitor._var_map["x"] = "input_0"  # 假设 x 是输入
        visitor._var_map["input"] = "input_0"

        visitor.visit(tree)

        ops = visitor.ops

        # 添加输入
        for i, shape in enumerate(input_shapes):
            inputs.append((f"input_{i}", shape))

        # 推断输出 - 使用返回值或最后一个操作
        if "__return__" in visitor._var_map:
            output_var = visitor._var_map["__return__"]
            outputs.append((output_var, []))
        elif "output" in visitor._var_map:
            outputs.append((visitor._var_map["output"], []))
        elif ops:
            outputs.append((ops[-1].output, []))

        return ExpandedGraph(
            ops=ops,
            inputs=inputs,
            outputs=outputs,
            weights=weight_shapes or {},
            metadata={"expansion_method": "ast"}
        )


class _ASTOpExtractor(ast.NodeVisitor):
    """
    增强的 AST 操作提取器

    支持：
    - 方法调用链: x.pow(2).mean(dim=-1)
    - 二元操作: x * y, x + y
    - 函数调用: torch.rsqrt(x), F.linear(x, w, b)
    - 赋值语句: variance = x.pow(2).mean(...)
    """

    def __init__(self, expander: FXExpander):
        self.expander = expander
        self.ops: List[LowLevelOp] = []
        self._counter = 0
        self._var_map: Dict[str, str] = {}
        self._input_name = "input_0"  # 默认输入名

    def _new_var(self) -> str:
        name = f"t_{self._counter}"
        self._counter += 1
        return name

    def process_expr(self, node: ast.AST) -> str:
        """处理任意表达式，返回结果变量名"""
        if isinstance(node, ast.Name):
            name = node.id
            # 特殊处理 self.xxx
            if name == "self":
                return "self"
            return self._var_map.get(name, name)

        elif isinstance(node, ast.Attribute):
            # 处理 self.weight, self.bias, self.eps 等
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                attr_name = node.attr
                # 常见的属性
                if attr_name in ("weight", "bias", "eps", "norm_weight", "scale"):
                    return attr_name
                return attr_name
            # 处理方法调用的基础对象
            return self.process_expr(node.value)

        elif isinstance(node, ast.Call):
            return self._process_call(node)

        elif isinstance(node, ast.BinOp):
            return self._process_binop(node)

        elif isinstance(node, ast.Constant):
            # 常量值
            return f"const_{node.value}"

        elif isinstance(node, ast.UnaryOp):
            operand = self.process_expr(node.operand)
            if isinstance(node.op, ast.USub):
                output = self._new_var()
                self.ops.append(LowLevelOp(
                    op="neg",
                    inputs=[operand],
                    output=output,
                ))
                return output
            return operand

        elif isinstance(node, ast.Subscript):
            # 处理索引操作 x[0], x[:, -1]
            base = self.process_expr(node.value)
            return base  # 简化处理

        return f"unknown_{self._counter}"

    def _process_call(self, node: ast.Call) -> str:
        """处理函数/方法调用"""
        # 判断是方法调用还是函数调用
        if isinstance(node.func, ast.Attribute):
            # 方法调用: obj.method(args) 或 module.func(args)
            method_name = node.func.attr

            # 检查是否是 torch.xxx 或 F.xxx
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name in ("torch", "F"):
                    return self._process_module_call(module_name, method_name, node.args, node.keywords)
                # 检查是否是 self.xxx(args) - 调用子模块
                if module_name == "self":
                    return self._process_submodule_call(method_name, node.args, node.keywords)

            # 否则是对象方法调用: x.pow(2), x.mean(...)
            obj = self.process_expr(node.func.value)
            return self._process_method_call(obj, method_name, node.args, node.keywords)

        elif isinstance(node.func, ast.Name):
            # 直接函数调用: func(args)
            func_name = node.func.id
            args = [self.process_expr(arg) for arg in node.args]
            kwargs = self._extract_kwargs(node.keywords)

            output = self._new_var()
            low_level_op = TORCH_FUNC_MAP.get(func_name, func_name)
            self.ops.append(LowLevelOp(
                op=low_level_op,
                inputs=args,
                output=output,
                attrs=kwargs,
            ))
            return output

        return self._new_var()

    def _process_submodule_call(
        self,
        submodule: str,
        args: List[ast.AST],
        keywords: List[ast.keyword]
    ) -> str:
        """处理 self.xxx(args) 子模块调用，如 self.linear(x)"""
        processed_args = [self.process_expr(arg) for arg in args]
        kwargs = self._extract_kwargs(keywords)
        output = self._new_var()

        # 识别子模块类型
        submodule_lower = submodule.lower()

        # Linear 层: self.linear(x) -> matmul(x, weight) + bias
        if "linear" in submodule_lower or submodule_lower in ("fc", "proj", "dense"):
            input_tensor = processed_args[0] if processed_args else "input_0"
            mm_output = f"{output}_mm"
            self.ops.append(LowLevelOp(
                op="matmul",
                inputs=[input_tensor, "weight"],
                output=mm_output,
            ))
            # 假设有 bias
            self.ops.append(LowLevelOp(
                op="add",
                inputs=[mm_output, "bias"],
                output=output,
            ))
            return output

        # LayerNorm: self.layer_norm(x) -> 展开为低级操作
        if "layernorm" in submodule_lower or "layer_norm" in submodule_lower or submodule_lower == "norm":
            input_tensor = processed_args[0] if processed_args else "input_0"
            # 简化: 直接作为 layer_norm 操作
            self.ops.append(LowLevelOp(
                op="layer_norm",
                inputs=[input_tensor, "weight", "bias"],
                output=output,
                attrs=kwargs,
            ))
            return output

        # RMSNorm: self.rms_norm(x)
        if "rmsnorm" in submodule_lower or "rms_norm" in submodule_lower:
            input_tensor = processed_args[0] if processed_args else "input_0"
            # 展开 RMSNorm
            eps = kwargs.get("eps", 1e-5)

            # x^2
            sq_output = f"{output}_sq"
            self.ops.append(LowLevelOp(
                op="mul",
                inputs=[input_tensor, input_tensor],
                output=sq_output,
            ))
            # mean(x^2)
            mean_output = f"{output}_mean"
            self.ops.append(LowLevelOp(
                op="reduce_mean",
                inputs=[sq_output],
                output=mean_output,
                attrs={"dim": -1, "keepdim": True},
            ))
            # mean + eps
            add_eps_output = f"{output}_add_eps"
            self.ops.append(LowLevelOp(
                op="add",
                inputs=[mean_output, f"const_{eps}"],
                output=add_eps_output,
            ))
            # rsqrt(mean + eps)
            rsqrt_output = f"{output}_rsqrt"
            self.ops.append(LowLevelOp(
                op="rsqrt",
                inputs=[add_eps_output],
                output=rsqrt_output,
            ))
            # x * rsqrt
            norm_output = f"{output}_norm"
            self.ops.append(LowLevelOp(
                op="mul",
                inputs=[input_tensor, rsqrt_output],
                output=norm_output,
            ))
            # norm * weight
            self.ops.append(LowLevelOp(
                op="mul",
                inputs=[norm_output, "weight"],
                output=output,
            ))
            return output

        # 其他子模块: 保持原样
        self.ops.append(LowLevelOp(
            op=submodule,
            inputs=processed_args,
            output=output,
            attrs=kwargs,
        ))
        return output

    def _process_module_call(
        self,
        module: str,
        func: str,
        args: List[ast.AST],
        keywords: List[ast.keyword]
    ) -> str:
        """处理 torch.xxx() 或 F.xxx() 调用"""
        processed_args = [self.process_expr(arg) for arg in args]
        kwargs = self._extract_kwargs(keywords)
        output = self._new_var()

        # 特殊处理
        if func == "rsqrt":
            self.ops.append(LowLevelOp(
                op="rsqrt",
                inputs=processed_args,
                output=output,
                attrs=kwargs,
            ))
        elif func == "linear":
            # F.linear(input, weight, bias) -> matmul + add
            if len(processed_args) >= 2:
                mm_output = f"{output}_mm"
                self.ops.append(LowLevelOp(
                    op="matmul",
                    inputs=[processed_args[0], processed_args[1]],
                    output=mm_output,
                ))
                if len(processed_args) >= 3 and processed_args[2] not in ("None", "none"):
                    self.ops.append(LowLevelOp(
                        op="add",
                        inputs=[mm_output, processed_args[2]],
                        output=output,
                    ))
                else:
                    # 没有 bias，重命名输出
                    self.ops[-1].output = output
            else:
                self.ops.append(LowLevelOp(
                    op="linear",
                    inputs=processed_args,
                    output=output,
                    attrs=kwargs,
                ))
        elif func == "mean":
            self.ops.append(LowLevelOp(
                op="reduce_mean",
                inputs=processed_args,
                output=output,
                attrs=kwargs,
            ))
        elif func == "sum":
            self.ops.append(LowLevelOp(
                op="reduce_sum",
                inputs=processed_args,
                output=output,
                attrs=kwargs,
            ))
        else:
            low_level_op = TORCH_FUNC_MAP.get(func, func)
            self.ops.append(LowLevelOp(
                op=low_level_op,
                inputs=processed_args,
                output=output,
                attrs=kwargs,
            ))

        return output

    def _process_method_call(
        self,
        obj: str,
        method: str,
        args: List[ast.AST],
        keywords: List[ast.keyword]
    ) -> str:
        """处理对象方法调用: x.pow(2), x.mean(dim=-1)"""
        processed_args = [self.process_expr(arg) for arg in args]
        kwargs = self._extract_kwargs(keywords)
        output = self._new_var()

        # 方法到操作的映射
        method_map = {
            "pow": "pow",
            "mean": "reduce_mean",
            "sum": "reduce_sum",
            "max": "reduce_max",
            "min": "reduce_min",
            "var": "reduce_var",
            "view": "reshape",
            "reshape": "reshape",
            "transpose": "transpose",
            "permute": "permute",
            "squeeze": "squeeze",
            "unsqueeze": "unsqueeze",
            "contiguous": "identity",
            "float": "cast",
            "half": "cast",
            "bfloat16": "cast",
            "to": "cast",
            "clone": "identity",
            "detach": "identity",
        }

        op = method_map.get(method, method)

        if op == "identity":
            # 直接传递
            return obj

        # 构建输入列表
        inputs = [obj] + processed_args

        self.ops.append(LowLevelOp(
            op=op,
            inputs=inputs,
            output=output,
            attrs=kwargs,
        ))

        return output

    def _process_binop(self, node: ast.BinOp) -> str:
        """处理二元操作"""
        left = self.process_expr(node.left)
        right = self.process_expr(node.right)

        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.FloorDiv: "floordiv",
            ast.MatMult: "matmul",
            ast.Pow: "pow",
            ast.Mod: "mod",
        }

        op = op_map.get(type(node.op), "unknown")
        output = self._new_var()

        self.ops.append(LowLevelOp(
            op=op,
            inputs=[left, right],
            output=output,
        ))

        return output

    def _extract_kwargs(self, keywords: List[ast.keyword]) -> Dict[str, Any]:
        """提取关键字参数"""
        kwargs = {}
        for kw in keywords:
            if kw.arg is None:
                continue
            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                if isinstance(kw.value.operand, ast.Constant):
                    kwargs[kw.arg] = -kw.value.operand.value
            elif isinstance(kw.value, ast.Name):
                if kw.value.id in ("True", "true"):
                    kwargs[kw.arg] = True
                elif kw.value.id in ("False", "false"):
                    kwargs[kw.arg] = False
                else:
                    kwargs[kw.arg] = kw.value.id
            elif isinstance(kw.value, ast.NameConstant):
                kwargs[kw.arg] = kw.value.value
        return kwargs

    def visit_Assign(self, node: ast.Assign) -> None:
        """处理赋值语句"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            result = self.process_expr(node.value)
            self._var_map[target_name] = result
        # 不调用 generic_visit，避免重复访问子节点

    def visit_Expr(self, node: ast.Expr) -> None:
        """处理表达式语句"""
        self.process_expr(node.value)
        # 不调用 generic_visit，避免重复访问子节点

    def visit_Return(self, node: ast.Return) -> None:
        """处理 return 语句"""
        if node.value:
            result = self.process_expr(node.value)
            self._var_map["__return__"] = result

    # 保留旧方法以兼容
    def visit_Call(self, node: ast.Call) -> str:
        return self._process_call(node)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        return self._process_binop(node)

    def _get_func_name(self, node: ast.AST) -> str:
        """获取函数名（兼容方法）"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id == "F":
                    return node.attr
                elif node.value.id == "torch":
                    return node.attr
            return node.attr
        return "unknown"

    def _get_operand(self, node: ast.AST) -> str:
        """获取操作数（兼容方法）"""
        return self.process_expr(node)


def expand_subgraph_json(
    item: Dict[str, Any],
    expand_norms: bool = True
) -> Dict[str, Any]:
    """
    展开 subgraph.json 中的单个子图

    Args:
        item: 原始子图 JSON
        expand_norms: 是否展开归一化操作

    Returns:
        展开后的子图 JSON
    """
    expander = FXExpander(expand_norms=expand_norms)

    # 获取源代码
    source = item.get("source", {})
    code = source.get("code", "")

    # 检查代码是否是模块定义代码（无法展开）
    if code:
        code_stripped = code.strip()
        # 模块定义代码，如 "self.linear = nn.Linear(...)"
        if code_stripped.startswith("self.") and "nn." in code_stripped and "=" in code_stripped:
            # 这是 __init__ 中的代码，直接从 ops 展开
            return _expand_from_ops(item, expander)

    if not code:
        # 没有源代码，尝试从 ops 重建
        return _expand_from_ops(item, expander)

    # 获取形状信息
    input_shape = item.get("input_shape", [])
    if isinstance(input_shape, list) and input_shape and not isinstance(input_shape[0], list):
        input_shapes = [input_shape]
    else:
        input_shapes = input_shape if input_shape else [[1, 512]]

    # 获取权重形状
    weight_shapes = item.get("weights_fused", {}) or item.get(
        "weights_original", {})

    dtype = item.get("dtype", "float32")

    # 展开
    try:
        expanded = expander.expand_from_code(
            code, input_shapes, weight_shapes, dtype)

        # 检查展开结果是否有效
        if not expanded.ops or expanded.metadata.get("expansion_method") in ("empty", "parse_failed"):
            # 展开失败或为空，回退到 ops 展开
            return _expand_from_ops(item, expander)

    except Exception as e:
        print(f"Expansion failed: {e}")
        return _expand_from_ops(item, expander)

    # 转换为 JSON 格式
    expanded_ops = []
    for op in expanded.ops:
        expanded_ops.append({
            "op": op.op,
            "inputs": op.inputs,
            "output": op.output,
            **op.attrs
        })

    # 构建结果
    result = {
        **item,
        "ops": expanded_ops,
        "expanded": True,
        "expansion_method": expanded.metadata.get("expansion_method", "unknown"),
        "original_ops": item.get("ops", []),
    }

    return result


def _expand_from_ops(item: Dict[str, Any], expander: FXExpander) -> Dict[str, Any]:
    """从 ops 列表展开"""
    ops = item.get("ops", [])
    expanded_ops = []

    prev_output = "input_0"
    counter = 0

    for op_item in ops:
        op_type = op_item.get("op", "unknown")

        # 特殊处理 rms_norm
        if op_type == "rms_norm":
            eps = op_item.get("eps", 1e-5)
            output = f"t_{counter}"
            counter += 1

            low_level = expander._expand_rms_norm(
                [prev_output], output, {"eps": eps}, "float32"
            )
            for ll_op in low_level:
                expanded_ops.append({
                    "op": ll_op.op,
                    "inputs": ll_op.inputs,
                    "output": ll_op.output,
                    **ll_op.attrs
                })
            prev_output = output

        # 特殊处理 linear -> matmul + add
        elif op_type == "linear":
            output = f"t_{counter}"
            counter += 1

            mm_output = f"{output}_mm"
            expanded_ops.append({
                "op": "matmul",
                "inputs": [prev_output, "weight"],
                "output": mm_output,
            })

            if op_item.get("bias", True):
                expanded_ops.append({
                    "op": "add",
                    "inputs": [mm_output, "bias"],
                    "output": output,
                })
                prev_output = output
            else:
                prev_output = mm_output

        # 特殊处理 layer_norm
        elif op_type == "layer_norm":
            eps = op_item.get("eps", 1e-5)
            output = f"t_{counter}"
            counter += 1

            low_level = expander._expand_layer_norm(
                [prev_output], output, {"eps": eps}, "float32"
            )
            for ll_op in low_level:
                expanded_ops.append({
                    "op": ll_op.op,
                    "inputs": ll_op.inputs,
                    "output": ll_op.output,
                    **ll_op.attrs
                })
            prev_output = output

        else:
            # 其他操作直接保留
            output = f"t_{counter}"
            counter += 1

            expanded_ops.append({
                "op": op_type,
                "inputs": [prev_output],
                "output": output,
                **{k: v for k, v in op_item.items() if k != "op"}
            })
            prev_output = output

    return {
        **item,
        "ops": expanded_ops,
        "expanded": True,
        "expansion_method": "ops",
        "original_ops": item.get("ops", []),
    }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="展开 subgraphs.json 中的高级操作")
    parser.add_argument("--input", "-i", required=True, help="输入文件")
    parser.add_argument("--output", "-o", required=True, help="输出文件")
    parser.add_argument("--no-expand-norms",
                        action="store_true", help="不展开归一化操作")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    expanded_items = []
    for item in items:
        expanded = expand_subgraph_json(
            item, expand_norms=not args.no_expand_norms)
        expanded_items.append(expanded)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(expanded_items, f, indent=2)

    print(f"✓ 展开完成: {len(items)} 个子图")
    for item in expanded_items:
        orig_count = len(item.get("original_ops", []))
        new_count = len(item.get("ops", []))
        print(f"  - {item.get('id', 'unknown')}: {orig_count} -> {new_count} ops")


if __name__ == "__main__":
    main()
