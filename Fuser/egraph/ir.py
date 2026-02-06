"""
统一代数 IR (AlgebraIR) 定义

定义用于 E-Graph 操作的中间表示，包括：
- 张量元信息 (shape, dtype, layout)
- 算子节点 (op, attrs, properties)
- 子图 IR (完整的计算图表示)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union


class Dtype(Enum):
    """数据类型"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, s: Optional[str]) -> "Dtype":
        if s is None:
            return cls.UNKNOWN
        s = s.lower().replace("torch.", "")
        mapping = {
            "float32": cls.FLOAT32, "float": cls.FLOAT32,
            "float16": cls.FLOAT16, "half": cls.FLOAT16,
            "bfloat16": cls.BFLOAT16, "bf16": cls.BFLOAT16,
            "int32": cls.INT32, "int": cls.INT32,
            "int64": cls.INT64, "long": cls.INT64,
            "bool": cls.BOOL,
        }
        return mapping.get(s, cls.UNKNOWN)


class OpProperty(Enum):
    """算子属性标记"""
    POINTWISE = auto()      # 逐元素操作
    REDUCTION = auto()      # 归约操作
    LINEAR = auto()         # 线性操作
    NONLINEAR = auto()      # 非线性操作
    LAYOUT_ONLY = auto()    # 仅布局变换
    COMMUTATIVE = auto()    # 交换律
    ASSOCIATIVE = auto()    # 结合律
    IDEMPOTENT = auto()     # 幂等性
    HAS_NUMERIC_RISK = auto()  # 数值风险 (exp/div/sqrt)


@dataclass(frozen=True)
class TensorMeta:
    """张量元信息"""
    shape: Tuple[Union[int, str], ...]  # 支持符号维度
    dtype: Dtype = Dtype.FLOAT32
    layout: Optional[str] = None  # NCHW, NHWC, etc.

    def __post_init__(self):
        # 确保 shape 是 tuple
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, 'shape', tuple(self.shape))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> Optional[int]:
        """计算元素数量，如果有符号维度则返回 None"""
        result = 1
        for dim in self.shape:
            if isinstance(dim, int):
                result *= dim
            else:
                return None
        return result

    def bytes_size(self) -> Optional[int]:
        """计算字节大小"""
        numel = self.numel
        if numel is None:
            return None
        dtype_bytes = {
            Dtype.FLOAT32: 4, Dtype.FLOAT16: 2, Dtype.BFLOAT16: 2,
            Dtype.INT32: 4, Dtype.INT64: 8, Dtype.BOOL: 1,
        }
        return numel * dtype_bytes.get(self.dtype, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorMeta":
        return cls(
            shape=tuple(d.get("shape", [])),
            dtype=Dtype.from_str(d.get("dtype")),
            layout=d.get("layout"),
        )


@dataclass(frozen=True)
class OpNode:
    """
    算子节点 - E-Graph 的基本单元

    设计为不可变 (frozen) 以支持哈希和去重
    """
    op: str  # 操作类型
    children: Tuple[str, ...]  # 子节点 ID 列表
    attrs: Tuple[Tuple[str, Any], ...]  # 属性 (作为元组以支持哈希)
    output_meta: Optional[TensorMeta] = None

    def __post_init__(self):
        # 确保 children 是 tuple
        if not isinstance(self.children, tuple):
            object.__setattr__(self, 'children', tuple(self.children))
        # 确保 attrs 是 tuple of tuples
        if not isinstance(self.attrs, tuple):
            attrs = tuple((k, v) for k, v in self.attrs) if isinstance(
                self.attrs, dict) else tuple(self.attrs)
            object.__setattr__(self, 'attrs', attrs)

    @property
    def attrs_dict(self) -> Dict[str, Any]:
        """获取属性字典"""
        return dict(self.attrs)

    def get_attr(self, key: str, default: Any = None) -> Any:
        """获取单个属性"""
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    def with_children(self, new_children: Tuple[str, ...]) -> "OpNode":
        """创建具有新子节点的副本"""
        return OpNode(
            op=self.op,
            children=new_children,
            attrs=self.attrs,
            output_meta=self.output_meta,
        )

    def with_attrs(self, **new_attrs) -> "OpNode":
        """创建具有新属性的副本"""
        attrs_dict = self.attrs_dict
        attrs_dict.update(new_attrs)
        return OpNode(
            op=self.op,
            children=self.children,
            attrs=tuple(attrs_dict.items()),
            output_meta=self.output_meta,
        )

    @property
    def properties(self) -> FrozenSet[OpProperty]:
        """推导算子属性"""
        props = set()

        # 逐元素操作
        pointwise_ops = {"add", "sub", "mul", "div", "neg", "abs", "sqrt", "rsqrt",
                         "exp", "log", "tanh", "sigmoid", "relu", "gelu", "silu",
                         "sin", "cos", "where", "clamp", "cast"}
        if self.op in pointwise_ops:
            props.add(OpProperty.POINTWISE)

        # 归约操作
        reduction_ops = {"reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
                         "sum", "mean", "max", "min", "softmax", "layer_norm", "rms_norm"}
        if self.op in reduction_ops:
            props.add(OpProperty.REDUCTION)

        # 线性操作
        linear_ops = {"add", "sub", "mul", "matmul", "conv2d", "linear",
                      "reduce_sum", "reduce_mean", "transpose", "reshape"}
        if self.op in linear_ops:
            props.add(OpProperty.LINEAR)

        # 非线性操作
        nonlinear_ops = {"relu", "gelu", "silu", "tanh", "sigmoid", "softmax",
                         "exp", "log", "sqrt", "rsqrt"}
        if self.op in nonlinear_ops:
            props.add(OpProperty.NONLINEAR)

        # 布局操作
        layout_ops = {"reshape", "transpose", "permute", "view", "contiguous",
                      "squeeze", "unsqueeze", "flatten", "broadcast"}
        if self.op in layout_ops:
            props.add(OpProperty.LAYOUT_ONLY)

        # 交换律
        commutative_ops = {"add", "mul", "max", "min"}
        if self.op in commutative_ops:
            props.add(OpProperty.COMMUTATIVE)

        # 结合律
        associative_ops = {"add", "mul"}
        if self.op in associative_ops:
            props.add(OpProperty.ASSOCIATIVE)

        # 幂等性
        idempotent_ops = {"relu", "abs", "max", "min"}
        if self.op in idempotent_ops:
            props.add(OpProperty.IDEMPOTENT)

        # 数值风险
        risky_ops = {"exp", "log", "div", "sqrt", "rsqrt", "softmax"}
        if self.op in risky_ops:
            props.add(OpProperty.HAS_NUMERIC_RISK)

        return frozenset(props)

    def structural_hash(self) -> str:
        """计算结构哈希 (忽略子节点具体值，用于模式匹配)"""
        h = hashlib.sha256()
        h.update(self.op.encode())
        h.update(str(len(self.children)).encode())
        h.update(json.dumps(dict(self.attrs), sort_keys=True).encode())
        return h.hexdigest()[:16]

    def __hash__(self) -> int:
        return hash((self.op, self.children, self.attrs))

    def __eq__(self, other) -> bool:
        if not isinstance(other, OpNode):
            return False
        return (self.op == other.op and
                self.children == other.children and
                self.attrs == other.attrs)


@dataclass
class SubgraphIR:
    """
    子图中间表示

    使用 DAG 结构表示计算图，每个节点有唯一 ID
    """
    id: str
    nodes: Dict[str, OpNode]  # node_id -> OpNode
    inputs: List[Tuple[str, TensorMeta]]  # [(input_id, meta), ...]
    outputs: List[str]  # [output_node_id, ...]
    weights: Dict[str, TensorMeta]  # weight_name -> meta
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 验证图结构
        self._validate()

    def _validate(self):
        """验证图结构完整性"""
        all_ids = set(self.nodes.keys()) | {inp[0] for inp in self.inputs}
        for node_id, node in self.nodes.items():
            for child in node.children:
                if child not in all_ids:
                    raise ValueError(
                        f"Node {node_id} references unknown child {child}")
        for out_id in self.outputs:
            if out_id not in self.nodes:
                raise ValueError(f"Output {out_id} not found in nodes")

    def topological_order(self) -> List[str]:
        """返回拓扑排序的节点 ID 列表"""
        visited = set()
        order = []
        input_ids = {inp[0] for inp in self.inputs}

        def visit(node_id: str):
            if node_id in visited or node_id in input_ids:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                for child in node.children:
                    visit(child)
                order.append(node_id)

        for out_id in self.outputs:
            visit(out_id)

        return order

    def get_node(self, node_id: str) -> Optional[OpNode]:
        return self.nodes.get(node_id)

    def add_node(self, node_id: str, node: OpNode) -> None:
        self.nodes[node_id] = node

    def clone(self) -> "SubgraphIR":
        """深拷贝"""
        return SubgraphIR(
            id=self.id,
            nodes=dict(self.nodes),
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            weights=dict(self.weights),
            metadata=dict(self.metadata),
        )

    def structural_signature(self) -> str:
        """计算结构签名用于去重"""
        h = hashlib.sha256()
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            h.update(node.structural_hash().encode())
        return h.hexdigest()[:32]


def parse_subgraph_json(item: Dict[str, Any]) -> SubgraphIR:
    """
    将 subgraphs.json 中的单个条目解析为 SubgraphIR
    """
    sg_id = item.get(
        "id", f"sg_{hash(json.dumps(item, sort_keys=True)) & 0xFFFFFFFF:08x}")

    # 解析输入
    input_shape = item.get("input_shape") or item.get("inputs", [[]])[0]
    if not isinstance(input_shape, list):
        input_shape = []
    input_meta = TensorMeta(
        shape=tuple(input_shape),
        dtype=Dtype.from_str(item.get("dtype")),
        layout=item.get("data_layout"),
    )
    inputs = [("input_0", input_meta)]

    # 处理多输入情况
    multi_inputs = item.get("inputs")
    if isinstance(multi_inputs, list) and len(multi_inputs) > 1:
        inputs = []
        for i, inp_shape in enumerate(multi_inputs):
            meta = TensorMeta(
                shape=tuple(inp_shape) if isinstance(
                    inp_shape, list) else (inp_shape,),
                dtype=Dtype.from_str(item.get("dtype")),
            )
            inputs.append((f"input_{i}", meta))

    # 解析权重
    weights = {}
    for key in ["weights_fused", "weights_original", "weights"]:
        w = item.get(key)
        if isinstance(w, dict):
            for name, shape in w.items():
                if isinstance(shape, list):
                    weights[name] = TensorMeta(shape=tuple(shape))

    # 解析 ops 列表构建节点
    nodes = {}
    ops_list = item.get("ops", [])
    prev_output = inputs[0][0] if inputs else "input_0"

    # 检查是否是展开后的格式（带有 inputs/output 字段）
    is_expanded_format = any(
        isinstance(op, dict) and ("inputs" in op or "output" in op)
        for op in ops_list
    )

    # 用于追踪节点输出名到 node_id 的映射
    output_name_to_id: Dict[str, str] = {}
    input_names = {inp[0] for inp in inputs}
    weight_names = set(weights.keys())

    for i, op_dict in enumerate(ops_list):
        if not isinstance(op_dict, dict):
            continue

        op_name = op_dict.get("op", "unknown")

        # 提取属性（排除特殊字段）
        special_keys = {"op", "input_shape",
                        "output_shape", "inputs", "output"}
        attrs = {}
        for k, v in op_dict.items():
            if k not in special_keys:
                attrs[k] = v

        # 确定输出形状
        out_shape = op_dict.get("output_shape")
        if out_shape:
            out_meta = TensorMeta(
                shape=tuple(out_shape) if isinstance(out_shape, list) else (),
                dtype=Dtype.from_str(op_dict.get("dtype")),
            )
        else:
            out_meta = None

        if is_expanded_format:
            # 展开格式：使用显式的 inputs 和 output
            op_inputs = op_dict.get("inputs", [prev_output])
            op_output = op_dict.get("output", f"t_{i}")

            # 解析输入引用
            children = []
            for inp in op_inputs:
                if inp in input_names:
                    children.append(inp)
                elif inp in output_name_to_id:
                    children.append(output_name_to_id[inp])
                elif inp in weight_names:
                    # 权重引用，创建权重节点（如果尚未创建）
                    weight_node_id = f"w_{inp}"
                    if weight_node_id not in nodes:
                        weight_node = OpNode(
                            op="weight",
                            children=(),
                            attrs=(("name", inp),),
                            output_meta=weights.get(inp),
                        )
                        nodes[weight_node_id] = weight_node
                    children.append(weight_node_id)
                elif inp.startswith("const_"):
                    # 常量引用，创建常量节点
                    const_id = f"const_{i}_{len(children)}"
                    const_val = attrs.get(
                        "const_value", inp.replace("const_", ""))
                    const_node = OpNode(
                        op="const",
                        children=(),
                        attrs=(("value", const_val),),
                        output_meta=None,
                    )
                    nodes[const_id] = const_node
                    children.append(const_id)
                else:
                    # 未知引用，可能是权重名（不在 weights 字典中）
                    # 创建一个占位符权重节点
                    weight_node_id = f"w_{inp}"
                    if weight_node_id not in nodes:
                        weight_node = OpNode(
                            op="weight",
                            children=(),
                            attrs=(("name", inp),),
                            output_meta=None,
                        )
                        nodes[weight_node_id] = weight_node
                    children.append(weight_node_id)

            node_id = op_output
            output_name_to_id[op_output] = node_id

            node = OpNode(
                op=op_name,
                children=tuple(children),
                attrs=tuple(attrs.items()),
                output_meta=out_meta,
            )
            nodes[node_id] = node
            prev_output = node_id
        else:
            # 原始格式：线性链式结构
            node_id = f"n_{i}"

            node = OpNode(
                op=op_name,
                children=(prev_output,),
                attrs=tuple(attrs.items()),
                output_meta=out_meta,
            )
            nodes[node_id] = node
            prev_output = node_id

    # 确定输出
    outputs = [prev_output] if nodes else []

    # 解析输出形状
    output_shape = item.get("output_shape", [])

    return SubgraphIR(
        id=sg_id,
        nodes=nodes,
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        metadata={
            "original": item,
            "type": item.get("type"),
            "where": item.get("where"),
            "count": item.get("count", 1),
            "source": item.get("source"),
            "output_shape": output_shape,
        },
    )


def ir_to_json(ir: SubgraphIR) -> Dict[str, Any]:
    """
    将 SubgraphIR 转换回 JSON 格式
    """
    # 重建 ops 列表
    ops = []
    for node_id in ir.topological_order():
        node = ir.nodes[node_id]
        op_dict = {
            "op": node.op,
            "inputs": list(node.children),
            "output": node_id,
            **node.attrs_dict
        }
        if node.output_meta:
            op_dict["output_shape"] = list(node.output_meta.shape)
            if node.output_meta.dtype != Dtype.UNKNOWN:
                op_dict["dtype"] = node.output_meta.dtype.value
        ops.append(op_dict)

    # 构建输出
    result = {
        "id": ir.id,
        "ops": ops,
    }

    # 输入形状
    if len(ir.inputs) == 1:
        result["input_shape"] = list(ir.inputs[0][1].shape)
    elif len(ir.inputs) > 1:
        result["inputs"] = [list(inp[1].shape) for inp in ir.inputs]

    # 输出形状
    if ir.metadata.get("output_shape"):
        result["output_shape"] = ir.metadata["output_shape"]
    elif ir.outputs and ir.outputs[0] in ir.nodes:
        out_node = ir.nodes[ir.outputs[0]]
        if out_node.output_meta:
            result["output_shape"] = list(out_node.output_meta.shape)

    # 权重
    if ir.weights:
        result["weights_fused"] = {
            name: list(meta.shape) for name, meta in ir.weights.items()
        }

    # 元数据
    for key in ["type", "where", "count", "source", "data_layout", "dtype"]:
        if key in ir.metadata and ir.metadata[key] is not None:
            result[key] = ir.metadata[key]

    # 变换信息
    if "transforms" in ir.metadata:
        result["transforms"] = ir.metadata["transforms"]
    if "proof" in ir.metadata:
        result["proof"] = ir.metadata["proof"]
    if "cost_estimate" in ir.metadata:
        result["cost_estimate"] = ir.metadata["cost_estimate"]

    return result
