#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
"""
计算图聚合器

对 compute_graph/ 目录下的每个完整计算图进行聚合。
每个计算图是独立处理的，不会出现变体混淆的问题。

修复的问题：
1. 添加数据流验证，确保融合的子图确实有数据依赖
2. 更灵活的模式匹配，支持 type 字段和 ops 列表
3. 正确合并 codegen_hints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def aggregate_single_compute_graph(
    compute_graph: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    聚合单个计算图

    Args:
        compute_graph: 计算图（包含 subgraphs 列表）

    Returns:
        (聚合后的子图列表, 统计信息)
    """
    subgraphs = compute_graph.get("subgraphs", [])

    # 构建数据流图（用于验证连接性）
    dataflow_graph = _build_dataflow_graph(subgraphs)

    # 识别融合模式
    fusions_found = []
    aggregated_subgraphs = []
    used_indices: Set[int] = set()

    # 模式 1: Flash Attention (matmul → softmax → matmul)
    flash_attention_fusions = _find_flash_attention_pattern(
        subgraphs, dataflow_graph)
    for fusion in flash_attention_fusions:
        fusions_found.append(fusion)
        for idx in fusion["indices"]:
            used_indices.add(idx)
        aggregated_subgraphs.append(fusion["fused_subgraph"])

    # 模式 2: MLP (linear → activation → linear)
    mlp_fusions = _find_mlp_pattern(subgraphs, used_indices, dataflow_graph)
    for fusion in mlp_fusions:
        fusions_found.append(fusion)
        for idx in fusion["indices"]:
            used_indices.add(idx)
        aggregated_subgraphs.append(fusion["fused_subgraph"])

    # 模式 3: MatMul + Bias + Activation
    matmul_act_fusions = _find_matmul_activation_pattern(
        subgraphs, used_indices, dataflow_graph)
    for fusion in matmul_act_fusions:
        fusions_found.append(fusion)
        for idx in fusion["indices"]:
            used_indices.add(idx)
        aggregated_subgraphs.append(fusion["fused_subgraph"])

    # 添加未融合的子图
    for i, sg in enumerate(subgraphs):
        if i not in used_indices:
            aggregated_subgraphs.append(sg)

    stats = {
        "input_count": len(subgraphs),
        "output_count": len(aggregated_subgraphs),
        "fusions_applied": len(fusions_found),
        "fusion_details": fusions_found,
    }

    return aggregated_subgraphs, stats


def _build_dataflow_graph(subgraphs: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
    """
    构建数据流图

    返回: 子图索引 -> 依赖的子图索引集合
    """
    # 构建输出到索引的映射
    output_to_idx: Dict[str, int] = {}
    for i, sg in enumerate(subgraphs):
        sg_id = sg.get("id", f"sg_{i}")
        output_to_idx[sg_id] = i

        # 也记录 output_shape 对应的索引（用于形状匹配）
        output_shape = sg.get("output_shape")
        if output_shape:
            shape_key = str(output_shape)
            # 不覆盖已有的映射
            if shape_key not in output_to_idx:
                output_to_idx[shape_key] = i

    # 构建依赖图
    dataflow: Dict[int, Set[int]] = {i: set() for i in range(len(subgraphs))}

    for i, sg in enumerate(subgraphs):
        # 检查输入是否来自其他子图的输出
        input_shape = sg.get("input_shape")

        # 检查前一个子图的输出是否匹配当前子图的输入
        if i > 0:
            prev_sg = subgraphs[i - 1]
            prev_output = prev_sg.get("output_shape")

            # 形状匹配检查
            if prev_output and input_shape:
                if _shapes_compatible(prev_output, input_shape):
                    dataflow[i].add(i - 1)

        # 检查 ops 中的输入引用
        ops = sg.get("ops", [])
        for op in ops:
            if isinstance(op, dict):
                op_inputs = op.get("inputs", [])
                for inp in op_inputs:
                    if isinstance(inp, str) and inp in output_to_idx:
                        dep_idx = output_to_idx[inp]
                        if dep_idx != i:
                            dataflow[i].add(dep_idx)

    return dataflow


def _shapes_compatible(shape1: Any, shape2: Any) -> bool:
    """检查两个形状是否兼容（可以连接）"""
    if shape1 is None or shape2 is None:
        return True  # 无法判断时假设兼容

    # 转换为列表
    if not isinstance(shape1, list):
        shape1 = [shape1]
    if not isinstance(shape2, list):
        shape2 = [shape2]

    # 检查维度数是否相同
    if len(shape1) != len(shape2):
        return False

    # 检查每个维度
    for d1, d2 in zip(shape1, shape2):
        if d1 != d2 and d1 != -1 and d2 != -1:
            return False

    return True


def _verify_dataflow_connection(
    indices: List[int],
    dataflow_graph: Dict[int, Set[int]],
) -> bool:
    """
    验证子图序列是否有数据流连接

    检查 indices[i+1] 是否依赖于 indices[i]
    """
    for i in range(len(indices) - 1):
        current_idx = indices[i]
        next_idx = indices[i + 1]

        # 检查 next_idx 是否依赖于 current_idx
        if current_idx not in dataflow_graph.get(next_idx, set()):
            # 也检查是否是连续的（隐式依赖）
            if next_idx != current_idx + 1:
                return False

    return True


def _get_op_types(subgraph: Dict[str, Any]) -> List[str]:
    """获取子图中的操作类型列表"""
    ops = subgraph.get("ops", [])
    op_types = []
    for op in ops:
        if isinstance(op, dict):
            op_type = op.get("op", "")
            if op_type and op_type not in ("weight", "const"):
                op_types.append(op_type)
    return op_types


def _merge_codegen_hints(
    subgraphs: List[Dict[str, Any]],
    base_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """合并多个子图的 codegen_hints"""
    merged = dict(base_hints)

    for sg in subgraphs:
        existing_hints = sg.get("codegen_hints", {})
        if existing_hints:
            for k, v in existing_hints.items():
                if k not in merged:
                    merged[k] = v
                elif k == "original_ops" and isinstance(v, dict):
                    # 合并 original_ops
                    if k not in merged:
                        merged[k] = {}
                    merged[k].update(v)

    return merged


def _collect_weights(subgraphs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """收集多个子图的权重信息"""
    weights_fused = {}
    for sg in subgraphs:
        wf = sg.get("weights_fused") or sg.get("weights", {})
        if wf:
            weights_fused.update(wf)
    return weights_fused if weights_fused else None


def _find_flash_attention_pattern(
    subgraphs: List[Dict[str, Any]],
    dataflow_graph: Dict[int, Set[int]],
) -> List[Dict[str, Any]]:
    """
    查找 Flash Attention 模式

    模式: matmul → softmax/online_softmax → matmul
    支持更灵活的匹配：
    - 基于 type 字段
    - 基于 ops 列表中的操作
    - 验证数据流连接
    """
    fusions = []

    for i in range(len(subgraphs) - 2):
        sg1 = subgraphs[i]
        sg2 = subgraphs[i + 1]
        sg3 = subgraphs[i + 2]

        # 获取操作类型
        ops1 = _get_op_types(sg1)
        ops2 = _get_op_types(sg2)
        ops3 = _get_op_types(sg3)

        # 检查类型（支持 type 字段和 ops 列表）
        type1 = sg1.get("type", "")
        type2 = sg2.get("type", "")
        type3 = sg3.get("type", "")

        # 判断是否是 matmul
        is_matmul1 = (type1 in ("matmul", "linear") or
                      any(op in ops1 for op in ["matmul", "matmul_bias", "linear"]))
        is_matmul3 = (type3 in ("matmul", "linear") or
                      any(op in ops3 for op in ["matmul", "matmul_bias", "linear"]))

        # 判断是否是 softmax
        is_softmax = (type2 in ("softmax", "online_softmax", "scaled_softmax") or
                      any(op in ops2 for op in ["softmax", "online_softmax"]))

        if is_matmul1 and is_softmax and is_matmul3:
            # 验证数据流连接
            if not _verify_dataflow_connection([i, i + 1, i + 2], dataflow_graph):
                continue

            # 检查是否有 scale 操作
            has_scale = any(op in ops2 for op in ["mul", "div"])

            # 合并 codegen_hints
            base_hints = {
                "use_flash_attention": True,
                "online_softmax": "online_softmax" in ops2,
                "has_scale": has_scale,
                "original_ops": {
                    "qk_matmul": ops1,
                    "softmax": ops2,
                    "av_matmul": ops3,
                }
            }
            merged_hints = _merge_codegen_hints([sg1, sg2, sg3], base_hints)

            # 创建融合子图
            fused_sg = {
                "id": f"fused_flash_attention_{i}",
                "type": "flash_attention",
                "ops": [{"op": "flash_attention"}],
                "input_shape": sg1.get("input_shape") or (sg1.get("inputs", [[]])[0] if sg1.get("inputs") else None),
                "output_shape": sg3.get("output_shape"),
                "weights_fused": _collect_weights([sg1, sg2, sg3]),
                "fused_from": [sg1.get("id"), sg2.get("id"), sg3.get("id")],
                "codegen_hints": merged_hints,
                "where": f"Fused: {sg1.get('where', '')} + {sg2.get('where', '')} + {sg3.get('where', '')}",
            }

            fusions.append({
                "pattern": "flash_attention",
                "indices": [i, i + 1, i + 2],
                "fused_subgraph": fused_sg,
            })

    return fusions


def _find_mlp_pattern(
    subgraphs: List[Dict[str, Any]],
    used_indices: Set[int],
    dataflow_graph: Dict[int, Set[int]],
) -> List[Dict[str, Any]]:
    """
    查找 MLP 模式

    模式: linear → activation (gelu/relu/silu) → linear
    支持更灵活的匹配和数据流验证
    """
    fusions = []
    activation_ops = ["gelu", "relu", "silu", "sigmoid", "tanh", "swish"]

    for i in range(len(subgraphs) - 2):
        if i in used_indices or (i + 1) in used_indices or (i + 2) in used_indices:
            continue

        sg1 = subgraphs[i]
        sg2 = subgraphs[i + 1]
        sg3 = subgraphs[i + 2]

        # 获取操作类型
        ops1 = _get_op_types(sg1)
        ops2 = _get_op_types(sg2)
        ops3 = _get_op_types(sg3)

        type1 = sg1.get("type", "")
        type2 = sg2.get("type", "")
        type3 = sg3.get("type", "")

        # 检查是否是 MLP 模式
        is_linear1 = (type1 in ("linear", "matmul") or
                      any(op in ops1 for op in ["matmul", "matmul_bias", "linear"]))
        is_linear2 = (type3 in ("linear", "matmul") or
                      any(op in ops3 for op in ["matmul", "matmul_bias", "linear"]))
        is_activation = (type2 in ("activation", "gelu", "relu", "silu", "sigmoid") or
                         any(op in ops2 for op in activation_ops))

        if is_linear1 and is_activation and is_linear2:
            # 验证数据流连接
            if not _verify_dataflow_connection([i, i + 1, i + 2], dataflow_graph):
                continue

            # 确定激活函数类型
            activation_type = None
            for op in activation_ops:
                if op in ops2 or type2 == op:
                    activation_type = op
                    break

            if not activation_type:
                activation_type = "unknown"

            # 合并 codegen_hints
            base_hints = {
                "fused_mlp": True,
                "activation": activation_type,
                "original_ops": {
                    "linear1": ops1,
                    "activation": ops2,
                    "linear2": ops3,
                }
            }
            merged_hints = _merge_codegen_hints([sg1, sg2, sg3], base_hints)

            fused_sg = {
                "id": f"fused_mlp_{activation_type}_{i}",
                "type": f"mlp_{activation_type}",
                "ops": [{"op": f"fused_mlp_{activation_type}"}],
                "input_shape": sg1.get("input_shape"),
                "output_shape": sg3.get("output_shape"),
                "weights_fused": _collect_weights([sg1, sg2, sg3]),
                "fused_from": [sg1.get("id"), sg2.get("id"), sg3.get("id")],
                "codegen_hints": merged_hints,
                "where": f"Fused MLP: {sg1.get('where', '')}",
            }

            fusions.append({
                "pattern": f"mlp_{activation_type}",
                "indices": [i, i + 1, i + 2],
                "fused_subgraph": fused_sg,
            })

    return fusions


def _find_matmul_activation_pattern(
    subgraphs: List[Dict[str, Any]],
    used_indices: Set[int],
    dataflow_graph: Dict[int, Set[int]],
) -> List[Dict[str, Any]]:
    """
    查找 MatMul + Activation 模式

    模式: matmul → activation (可选 bias)
    """
    fusions = []
    activation_ops = ["gelu", "relu", "silu", "sigmoid", "tanh"]

    for i in range(len(subgraphs) - 1):
        if i in used_indices or (i + 1) in used_indices:
            continue

        sg1 = subgraphs[i]
        sg2 = subgraphs[i + 1]

        ops1 = _get_op_types(sg1)
        ops2 = _get_op_types(sg2)

        type1 = sg1.get("type", "")
        type2 = sg2.get("type", "")

        # 检查是否是 matmul + activation
        is_matmul = (type1 in ("linear", "matmul") or
                     any(op in ops1 for op in ["matmul", "matmul_bias", "linear"]))
        is_activation = (type2 in ("activation", "gelu", "relu", "silu", "sigmoid") or
                         any(op in ops2 for op in activation_ops))

        if is_matmul and is_activation:
            # 验证数据流连接
            if not _verify_dataflow_connection([i, i + 1], dataflow_graph):
                continue

            # 确定激活函数类型
            activation_type = None
            for op in activation_ops:
                if op in ops2 or type2 == op:
                    activation_type = op
                    break

            if not activation_type:
                continue

            # 合并 codegen_hints
            base_hints = {
                "matmul_bias_fused": "add" in ops1 or "bias" in str(ops1),
                "activation": activation_type,
                "original_ops": {
                    "matmul": ops1,
                    "activation": ops2,
                }
            }
            merged_hints = _merge_codegen_hints([sg1, sg2], base_hints)

            fused_sg = {
                "id": f"fused_matmul_{activation_type}_{i}",
                "type": f"matmul_{activation_type}",
                "ops": [{"op": f"matmul_{activation_type}"}],
                "input_shape": sg1.get("input_shape"),
                "output_shape": sg2.get("output_shape"),
                "weights_fused": _collect_weights([sg1, sg2]),
                "fused_from": [sg1.get("id"), sg2.get("id")],
                "codegen_hints": merged_hints,
                "where": f"Fused: {sg1.get('where', '')} + {sg2.get('where', '')}",
            }

            fusions.append({
                "pattern": f"matmul_{activation_type}",
                "indices": [i, i + 1],
                "fused_subgraph": fused_sg,
            })

    return fusions


def aggregate_all_compute_graphs(
    compute_graph_dir: Path,
    output_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    """
    聚合所有计算图

    Args:
        compute_graph_dir: compute_graph/ 目录
        output_dir: 输出目录（通常是 run_dir）

    Returns:
        (聚合结果目录, 统计信息)
    """
    # 创建输出目录
    aggregated_dir = output_dir / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    # 读取清单
    manifest_path = compute_graph_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 统计信息
    stats = {
        "total_graphs": manifest["total_graphs"],
        "aggregated_count": 0,
        "total_fusions": 0,
        "fusion_patterns": {},
        "aggregated_graphs": [],
    }

    best_graph = None
    best_fusion_count = -1

    # 处理每个计算图
    for graph_info in manifest.get("graphs", []):
        graph_id = graph_info["id"]
        graph_path = compute_graph_dir / f"{graph_id}.json"

        if not graph_path.exists():
            continue

        with open(graph_path, 'r', encoding='utf-8') as f:
            compute_graph = json.load(f)

        # 聚合
        aggregated_subgraphs, agg_stats = aggregate_single_compute_graph(
            compute_graph)

        # 保存聚合结果
        aggregated_graph = {
            "id": f"{graph_id}_aggregated",
            "source_graph_id": graph_id,
            "is_original": graph_info.get("is_original", False),
            "subgraphs": aggregated_subgraphs,
            "aggregation_stats": agg_stats,
        }

        aggregated_path = aggregated_dir / f"{graph_id}_aggregated.json"
        with open(aggregated_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_graph, f, indent=2)

        # 更新统计
        stats["aggregated_count"] += 1
        stats["total_fusions"] += agg_stats["fusions_applied"]

        for fusion in agg_stats.get("fusion_details", []):
            pattern = fusion["pattern"]
            stats["fusion_patterns"][pattern] = stats["fusion_patterns"].get(
                pattern, 0) + 1

        stats["aggregated_graphs"].append({
            "graph_id": graph_id,
            "fusions": agg_stats["fusions_applied"],
            "patterns": [f["pattern"] for f in agg_stats.get("fusion_details", [])],
        })

        # 跟踪最佳图（融合最多的）
        if agg_stats["fusions_applied"] > best_fusion_count:
            best_fusion_count = agg_stats["fusions_applied"]
            best_graph = aggregated_graph

    # 保存最佳聚合图（用于后续 dispatch）
    if best_graph:
        best_path = aggregated_dir / "best_aggregated.json"
        # 转换为 subgraphs.json 格式（只包含 subgraphs 列表）
        with open(best_path, 'w', encoding='utf-8') as f:
            json.dump(best_graph["subgraphs"], f, indent=2)

        stats["best_graph_id"] = best_graph["source_graph_id"]
        stats["best_fusion_count"] = best_fusion_count

    # 保存统计信息
    stats_path = aggregated_dir / "aggregation_summary.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    return aggregated_dir, stats
