# 计算图架构 - 方案1实现

## 概述

本文档描述了新的计算图架构，解决了 E-Graph 变体混淆导致的聚合失败问题。

## 问题背景

### 原有问题

当 E-Graph 优化生成变体时，会产生如下结构：

```json
[
  {"id": "linear_1", ...},      // 原始
  {"id": "linear_1_v1", ...},   // 变体
  {"id": "matmul_1", ...},      // 原始
  {"id": "softmax_1", ...},     // 变体
  {"id": "softmax_1_v1", ...},  // 原始
  {"id": "matmul_2", ...}       // 原始
]
```

**问题**：
1. 变体和原始版本混在一起
2. 破坏了原有的顺序结构
3. 聚合器无法识别 Flash Attention 模式（期望连续的 `matmul → softmax → matmul`）
4. 中间夹杂的变体导致模式匹配失败

## 解决方案：完整计算图变体

### 核心思路

不生成混合列表，而是生成**多个完整的计算图**，每个图是一个独立的变体组合。

### 架构流程

```
subgraphs.json (原始)
    ↓
E-Graph 优化
    ↓
subgraphs_transformed.json (包含所有变体)
    ↓
计算图构建器 (ComputeGraphBuilder)
    ↓
compute_graph/
├── compute_graph_0.json  (原始: [linear_1, matmul_1, softmax_1, matmul_2])
├── compute_graph_1.json  (变体1: [linear_1_v1, matmul_1, softmax_1, matmul_2])
├── compute_graph_2.json  (变体2: [linear_1, matmul_1, softmax_1_v1, matmul_2])
├── compute_graph_3.json  (变体3: [linear_1_v1, matmul_1, softmax_1_v1, matmul_2])
└── manifest.json
    ↓
计算图聚合器 (ComputeGraphAggregator)
    ↓
aggregated/
├── compute_graph_0_aggregated.json
├── compute_graph_1_aggregated.json
├── compute_graph_2_aggregated.json
├── compute_graph_3_aggregated.json
├── best_aggregated.json  (选择最佳的用于后续 dispatch)
└── aggregation_summary.json
```

## 关键组件

### 1. ComputeGraphBuilder

**文件**: `compute_graph_builder.py`

**功能**:
- 从 `subgraphs_transformed.json` 识别变体关系
- 生成多个完整的计算图变体组合
- 每个计算图是一个完整的、可独立执行的路径

**核心方法**:
```python
def build_from_transformed_subgraphs(
    transformed_subgraphs: List[Dict[str, Any]]
) -> Tuple[List[ComputeGraphVariant], Dict[str, Any]]:
    # 1. 识别变体组
    variant_groups = self._identify_variant_groups(...)
    
    # 2. 确定拓扑顺序
    ordered_groups = self._order_variant_groups(...)
    
    # 3. 生成变体组合
    compute_graphs = self._generate_combinations(...)
    
    # 4. 去重和排序
    compute_graphs = self._deduplicate_and_rank(...)
    
    return compute_graphs, stats
```

**变体组合策略**:

1. **完全组合**（当组合数较少时）:
   - 生成所有可能的笛卡尔积
   - 例如: 2个位置各有2个变体 → 4个组合

2. **采样组合**（当组合数太多时）:
   - 策略1: 原始组合（全部选基础版本）
   - 策略2: 最低代价组合
   - 策略3: 最大融合组合
   - 策略4-N: 每次只改变一个位置

### 2. ComputeGraphAggregator

**文件**: `compute_graph_aggregator.py`

**功能**:
- 对每个完整计算图独立进行聚合
- 识别融合模式（Flash Attention、MLP 等）
- 选择最佳聚合结果

**支持的融合模式**:

1. **Flash Attention**:
   ```
   matmul → softmax/online_softmax → matmul
   ```

2. **MLP**:
   ```
   linear → activation (gelu/relu/silu) → linear
   ```

**核心方法**:
```python
def aggregate_single_compute_graph(
    compute_graph: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # 识别融合模式
    flash_attention_fusions = _find_flash_attention_pattern(...)
    mlp_fusions = _find_mlp_pattern(...)
    
    # 创建融合子图
    aggregated_subgraphs = [...]
    
    return aggregated_subgraphs, stats
```

### 3. Pipeline 集成

**文件**: `pipeline.py`

**新增步骤**:

```python
# Step 1.5: E-Graph 优化
subgraphs.json → subgraphs_transformed.json

# Step 1.5.1: 构建完整计算图
subgraphs_transformed.json → compute_graph/

# Step 1.6: 聚合计算图
compute_graph/ → aggregated/best_aggregated.json

# Step 2: Dispatch (使用 best_aggregated.json)
```

## 数据结构

### ComputeGraphVariant

```python
@dataclass
class ComputeGraphVariant:
    id: str                              # compute_graph_0
    name: str                            # original / variant_1
    subgraphs: List[Dict[str, Any]]      # 完整的子图列表
    variant_selections: Dict[str, str]   # base_id -> selected_variant_id
    total_cost: float                    # 总代价
    is_original: bool                    # 是否是原始版本
```

### Manifest

```json
{
  "total_graphs": 4,
  "original_graph_id": "compute_graph_0",
  "variant_graph_ids": ["compute_graph_1", "compute_graph_2", "compute_graph_3"],
  "graphs": [
    {
      "id": "compute_graph_0",
      "name": "original",
      "is_original": true,
      "total_cost": 37.1,
      "variant_selections": {
        "linear_1": "linear_1",
        "matmul_1": "matmul_1",
        "softmax_1": "softmax_1",
        "matmul_2": "matmul_2"
      }
    },
    ...
  ]
}
```

## 使用方法

### 命令行

```bash
python -m Fuser.pipeline \
    --problem /path/to/problem.py \
    --enable-algebraic-rewrite \
    --rewrite-top-k 5 \
    --rewrite-rule-set attention \
    --extract-model gpt-5 \
    --dispatch-model o4-mini \
    --compose-model o4-mini
```

### 程序化调用

```python
from Fuser.compute_graph_builder import build_compute_graphs_from_file

# 构建计算图
compute_graph_dir, stats = build_compute_graphs_from_file(
    transformed_subgraphs_path=Path("subgraphs_transformed.json"),
    output_dir=Path("run_dir"),
    max_variants=10,
)

# 聚合
from Fuser.compute_graph_aggregator import aggregate_all_compute_graphs

aggregated_dir, agg_stats = aggregate_all_compute_graphs(
    compute_graph_dir=compute_graph_dir,
    output_dir=Path("run_dir"),
)
```

## 优势

### 1. 解决变体混淆问题

✅ 每个计算图是完整的、独立的
✅ 不会出现变体混在一起的情况
✅ 聚合器可以正确识别模式

### 2. 支持多种变体探索

✅ 可以生成多个变体组合
✅ 每个组合都可以独立评估
✅ 选择最佳的用于后续处理

### 3. 保持代码简洁

✅ 聚合器逻辑简单，不需要理解变体关系
✅ 基于列表顺序的模式匹配仍然有效
✅ 易于扩展新的融合模式

### 4. 可扩展性

✅ 支持任意数量的变体组
✅ 支持复杂的变体组合策略
✅ 可以并行处理多个计算图

## 示例输出

### compute_graph_0.json (原始)

```json
{
  "id": "compute_graph_0",
  "name": "original",
  "is_original": true,
  "total_cost": 37.1,
  "subgraphs": [
    {"id": "linear_1", "type": "linear", ...},
    {"id": "matmul_1", "type": "matmul", ...},
    {"id": "softmax_1", "type": "softmax", ...},
    {"id": "matmul_2", "type": "matmul", ...}
  ],
  "variant_selections": {
    "linear_1": "linear_1",
    "matmul_1": "matmul_1",
    "softmax_1": "softmax_1",
    "matmul_2": "matmul_2"
  }
}
```

### compute_graph_1_aggregated.json

```json
{
  "id": "compute_graph_1_aggregated",
  "source_graph_id": "compute_graph_1",
  "subgraphs": [
    {"id": "linear_1_v1", "type": "linear", ...},
    {
      "id": "fused_flash_attention_1",
      "type": "flash_attention",
      "fused_from": ["matmul_1", "softmax_1", "matmul_2"],
      "codegen_hints": {
        "use_flash_attention": true,
        "online_softmax": true
      }
    }
  ],
  "aggregation_stats": {
    "input_count": 4,
    "output_count": 2,
    "fusions_applied": 1,
    "fusion_details": [
      {
        "pattern": "flash_attention",
        "indices": [1, 2, 3]
      }
    ]
  }
}
```

## 测试

运行测试脚本：

```bash
cd KernelAgent/Fuser
python test_compute_graph_builder.py
```

预期输出：
```
Test Results
============================================================

Statistics:
  Total input subgraphs: 6
  Variant groups: 4
  Generated combinations: 4

Variant group details:
  linear_1: 2 variants
  matmul_1: 1 variants
  softmax_1: 2 variants
  matmul_2: 1 variants

Generated Compute Graphs:
  Graph 0: compute_graph_0 (original)
    Is original: True
    Total cost: 37.10
    Subgraphs: ['linear_1', 'matmul_1', 'softmax_1', 'matmul_2']

  Graph 1: compute_graph_1 (variant_1)
    Is original: False
    Total cost: 34.60
    Subgraphs: ['linear_1_v1', 'matmul_1', 'softmax_1', 'matmul_2']

  ...

Validation
============================================================
  Has original graph: ✓
  compute_graph_0 is complete: ✓
  compute_graph_1 is complete: ✓
  compute_graph_2 is complete: ✓
  compute_graph_3 is complete: ✓
  No duplicate signatures: ✓

  Flash Attention Pattern Check:
    compute_graph_0: ✓ Pattern found
    compute_graph_1: ✓ Pattern found
    compute_graph_2: ✓ Pattern found
    compute_graph_3: ✓ Pattern found

Test completed successfully!
```

## 未来改进

1. **更智能的变体选择策略**
   - 基于性能预测的选择
   - 基于硬件特性的选择

2. **更多融合模式**
   - LayerNorm + Linear
   - RMSNorm + MatMul
   - Conv + BatchNorm + ReLU

3. **并行聚合**
   - 多个计算图并行处理
   - 加速整体流程

4. **代价模型改进**
   - 学习型代价模型
   - 实际性能反馈

## 总结

新的计算图架构通过生成多个完整的计算图变体，彻底解决了变体混淆问题。每个计算图都是独立的、完整的，可以正确地进行模式识别和聚合。这为后续的内核生成和组合提供了坚实的基础。
