# UB/LB/DB 阈值系统使用指南

## 概述

阈值系统通过 UB/LB/DB 三个阈值控制 Worker 路径的生命周期，实现智能早停和经验积累。

## 三个阈值说明

### UB (Upper Bound) - 上限阈值
- **默认值**: 0.8
- **含义**: 当 Worker 的全局最佳 speedup 达到或超过此值时，认为路径已经足够优秀
- **触发行为**: 
  - Worker 提前停止
  - 将该路径的配置和代码写入经验池
  - 标记为 `REACHED_UB`

### LB (Lower Bound) - 下限阈值
- **默认值**: 0.3
- **含义**: 当 Worker 的全局最佳 speedup 持续低于此值时，认为路径失败
- **触发行为**:
  - Worker 提前停止
  - 不写入经验池
  - 标记为 `FAILED_LB`

### DB (Degradation Bound) - 回退阈值
- **默认值**: 0.4
- **含义**: 当某个 chunk 的性能相比历史最优回退超过此值时，认为严重退化
- **触发行为**:
  - Worker 提前停止
  - 记录退化信息到日志
  - 标记为 `DEGRADED`

## 命令行使用

### 基本用法（使用默认阈值）

```bash
python generate_kernel.py --level 1 --problem_id 1 --method direct
```

默认阈值：
- UB = 0.8
- LB = 0.3
- DB = 0.4

### 自定义阈值

```bash
# 示例 1：更严格的标准
python generate_kernel.py \
    --level 1 \
    --problem_id 1 \
    --method direct \
    --UB 1.0 \
    --LB 0.5 \
    --DB 0.3

# 示例 2：更宽松的标准
python generate_kernel.py \
    --level 2 \
    --problem_id 5 \
    --method direct \
    --UB 0.7 \
    --LB 0.2 \
    --DB 0.5

# 示例 3：配合其他参数
python generate_kernel.py \
    --level 3 \
    --problem_id 10 \
    --method direct \
    --workers 4 \
    --max_rounds 20 \
    --update_interval 5 \
    --UB 0.9 \
    --LB 0.4 \
    --DB 0.3 \
    --patience_chunks 2
```

## 代码中使用

```python
from triton_kernel_agent import TritonKernelAgent

# 使用默认阈值
agent = TritonKernelAgent(
    num_workers=4,
    max_rounds=10
)

# 自定义阈值
agent = TritonKernelAgent(
    num_workers=4,
    max_rounds=20,
    update_interval=5,
    UB=0.9,
    LB=0.4,
    DB=0.3,
    patience_chunks=2
)

result = agent.generate_kernel(
    problem_description="...",
    test_code="..."
)
```

## 阈值设置建议

### 根据优化目标选择

#### 追求高性能（严格标准）
```bash
--UB 1.2  # 要求至少 20% 加速
--LB 0.8  # 低于 baseline 20% 就放弃
--DB 0.2  # 回退 20% 就停止
```

#### 平衡探索（推荐）
```bash
--UB 0.8  # 达到 baseline 80% 即可
--LB 0.3  # 给予更多探索空间
--DB 0.4  # 容忍较大回退
```

#### 广泛探索（宽松标准）
```bash
--UB 0.6  # 较低的成功标准
--LB 0.1  # 几乎不放弃
--DB 0.6  # 容忍大幅回退
```

### 根据算子类型选择

#### Matmul（容易优化）
```bash
--UB 1.0  # 期望达到或超过 baseline
--LB 0.5
--DB 0.3
```

#### Reduce（中等难度）
```bash
--UB 0.8
--LB 0.4
--DB 0.4
```

#### Conv2D（较难优化）
```bash
--UB 0.7
--LB 0.3
--DB 0.5
```

## 工作流程示例

### 场景：10轮迭代，update_interval=5

```
Round 1-5 (Chunk 1):
  - 测试各种优化
  - 最佳 speedup = 0.5
  - 判断: 0.5 >= LB(0.3) ✓ 继续
  
Round 6-10 (Chunk 2):
  - 继续优化
  - 最佳 speedup = 0.85
  - 判断: 0.85 >= UB(0.8) ✓ 达到目标！
  - 行为: 提前停止，写入经验池
```

### 场景：性能回退

```
Round 1-5 (Chunk 1):
  - 最佳 speedup = 0.7
  
Round 6-10 (Chunk 2):
  - 当前 speedup = 0.25
  - 回退 delta = 0.7 - 0.25 = 0.45
  - 判断: 0.45 > DB(0.4) ✓ 严重退化！
  - 行为: 提前停止，标记为 DEGRADED
```

### 场景：持续失败

```
Round 1-5 (Chunk 1):
  - 最佳 speedup = 0.2
  - 判断: 0.2 < LB(0.3) ✓ 低于下限！
  - 行为: 提前停止，标记为 FAILED_LB
```

## 日志输出

### 正常运行
```
Chunk 1 (round 5): success=True, speedup=0.50x, sigma=0.50x, delta=0.00
Chunk 2 (round 10): success=True, speedup=0.85x, sigma=0.85x, delta=0.00
Worker 0 reached UB: REACHED_UB: sigma=0.85 >= UB=0.80
```

### 失败路径
```
Chunk 1 (round 5): success=False, speedup=0.00x, sigma=0.00x, delta=0.00
Chunk 2 (round 10): success=True, speedup=0.25x, sigma=0.25x, delta=0.00
Worker 1 stopped: FAILED_LB: sigma=0.25 < LB=0.30
```

### 性能退化
```
Chunk 1 (round 5): success=True, speedup=0.70x, sigma=0.70x, delta=0.00
Chunk 2 (round 10): success=True, speedup=0.20x, sigma=0.70x, delta=0.50
Worker 2 stopped: DEGRADED: delta=0.50 > DB=0.40
```

## 结果字段说明

Worker 返回的结果中包含以下新字段：

```python
{
    "worker_id": 0,
    "success": True,
    "speedup": 0.85,
    "best_speedup_global": 0.85,  # 全局最佳 speedup (σ_s)
    "early_stop_reason": "REACHED_UB: sigma=0.85 >= UB=0.80",  # 早停原因
    "thresholds": {  # 使用的阈值
        "UB": 0.8,
        "LB": 0.3,
        "DB": 0.4
    },
    "intermediate_results": [  # 每个 chunk 的详细信息
        {
            "round": 5,
            "chunk": 1,
            "success": True,
            "speedup": 0.50,
            "sigma_t": 0.50,
            "delta_t": 0.00
        },
        {
            "round": 10,
            "chunk": 2,
            "success": True,
            "speedup": 0.85,
            "sigma_t": 0.85,
            "delta_t": 0.00
        }
    ]
}
```

## 常见问题

### Q1: 为什么我的 Worker 总是提前停止？

**A**: 检查阈值设置是否过于严格。如果 UB 设置过高或 LB 设置过高，Worker 很容易触发早停。

### Q2: 如何禁用阈值系统？

**A**: 设置极端值：
```bash
--UB 999.0 --LB 0.0 --DB 999.0
```

### Q3: patience_chunks 是什么？

**A**: 连续多少个 chunk 没有性能提升就早停。默认 3，即 3 个 chunk (15轮) 无改进就停止。

### Q4: 阈值判断在第一个 chunk 生效吗？

**A**: LB 和 DB 判断会跳过第一个 chunk，给 Worker 一些预热时间。

## 最佳实践

1. **初次运行**：使用默认阈值，观察日志
2. **调整阈值**：根据实际性能分布调整
3. **记录经验**：达到 UB 的配置会自动保存到经验池
4. **分析日志**：查看 `early_stop_reason` 了解停止原因
5. **迭代优化**：根据经验池中的最佳配置调整阈值

## 总结

- **UB**: 成功标准，达到即可停止
- **LB**: 失败标准，低于即放弃
- **DB**: 退化标准，回退过多即停止
- **默认值**: UB=0.8, LB=0.3, DB=0.4（适合大多数场景）
- **命令行**: `--UB`, `--LB`, `--DB`, `--patience_chunks`
