# LLM-First Optimization System for Fuser

## 概述

这是一个基于LLM的自主优化发现系统，专门针对复杂算子的Triton内核生成。系统采用**4代理架构**（Planner/Coder/Critic/Refiner），能够自动发现算法级优化、代数变换和融合策略。

## 核心特性

### 1. 多代理协作
- **Planner**: 生成4-7个不同的优化计划（baseline、算法变体、代数重排等）
- **Coder**: 根据选定计划生成完整的Triton实现
- **Critic**: 在运行前审计代码（检查合规性、常见陷阱）
- **Refiner**: 基于错误和性能反馈迭代改进

### 2. 自主优化发现
系统能够自动发现：
- **算法级优化**: FlashAttention风格的流式softmax、split-k GEMM等
- **代数变换**: RMSNorm/GEMM epilogue scaling、gamma折叠、scale移动等
- **融合策略**: 激进融合、内存优化、launch数量优化

### 3. 并行探索
- 多个worker并行工作
- 每个worker独立生成和测试多个计划
- 自动选择最佳结果（基于speedup）

## 架构设计

```
OptimizingOrchestrator
├── Worker 1
│   ├── Planner → 生成4-7个计划
│   ├── 选择Top 3计划
│   └── 对每个计划:
│       ├── Coder → 生成代码
│       ├── Critic → 审计代码
│       ├── 运行 + Benchmark
│       └── Refiner → 修复/优化 (最多3轮)
├── Worker 2
│   └── (同上)
├── Worker 3
│   └── (同上)
└── Worker 4
    └── (同上)

最终选择: 所有worker中speedup最高的结果
```

## 使用方法

### 基本用法

```bash
python -m Fuser.optimize_cli \
    --problem /path/to/problem.py \
    --model gpt-5 \
    --workers 4 \
    --plans-per-worker 3 \
    --refinements-per-plan 3 \
    --verify
```

### 参数说明

- `--problem`: 问题文件路径（KernelBench风格的PyTorch代码）
- `--model`: LLM模型名称（默认: gpt-5）
- `--workers`: 并行worker数量（默认: 4）
- `--plans-per-worker`: 每个worker尝试的计划数（默认: 3）
- `--refinements-per-plan`: 每个计划的最大优化轮数（默认: 3）
- `--llm-timeout-s`: LLM调用超时（默认: 1200秒）
- `--run-timeout-s`: 代码执行超时（默认: 1200秒）
- `--stream-mode`: 输出流模式（all|winner|none，默认: all）
- `--verify`: 最终验证结果
- `--out-root`: 输出根目录（默认: ./.fuse）

### 示例场景

#### 1. 优化注意力机制

```bash
python -m Fuser.optimize_cli \
    --problem examples/attention_problem.py \
    --model gpt-5 \
    --workers 4 \
    --plans-per-worker 4 \
    --verify
```

系统会自动尝试：
- Baseline: naive attention
- FlashAttention风格的流式softmax
- 不同的融合边界
- 不同的schedule配置

#### 2. 优化RMSNorm+GEMM

```bash
python -m Fuser.optimize_cli \
    --problem examples/rmsnorm_gemm.py \
    --model gpt-5 \
    --workers 4 \
    --plans-per-worker 3 \
    --refinements-per-plan 5
```

系统会自动尝试：
- Baseline: 分离的RMSNorm和GEMM
- Epilogue scaling: 将scaling移到GEMM输出
- Gamma折叠: 将gamma折叠到权重中
- 完全融合的单kernel实现

#### 3. 优化卷积链

```bash
python -m Fuser.optimize_cli \
    --problem examples/conv_chain.py \
    --model gpt-5 \
    --workers 4 \
    --plans-per-worker 3
```

## 输出结构

成功运行后，会在 `.fuse/<run_id>/` 下生成：

```
.fuse/run_YYYYMMDD_HHMMSS_<hash>/
├── orchestrator/
│   ├── orchestrator.log          # 编排器日志
│   ├── stream.log                # 流式输出
│   ├── summary.json              # 运行摘要
│   └── winner/
│       ├── worker_id.txt         # 获胜worker
│       ├── plan_id.txt           # 获胜计划
│       └── speedup.txt           # 加速比
├── workers/
│   ├── worker_01/
│   │   ├── planning/
│   │   │   ├── planner_prompt.txt
│   │   │   ├── planner_response.txt
│   │   │   └── planner_output.json  # 所有生成的计划
│   │   ├── implementations/
│   │   │   ├── plan_0_P1/
│   │   │   │   ├── plan.json
│   │   │   │   ├── iteration_0/
│   │   │   │   │   ├── coder_prompt.txt
│   │   │   │   │   ├── coder_output.py
│   │   │   │   │   ├── critic_output.json
│   │   │   │   │   └── code.py
│   │   │   │   ├── iteration_1/
│   │   │   │   │   ├── refiner_prompt.txt
│   │   │   │   │   └── code.py
│   │   │   │   └── ...
│   │   │   ├── plan_1_P2/
│   │   │   └── plan_2_P3/
│   │   ├── runs/                 # 执行结果
│   │   ├── artifacts/
│   │   │   ├── best_code.py      # 最佳代码
│   │   │   └── best_result.json  # 最佳结果
│   │   └── logs/
│   ├── worker_02/
│   ├── worker_03/
│   └── worker_04/
├── shared/
│   └── digests/                  # 去重哈希
└── result.tar.gz                 # 打包的最终结果
```

## 关键设计决策

### 1. 强制多样性探索

Planner必须生成：
- 至少1个baseline计划
- 至少1个算法变体
- 至少1个代数变换
- 对于attention: 必须有FlashAttention风格的计划
- 对于RMSNorm+GEMM: 必须有epilogue scaling计划

### 2. 最小系统约束

系统只强制3个硬约束：
1. **正确性**: 必须通过 `run_tests()` 并打印 `PASS`
2. **禁止捷径**: `kernel_function` 不能使用torch数学操作
3. **性能闭环**: 运行microbench并记录speedup

其他一切（算法选择、融合边界、schedule等）都由LLM决定。

### 3. 两层产物

- **Plan层**: 结构化的JSON，描述优化策略
- **Code层**: 可执行的Python文件，包含kernel + tests + microbench

### 4. 迭代优化

每个计划最多经过N轮refinement：
- 第1轮: Coder生成初始实现
- 第2-N轮: Refiner基于错误和性能反馈改进

### 5. 并行 + 选优

- 多个worker并行探索不同的计划空间
- 每个worker独立生成和测试多个计划
- 最终选择所有结果中speedup最高的

## 与原Fuser的区别

| 特性 | 原Fuser | LLM-First优化器 |
|------|---------|----------------|
| 优化发现 | 单一prompt，依赖LLM灵感 | 4代理系统，强制多样性探索 |
| 计划数量 | 1个 | 4-7个/worker，多worker并行 |
| 算法变体 | 隐式，不保证 | 显式要求，强制包含 |
| 代数变换 | 隐式，不保证 | 显式要求，强制包含 |
| 审计机制 | 无 | Critic代理预审 |
| 性能反馈 | 仅错误信息 | 错误 + 性能数据 |
| 选择策略 | 第一个成功 | 所有结果中最优 |

## 适用场景

### 最适合

1. **复杂算子**: Attention、LayerNorm+GEMM、RMSNorm+GEMM
2. **多操作融合**: 长操作链、多分支融合
3. **性能关键**: 需要探索多种优化策略
4. **算法不确定**: 不确定哪种算法最优

### 不太适合

1. **简单算子**: 单一pointwise操作（用原KernelAgent更快）
2. **时间受限**: 需要快速结果（LLM-First需要更多时间）
3. **资源受限**: 需要大量LLM调用和并行计算

## 性能调优建议

### 1. 调整并行度

```bash
# 更多探索，更长时间
--workers 8 --plans-per-worker 5

# 更快结果，较少探索
--workers 2 --plans-per-worker 2
```

### 2. 调整refinement轮数

```bash
# 更多优化机会
--refinements-per-plan 5

# 更快失败，快速尝试其他计划
--refinements-per-plan 2
```

### 3. 选择合适的模型

```bash
# 高质量，慢速
--model gpt-5

# 平衡
--model o4-mini

# 快速，可能质量较低
--model gpt-4-turbo
```

## 环境配置

需要设置以下环境变量：

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-5

# 或 Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# 或 DeepSeek
export DEEPSEEK_API_KEY=sk-...

# 或 自定义中继
export LLM_RELAY_URL=http://...
export LLM_RELAY_TIMEOUT_S=120
```

## 故障排查

### 1. 所有计划都失败

检查：
- 问题文件是否包含完整的PyTorch实现
- 是否有 `get_init_inputs()` 和 `get_inputs()` 函数
- 模型是否支持高推理能力（reasoning effort）

### 2. Planner生成的计划不合理

尝试：
- 使用更强的模型（gpt-5）
- 增加 `--llm-timeout-s`
- 检查问题文件是否清晰描述了算子

### 3. Coder生成的代码总是失败

尝试：
- 增加 `--refinements-per-plan`
- 检查Critic输出，看是否有明显问题
- 手动检查生成的代码

### 4. 性能不如预期

尝试：
- 增加 `--plans-per-worker` 探索更多策略
- 增加 `--workers` 并行探索
- 检查是否有更好的计划被生成但未被选中

## 扩展和定制

### 1. 添加新的优化模式

编辑 `llm_optimizer.py` 中的 `_build_planner_prompt`，添加新的模式检测和计划要求。

### 2. 自定义Critic规则

编辑 `llm_optimizer.py` 中的 `_build_critic_prompt`，添加新的审计规则。

### 3. 调整选择策略

编辑 `optimizing_orchestrator.py` 中的结果选择逻辑，可以基于：
- Speedup
- Launch数量
- 内存使用
- 数值稳定性

## 贡献

欢迎贡献新的优化模式、审计规则和选择策略！

## 许可证

Apache License 2.0
