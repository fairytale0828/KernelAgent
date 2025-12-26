# Fuser 算法级优化 (Mirage-Inspired)

## 概述

本模块为 Fuser 添加了 Mirage 风格的算法级优化能力，能够在子图级别识别和应用高级融合模式。

## 核心组件

### 1. AlgorithmPatternDetector (`algorithm_analyzer.py`)

检测以下算法优化模式：

- **norm_gemm_fusion**: RMSNorm/LayerNorm + GEMM 融合
- **streaming_attention**: FlashAttention 风格的流式注意力
- **scale_absorption**: Scale/Bias 吸收到 GEMM
- **broadcast_sharing**: GQA 风格的 broadcast 转共享
- **epilogue_fusion**: Pointwise 链融入 GEMM epilogue

### 2. AlgorithmRewriter (`algorithm_rewriter.py`)

将检测到的模式重写为融合子图，生成包含算法级元数据的新子图描述。

### 3. Enhanced Pipeline (`pipeline_enhanced.py`)

集成算法优化的完整管道：
```
Extract → Analyze → Rewrite → Dispatch → Compose
```

## 使用方法

### 基本用法

```bash
python -m Fuser.pipeline_enhanced \
  --problem /path/to/problem.py \
  --extract-model deepseek-chat \
  --dispatch-model deepseek-chat \
  --compose-model deepseek-chat \
  --workers 4 \
  --max-iters 5 \
  --verify \
  --enable-algorithmic-rewrite
```

### 使用 DeepSeek 模型

```bash
# 设置 API Key
export DEEPSEEK_API_KEY=sk-xxx

# 运行增强管道
python -m Fuser.pipeline_enhanced \
  --problem examples/rmsnorm_matmul.py \
  --extract-model deepseek-chat \
  --dispatch-model deepseek-chat \
  --compose-model deepseek-chat \
  --verify \
  --enable-algorithmic-rewrite
```

### 禁用算法优化

```bash
python -m Fuser.pipeline_enhanced \
  --problem /path/to/problem.py \
  --no-algorithmic-rewrite
```

### 调整收益阈值

```bash
python -m Fuser.pipeline_enhanced \
  --problem /path/to/problem.py \
  --rewrite-benefit-threshold 0.8  # 只应用高收益优化 (>0.8)
```

## RMSNorm + Matmul 融合示例

### 原始 PyTorch 代码

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(512)
        self.linear = nn.Linear(512, 1024)
    
    def forward(self, x):
        x = self.norm(x)  # RMSNorm
        x = self.linear(x)  # Matmul
        return x
```

### Fuser 流程

1. **Orchestrator** 生成融合代码
2. **SubgraphExtractor** 识别 2 个子图：
   - `sg_rmsnorm_001`: RMSNorm
   - `sg_linear_002`: Linear/GEMM

3. **AlgorithmPatternDetector** 检测到 `norm_gemm_fusion` 模式：
   ```json
   {
     "pattern_type": "norm_gemm_fusion",
     "subgraph_ids": ["sg_rmsnorm_001", "sg_linear_002"],
     "estimated_benefit": 0.85,
     "fusion_opportunity": {
       "norm_type": "rms_norm",
       "fusion_point": "reduction_into_gemm_loop"
     }
   }
   ```

4. **AlgorithmRewriter** 生成融合子图：
   ```json
   {
     "id": "fused_norm_gemm_sg_rmsnorm_001_sg_linear_002",
     "type": "algorithmic_fusion",
     "algorithm_type": "norm_gemm_fused",
     "ops": [{
       "op": "fused_norm_gemm",
       "algorithm_description": "Fuse RMS reduction into GEMM loop..."
     }],
     "generation_hints": {
       "triton_strategy": "single_kernel_with_running_reduction",
       "key_optimizations": [
         "Maintain running sum for normalization in registers",
         "Fuse normalization scale into GEMM epilogue",
         "Avoid materializing intermediate normalized tensor"
       ]
     }
   }
   ```

5. **Dispatcher** 生成 Mirage 风格的问题描述：
   ```
   Implement a Triton kernel that fuses RMSNORM with GEMM using algorithmic fusion.
   
   **Algorithm Strategy (Mirage-style):**
   This is NOT a simple epilogue fusion. Instead, we fuse the normalization's 
   reduction INTO the GEMM's main accumulation loop.
   
   **Pseudo-algorithm:**
   for each output tile (i, j):
       acc = 0.0  # GEMM accumulator
       sum_sq = 0.0  # RMSNorm accumulator
       
       for k in range(K):  # Shared loop!
           x_val = load X[i, k]
           w_val = load W[k, j]
           
           acc += x_val * w_val  # GEMM
           sum_sq += x_val * x_val  # RMS reduction
       
       # Epilogue: apply normalization
       rms = sqrt(sum_sq / K + eps)
       output[i, j] = (acc * gamma) / rms
   ```

6. **KernelAgent** 生成融合的 Triton 内核

7. **Composer** 组合最终内核并验证

### 性能收益

- **内存流量**: ~1.5x 减少（避免中间张量物化）
- **Kernel Launch 开销**: 减少 1 次（2 个 kernel → 1 个 kernel）
- **寄存器利用**: 更高效（共享累加器）

## 支持的模型

### DeepSeek 模型

- `deepseek-chat`: DeepSeek-V3.2-Exp (非思考模式)
- `deepseek-reasoner`: DeepSeek-V3.2-Exp (思考模式)
- `deepseek-coder`: 代码生成专用

### OpenAI 模型

- `gpt-5`: GPT-5 旗舰模型
- `o4-mini`: 快速推理模型

### Anthropic 模型

- `claude-sonnet-4-20250514`: Claude 4 Sonnet
- `claude-opus-4-1-20250805`: Claude 4.1 Opus

## 配置环境变量

```bash
# DeepSeek
export DEEPSEEK_API_KEY=sk-xxx

# OpenAI
export OPENAI_API_KEY=sk-xxx

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-xxx
```

## 输出产物

```
.fuse/<run_id>/
├── orchestrator/
│   ├── code.py.tgz                    # 融合后的 PyTorch 代码
│   └── summary.json
├── subgraphs.json                     # 原始子图
├── subgraphs_rewritten.json           # 重写后的子图 (NEW!)
├── algorithm_rewrite_metadata.json    # 重写元数据 (NEW!)
├── kernels_out/
│   ├── fused_norm_gemm_*/             # 融合子图的内核
│   │   ├── problem.txt                # Mirage 风格的问题描述
│   │   └── kernel.py
│   └── summary.json
└── compose_out/
    ├── composed_kernel.py             # 最终组合内核
    └── composition_summary.json
```

## 算法优化模式详解

### 1. Norm + GEMM 融合

**检测条件**:
- Norm 操作（RMSNorm/LayerNorm）后紧跟 GEMM
- 数据依赖：Norm 输出 = GEMM 输入

**优化策略**:
- 将归约计算融入 GEMM 主循环
- 在 epilogue 应用归一化

**收益**: 0.85

### 2. Streaming Attention

**检测条件**:
- QK^T → Softmax → @V 序列
- K/V 维度可分块

**优化策略**:
- Block-wise K/V 迭代
- 维护 running max/sum
- Online softmax

**收益**: 0.95

### 3. Epilogue Fusion

**检测条件**:
- GEMM 后跟 2+ 个 pointwise 操作

**优化策略**:
- 所有 pointwise 操作在寄存器中完成
- 避免中间张量存储

**收益**: 0.75

## 调试和日志

增强管道会输出详细的进度信息：

```
🚀 Starting enhanced Fuser pipeline with algorithmic optimization
   Problem: rmsnorm_matmul.py
   Models: extract=deepseek-chat, dispatch=deepseek-chat, compose=deepseek-chat
   Algorithmic rewrite: enabled

📊 Step 1: Extracting subgraphs...
   ✓ Subgraphs extracted to: .fuse/run_xxx/subgraphs.json

🔬 Step 1.5: Analyzing for algorithmic optimization opportunities...
   Original subgraphs: 2
   🔍 Detected 1 high-value optimization opportunities:
      - norm_gemm_fusion: ['sg_rmsnorm_001', 'sg_linear_002']
        Benefit: 0.85, Strategy: fuse_norm_reduction_into_gemm_accumulation
   🔧 Applying algorithmic rewrites...
   ✓ Applied 1 algorithmic rewrites
   ✓ Eliminated 2 intermediate subgraphs
   ✓ Created 1 fused subgraphs
   ✓ Final subgraph count: 1

⚙️  Step 2: Dispatching subgraphs to KernelAgent...
   Parallel jobs: 1
   ✓ Kernels generated: .fuse/run_xxx/kernels_out/summary.json

🔗 Step 3: Composing end-to-end kernel...
   ✓ Composition complete
   ✅ Verification PASSED

============================================================
📋 Pipeline Summary:
   Run directory: .fuse/run_xxx
   Subgraphs: .fuse/run_xxx/subgraphs_rewritten.json
   Kernels: .fuse/run_xxx/kernels_out/summary.json
   Composed kernel: .fuse/run_xxx/compose_out/composed_kernel.py
   Algorithmic optimizations applied: 1
============================================================
```

## 扩展新的优化模式

### 1. 在 `algorithm_analyzer.py` 中添加检测器

```python
def _detect_my_pattern(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
    patterns = []
    # 检测逻辑
    for sg in subgraphs:
        if self._matches_my_pattern(sg):
            patterns.append(AlgorithmPattern(
                pattern_type="my_pattern",
                subgraph_ids=[sg["id"]],
                fusion_opportunity={...},
                estimated_benefit=0.7,
                rewrite_strategy="my_rewrite_strategy"
            ))
    return patterns
```

### 2. 在 `algorithm_rewriter.py` 中添加重写器

```python
def _rewrite_my_pattern(self, subgraphs: List[Dict], pattern: AlgorithmPattern) -> Dict:
    # 生成融合子图描述
    fused_sg = {
        "id": f"fused_my_pattern_{...}",
        "type": "algorithmic_fusion",
        "algorithm_type": "my_pattern_fused",
        "generation_hints": {
            "triton_strategy": "...",
            "key_optimizations": [...]
        }
    }
    return fused_sg
```

### 3. 在 `dispatch_kernel_agent_enhanced.py` 中添加提示生成器

```python
def _generate_my_pattern_prompt(item: Dict) -> str:
    return textwrap.dedent(f"""
    Implement a Triton kernel for my pattern fusion.
    
    **Algorithm Strategy:**
    ...
    
    **Pseudo-algorithm:**
    ```python
    ...
    ```
    """)
```

## 参考文献

- **Mirage**: A Multi-Level Superoptimizer for Tensor Programs
- **FlashAttention**: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **FlashAttention-2**: Faster Attention with Better Parallelism and Work Partitioning

## 贡献

欢迎贡献新的算法优化模式！请参考上述扩展指南。
