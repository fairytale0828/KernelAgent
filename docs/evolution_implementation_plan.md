# 进化式优化系统实现计划

## 已完成

✅ **第1步：经验池管理器**
- 创建 `experience_pool.py`
- 实现 `ExperienceEntry` 和 `ExperiencePool` 类
- 支持经验的添加、查询、持久化

✅ **第2步：阈值配置系统**
- 在 `agent.py` 中添加 UB/LB/DB 参数
- 在 `manager.py` 中传递阈值参数
- 在 `worker.py` 中实现阈值判断逻辑

✅ **第3步：Worker 生命周期改造**
- 修改 `worker.py` 的 `run()` 方法
- 添加 chunk 级别的阈值检查
- 实现早停逻辑（LB、DB、patience）
- 添加 `best_speedup_global` 和 `early_stop_reason` 字段

✅ **第4步：命令行参数**
- 在 `generate_kernel.py` 中添加：
  - `--UB` (默认 0.8)
  - `--LB` (默认 0.3)
  - `--DB` (默认 0.4)
  - `--patience_chunks` (默认 3)

✅ **第5步：文档**
- 创建 `threshold_system_usage.md` 使用指南

## 待实现

### 第6步：经验池集成
- [ ] 在 `agent.py` 中集成 `ExperiencePool`
- [ ] 实现 UB 达成时的经验写入逻辑
- [ ] 添加经验池查询和展示功能

### 第4步：Manager 回调系统
- [ ] 在 `manager.py` 中添加回调方法：
  - `on_path_failed()`
  - `on_path_degraded()`
  - `on_path_reached_UB()`
  - `on_path_no_progress()`

### 第5步：融合 Worker 生成
- [ ] 实现融合策略生成逻辑
- [ ] 创建融合 prompt 模板
- [ ] 注册虚拟策略到 `StrategyManager`

### 第6步：命令行参数
- [ ] 在 `generate_kernel.py` 中添加：
  - `--UB`
  - `--LB`
  - `--DB`
  - `--enable_fusion`

### 第7步：UI 界面（可选）
- [ ] 创建配置修正界面
- [ ] 支持人工干预和重启

## 当前状态

正在实现第2步：阈值配置系统

## 注意事项

1. 所有数据都基于实际运行生成，不使用模拟数据
2. 保持向后兼容，默认不启用进化功能
3. 逐步测试每个功能模块
