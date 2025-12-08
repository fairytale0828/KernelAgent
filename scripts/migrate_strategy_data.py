#!/usr/bin/env python3
"""
1. 从triton_kernel_logs/strategy_scores.json读取旧数据
2. 转换为新的格式（包含stats和priors）
3. 保存到data/policy_state.json
"""

import json
import sys
from pathlib import Path


def migrate_old_scores_to_new_format(old_file: Path, new_file: Path):
    """
    将旧的得分矩阵格式迁移到新的策略状态格式
    
    旧格式: {op_type: {strategy_id: score}}
    新格式: {
        "stats": {op_type: {strategy_id: {"n": count, "mu_data": avg}}},
        "priors": {op_type: {strategy_id: mu_prior}},
        "global_iteration": T
    }
    """
    print(f"Reading old scores from: {old_file}")
    
    if not old_file.exists():
        print(f"Old file not found: {old_file}")
        print("No migration needed - will start fresh")
        return False
    
    try:
        with open(old_file, "r", encoding="utf-8") as f:
            old_scores = json.load(f)
    except Exception as e:
        print(f"Error reading old file: {e}")
        return False
    
    # 转换为新格式
    new_data = {
        "stats": {},
        "priors": {},
        "global_iteration": 0,
        "metadata": {
            "prior_weight": 5.0,
            "exploration_coef": 2.0,
            "migrated_from": str(old_file)
        }
    }
    
    # 默认先验值
    default_priors = {
        "matmul": {
            "Triton.TensorCore.v1": 0.0,
            "Triton.TileOnly.v1": 0.0,
            "Triton.SharedMemOpt.v1": 0.0,
            "Triton.AsyncCopy.v1": 0.0,
            "Triton.TileVectorize.v1": 0.0,
            "Triton.ReduceOpt.v1": 0.0,
            "Triton.WarpOpt.v1": 0.0,
            "Triton.CoalescedAccess.v1": 0.0,
        },
        "reduce": {
            "Triton.ReduceOpt.v1": 0.0,
            "Triton.WarpOpt.v1": 0.0,
            "Triton.SharedMemOpt.v1": 0.0,
            "Triton.TileOnly.v1": 0.0,
            "Triton.TileVectorize.v1": 0.0,
            "Triton.CoalescedAccess.v1": 0.0,
            "Triton.TensorCore.v1": 0.0,
            "Triton.AsyncCopy.v1": 0.0,
        },
        "elementwise": {
            "Triton.CoalescedAccess.v1": 0.0,
            "Triton.TileVectorize.v1": 0.0,
            "Triton.TileOnly.v1": 0.0,
            "Triton.ReduceOpt.v1": 0.0,
            "Triton.SharedMemOpt.v1": 0.0,
            "Triton.WarpOpt.v1": 0.0,
            "Triton.TensorCore.v1": 0.0,
            "Triton.AsyncCopy.v1": 0.0,
        },
    }
    
    # 转换每个算子类型的数据
    for op_type, strategies in old_scores.items():
        new_data["stats"][op_type] = {}
        new_data["priors"][op_type] = {}
        
        for strategy_id, old_score in strategies.items():
            # 将旧的score转换为统计数据
            # 假设：如果score > 0，说明有一些成功的尝试
            if old_score > 0:
                # 估算尝试次数（基于score的大小）
                estimated_n = max(1, int(old_score * 10))
                # 估算平均speedup（假设score接近平均speedup）
                estimated_mu = max(0.0, old_score)
            else:
                estimated_n = 0
                estimated_mu = 0.0
            
            new_data["stats"][op_type][strategy_id] = {
                "n": estimated_n,
                "mu_data": estimated_mu
            }
            
            # 设置先验
            prior = default_priors.get(op_type, {}).get(strategy_id, 1.0)
            new_data["priors"][op_type][strategy_id] = prior
    
    # 保存新格式
    new_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing new format to: {new_file}")
    with open(new_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Migration completed successfully")
    print(f"  Migrated {len(old_scores)} operator types")
    
    # 备份旧文件
    backup_file = old_file.with_suffix(".json.backup")
    print(f"Creating backup: {backup_file}")
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(old_scores, f, indent=2, ensure_ascii=False)
    
    return True


def main():
    """主函数"""
    # 确定项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 旧文件路径
    old_file = project_root / "triton_kernel_logs" / "strategy_scores.json"
    
    # 新文件路径
    new_file = project_root / "data" / "policy_state.json"
    
    print("=" * 60)
    print("Strategy Data Migration Script")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Old file: {old_file}")
    print(f"New file: {new_file}")
    print()
    
    # 检查新文件是否已存在
    if new_file.exists():
        print(f"Warning: New file already exists: {new_file}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled")
            return 0
    
    # 执行迁移
    success = migrate_old_scores_to_new_format(old_file, new_file)
    
    if success:
        print()
        print("=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print("✓ Old data backed up")
        print("✓ New format created")
        print()
        print("Next steps:")
        print("1. Review the new file to ensure data looks correct")
        print("2. Run your agent to verify it loads the new format")
        print("3. If everything works, you can delete the old backup")
        return 0
    else:
        print()
        print("Migration not performed (old file not found or error occurred)")
        print("The system will start with fresh state on next run")
        return 0


if __name__ == "__main__":
    sys.exit(main())
