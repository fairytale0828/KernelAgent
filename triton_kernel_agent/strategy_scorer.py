"""策略得分矩阵管理器 - 基于Bayesian Bandit的策略选择"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from .op_classifier import OpType
from .strategy_manager import StrategyManager


class StrategyScorer:
    """管理算子类别与优化策略的得分矩阵 - 使用先验知识和UCB探索"""

    def __init__(
        self, 
        score_file: Optional[Path] = None,
        prior_weight: float = 5.0,
        exploration_coef: float = 2.0
    ):
        """
        初始化策略得分管理器

        Args:
            score_file: 得分矩阵持久化文件路径（应在data/目录下，不在logs/）
            prior_weight: 先验权重n0，控制先验知识的影响力
            exploration_coef: UCB探索系数c
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategy_manager = StrategyManager()

        # 统计数据: {op_type: {strategy_id: {"n": count, "mu_data": avg_speedup}}}
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        # 先验知识: {op_type: {strategy_id: mu_prior}}
        self.priors: Dict[str, Dict[str, float]] = {}
        
        # 超参数
        self.prior_weight = prior_weight  # n0
        self.exploration_coef = exploration_coef  # c
        self.global_iteration = 0  # T - 全局迭代计数

        # 持久化文件 - 确保在data/目录下
        if score_file is None:
            # 默认使用项目根目录下的data/目录
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            self.score_file = data_dir / "policy_state.json"
        else:
            self.score_file = Path(score_file)
            # 确保父目录存在
            self.score_file.parent.mkdir(parents=True, exist_ok=True)

        # 初始化或加载状态
        self._init_state()

    def _init_state(self):
        """初始化或加载策略状态"""
        # 尝试从文件加载
        if self.score_file.exists():
            try:
                with open(self.score_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stats = data.get("stats", {})
                    self.priors = data.get("priors", {})
                    self.global_iteration = data.get("global_iteration", 0)
                self.logger.info(f"Loaded policy state from {self.score_file}")
                self.logger.info(f"Global iteration: {self.global_iteration}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load policy state: {e}")

        # 初始化新状态
        self._init_fresh_state()
        self.logger.info("Initialized fresh policy state with priors")

    def _init_fresh_state(self):
        """初始化全新的策略状态（包含先验知识）"""
        strategy_ids = self.strategy_manager.get_strategy_ids()
        
        # 为每个算子类型初始化统计和先验
        for op_type in OpType:
            if op_type == OpType.UNKNOWN:
                continue
                
            op_key = op_type.value
            self.stats[op_key] = {}
            self.priors[op_key] = {}
            
            for strategy_id in strategy_ids:
                # 初始化统计数据
                self.stats[op_key][strategy_id] = {
                    "n": 0,
                    "mu_data": 0.0
                }
                
                # 设置先验知识（基于经验的合理猜测）
                self.priors[op_key][strategy_id] = self._get_prior_speedup(
                    op_type, strategy_id
                )
        
        self.global_iteration = 0

    def _get_prior_speedup(self, op_type: OpType, strategy_id: str) -> float:
        """
        获取先验speedup期望（基于领域知识的合理猜测）
        
        这些先验值反映了我们对不同策略在不同算子上的预期性能
        """
        # 默认先验：假设能达到PyTorch性能（speedup=1.0）
        default_prior = 1.0
        
        # 针对特定算子类型和策略的先验知识
        priors_map = {
            OpType.MATMUL: {
                "Triton.TensorCore.v1": 1.0,
                "Triton.TileOnly.v1": 1.0,
                "Triton.SharedMemOpt.v1": 1.0,
                "Triton.AsyncCopy.v1": 1.0,
            },
            OpType.CONV2D: {
                "Triton.TensorCore.v1": 1.0,
                "Triton.SharedMemOpt.v1": 1.0,
                "Triton.TileOnly.v1": 1.0,
            },
            OpType.REDUCE: {
                "Triton.ReduceOpt.v1": 1.0,
                "Triton.WarpOpt.v1": 1.0,
                "Triton.SharedMemOpt.v1": 1.0,
            },
            OpType.ELEMENTWISE: {
                "Triton.CoalescedAccess.v1": 1.0,
                "Triton.TileVectorize.v1": 1.0,
                "Triton.TileOnly.v1": 1.0,
            },
            OpType.NORMALIZATION: {
                "Triton.ReduceOpt.v1": 1.0,
                "Triton.WarpOpt.v1": 1.0,
                "Triton.SharedMemOpt.v1": 1.0,
            },
            OpType.ATTENTION: {
                "Triton.TensorCore.v1": 1.0,
                "Triton.SharedMemOpt.v1": 1.0,
                "Triton.AsyncCopy.v1": 1.0,
            },
        }
        
        return priors_map.get(op_type, {}).get(strategy_id, default_prior)

    def save_state(self):
        """保存策略状态到文件"""
        try:
            data = {
                "stats": self.stats,
                "priors": self.priors,
                "global_iteration": self.global_iteration,
                "metadata": {
                    "prior_weight": self.prior_weight,
                    "exploration_coef": self.exploration_coef,
                }
            }
            with open(self.score_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved policy state to {self.score_file}")
        except Exception as e:
            self.logger.error(f"Failed to save policy state: {e}")

    def get_top_strategies(
        self, 
        op_type: OpType, 
        top_k: int = 3, 
        exploration_rate: float = 0.2,
        increment_iteration: bool = True
    ) -> List[str]:
        """
        获取指定算子类型的Top-K策略（使用UCB选择）

        Args:
            op_type: 算子类型
            top_k: 返回前K个策略
            exploration_rate: 探索率（0-1），控制exploitation vs exploration平衡
            increment_iteration: 是否增加全局迭代计数（默认True，仅在任务开始时调用）

        Returns:
            策略ID列表
        """
        # # 仅在任务开始时增加全局迭代计数
        # # 中间更新时不增加（由 update_score 负责）
        # if increment_iteration:
        #     self.global_iteration += 1
        
        if op_type == OpType.UNKNOWN or op_type.value not in self.stats:
            # 未知类型，返回随机策略
            all_strategies = self.strategy_manager.get_strategy_ids()
            selected = random.sample(all_strategies, min(top_k, len(all_strategies)))
            self.logger.info(f"Unknown op_type, random selection: {selected}")
            return selected

        op_key = op_type.value
        
        # 计算每个策略的UCB分数
        strategy_scores = []
        for strategy_id in self.strategy_manager.get_strategy_ids():
            score = self._calculate_ucb_score(op_key, strategy_id)
            strategy_scores.append((strategy_id, score))
        
        # 按UCB分数排序
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择策略：混合exploitation和exploration
        num_exploit = max(1, int(top_k * (1 - exploration_rate)))
        num_explore = top_k - num_exploit
        
        # Exploitation: 选择UCB分数最高的
        selected_strategies = [s[0] for s in strategy_scores[:num_exploit]]
        
        # Exploration: 从剩余策略中随机选择
        if num_explore > 0:
            remaining = [s[0] for s in strategy_scores[num_exploit:]]
            if remaining:
                explore_strategies = random.sample(
                    remaining, 
                    min(num_explore, len(remaining))
                )
                selected_strategies.extend(explore_strategies)
        
        self.logger.info(
            f"Selected strategies for {op_type.value} (iter={self.global_iteration}): "
            f"{selected_strategies}"
        )
        
        # 记录前3个策略的UCB分数（用于调试）
        top_3_scores = strategy_scores[:3]
        self.logger.debug(
            f"Top UCB scores: " + 
            ", ".join([f"{s[0]}: {s[1]:.3f}" for s in top_3_scores])
        )
        
        return selected_strategies

    def _calculate_ucb_score(self, op_key: str, strategy_id: str) -> float:
        """
        计算UCB (Upper Confidence Bound) 分数
        
        UCB = BlendedMean + Bonus
        BlendedMean = (n * mu_data + n0 * mu_prior) / (n + n0)
        Bonus = c * sqrt(log(T) / (n + 1))
        
        Args:
            op_key: 算子类型键
            strategy_id: 策略ID
            
        Returns:
            UCB分数
        """
        stats = self.stats[op_key][strategy_id]
        n = stats["n"]
        mu_data = stats["mu_data"]
        mu_prior = self.priors[op_key][strategy_id]
        
        # 计算融合均值（先验 + 经验数据）
        blended_mean = (n * mu_data + self.prior_weight * mu_prior) / (n + self.prior_weight)
        
        # 计算探索加成（UCB bonus）
        if self.global_iteration > 0:
            bonus = self.exploration_coef * math.sqrt(
                math.log(self.global_iteration) / (n + 1)
            )
        else:
            bonus = float('inf')  # 第一轮：所有策略都有无限探索价值
        
        ucb_score = blended_mean + bonus
        
        return ucb_score

    def update_score(
        self, 
        op_type: OpType, 
        strategy_id: str, 
        speedup: float, 
        success: bool
    ):
        """
        更新策略统计数据（在线增量更新）

        Args:
            op_type: 算子类型
            strategy_id: 策略ID
            speedup: 加速比（相对于baseline）
            success: 是否编译成功且数值正确
        """
        if op_type == OpType.UNKNOWN or op_type.value not in self.stats:
            self.logger.warning(f"Cannot update score for unknown op_type: {op_type}")
            return

        op_key = op_type.value
        if strategy_id not in self.stats[op_key]:
            self.logger.warning(f"Unknown strategy_id: {strategy_id}")
            return

        stats = self.stats[op_key][strategy_id]
        n = stats["n"]
        mu_data = stats["mu_data"]

        if not success:
            # 失败：记录为speedup=0
            reward = 0.0
            self.logger.info(
                f"Strategy {strategy_id} failed for {op_type.value}"
            )
        else:
            # 成功：使用实际speedup
            reward = speedup
            self.logger.info(
                f"Strategy {strategy_id} succeeded for {op_type.value} "
                f"with speedup {speedup:.2f}x"
            )

        # 在线增量更新均值（Welford's method）
        n_new = n + 1
        mu_data_new = mu_data + (reward - mu_data) / n_new
        
        # 更新统计数据
        self.stats[op_key][strategy_id]["n"] = n_new
        self.stats[op_key][strategy_id]["mu_data"] = mu_data_new
        
        # 每次更新统计时，增加全局迭代计数
        # 这样 global_iteration 代表"总共做了多少次更新"
        self.global_iteration += 1
        
        # 计算当前的融合均值（用于日志）
        mu_prior = self.priors[op_key][strategy_id]
        blended_mean = (n_new * mu_data_new + self.prior_weight * mu_prior) / (
            n_new + self.prior_weight
        )
        
        self.logger.info(
            f"Updated stats for {strategy_id} on {op_type.value}: "
            f"n={n_new}, mu_data={mu_data_new:.3f}, blended_mean={blended_mean:.3f}, "
            f"global_iteration={self.global_iteration}"
        )

        # 自动保存
        self.save_state()

    def get_blended_mean(self, op_type: OpType, strategy_id: str) -> float:
        """获取指定算子类型和策略的融合均值（先验+数据）"""
        if op_type == OpType.UNKNOWN or op_type.value not in self.stats:
            return 1.0  # 默认值
        
        op_key = op_type.value
        stats = self.stats[op_key][strategy_id]
        n = stats["n"]
        mu_data = stats["mu_data"]
        mu_prior = self.priors[op_key][strategy_id]
        
        blended_mean = (n * mu_data + self.prior_weight * mu_prior) / (n + self.prior_weight)
        return blended_mean

    def get_all_stats(self, op_type: OpType) -> Dict[str, Dict[str, float]]:
        """获取指定算子类型的所有策略统计数据"""
        if op_type == OpType.UNKNOWN or op_type.value not in self.stats:
            return {}
        return self.stats[op_type.value].copy()

    def reset_stats(self, op_type: Optional[OpType] = None):
        """
        重置策略统计数据

        Args:
            op_type: 如果指定，只重置该算子类型的统计；否则重置所有
        """
        if op_type and op_type != OpType.UNKNOWN:
            op_key = op_type.value
            if op_key in self.stats:
                for strategy_id in self.stats[op_key]:
                    self.stats[op_key][strategy_id] = {
                        "n": 0,
                        "mu_data": 0.0
                    }
                self.logger.info(f"Reset stats for {op_type.value}")
        else:
            self._init_fresh_state()
            self.logger.info("Reset all stats")

        self.save_state()
