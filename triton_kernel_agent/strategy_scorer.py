"""策略得分矩阵管理器"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from .op_classifier import OpType
from .strategy_manager import StrategyManager


class StrategyScorer:
    """管理算子类别与优化策略的得分矩阵"""

    def __init__(self, score_file: Optional[Path] = None):
        """
        初始化策略得分管理器

        Args:
            score_file: 得分矩阵持久化文件路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategy_manager = StrategyManager()

        # 得分矩阵: {op_type: {strategy_id: score}}
        self.score_matrix: Dict[str, Dict[str, float]] = {}

        # 持久化文件
        self.score_file = score_file or Path("strategy_scores.json")

        # 初始化或加载得分矩阵
        self._init_score_matrix()

    def _init_score_matrix(self):
        """初始化得分矩阵"""
        # 尝试从文件加载
        if self.score_file.exists():
            try:
                with open(self.score_file, "r", encoding="utf-8") as f:
                    self.score_matrix = json.load(f)
                self.logger.info(f"Loaded score matrix from {self.score_file}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load score matrix: {e}")

        # 初始化为全0矩阵
        strategy_ids = self.strategy_manager.get_strategy_ids()
        for op_type in OpType:
            if op_type != OpType.UNKNOWN:
                self.score_matrix[op_type.value] = {
                    strategy_id: 0.0 for strategy_id in strategy_ids
                }

        self.logger.info("Initialized score matrix with zeros")

    def save_scores(self):
        """保存得分矩阵到文件"""
        try:
            with open(self.score_file, "w", encoding="utf-8") as f:
                json.dump(self.score_matrix, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved score matrix to {self.score_file}")
        except Exception as e:
            self.logger.error(f"Failed to save score matrix: {e}")

    def get_top_strategies(
        self, 
        op_type: OpType, 
        top_k: int = 3, 
        exploration_rate: float = 0.2
    ) -> List[str]:
        """
        获取指定算子类型的Top-K策略

        Args:
            op_type: 算子类型
            top_k: 返回前K个策略
            exploration_rate: 探索率，用于随机选择策略

        Returns:
            策略ID列表
        """
        if op_type == OpType.UNKNOWN or op_type.value not in self.score_matrix:
            # 未知类型，返回随机策略
            all_strategies = self.strategy_manager.get_strategy_ids()
            return random.sample(all_strategies, min(top_k, len(all_strategies)))

        scores = self.score_matrix[op_type.value]

        # 按得分排序
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 选择Top-K策略
        top_strategies = [s[0] for s in sorted_strategies[:top_k]]

        # 添加探索：以一定概率随机选择其他策略
        num_explore = int(top_k * exploration_rate)
        if num_explore > 0:
            all_strategies = self.strategy_manager.get_strategy_ids()
            other_strategies = [s for s in all_strategies if s not in top_strategies]
            if other_strategies:
                explore_strategies = random.sample(
                    other_strategies, 
                    min(num_explore, len(other_strategies))
                )
                # 替换部分top策略
                top_strategies = top_strategies[:-num_explore] + explore_strategies

        self.logger.info(f"Selected strategies for {op_type.value}: {top_strategies}")
        return top_strategies

    def update_score(
        self, 
        op_type: OpType, 
        strategy_id: str, 
        speedup: float, 
        success: bool,
        learning_rate: float = 0.1
    ):
        """
        更新策略得分

        Args:
            op_type: 算子类型
            strategy_id: 策略ID
            speedup: 加速比（相对于baseline）
            success: 是否编译成功且数值正确
            learning_rate: 学习率，用于移动平均
        """
        if op_type == OpType.UNKNOWN or op_type.value not in self.score_matrix:
            self.logger.warning(f"Cannot update score for unknown op_type: {op_type}")
            return

        if strategy_id not in self.score_matrix[op_type.value]:
            self.logger.warning(f"Unknown strategy_id: {strategy_id}")
            return

        current_score = self.score_matrix[op_type.value][strategy_id]

        if not success:
            # 失败：负反馈
            new_score = current_score - 1.0
            self.logger.info(
                f"Strategy {strategy_id} failed for {op_type.value}, "
                f"score: {current_score:.2f} -> {new_score:.2f}"
            )
        else:
            # 成功：根据speedup更新
            if speedup > 0.8:
                # 显著加速，给予正反馈
                reward = speedup - 0.8
                # 使用移动平均更新
                new_score = (1 - learning_rate) * current_score + learning_rate * (current_score + reward)
                self.logger.info(
                    f"Strategy {strategy_id} succeeded for {op_type.value} "
                    f"with speedup {speedup:.2f}, score: {current_score:.2f} -> {new_score:.2f}"
                )
            else:
                # 加速不明显，小幅更新
                new_score = (1 - learning_rate) * current_score + learning_rate * speedup
                self.logger.info(
                    f"Strategy {strategy_id} succeeded for {op_type.value} "
                    f"with modest speedup {speedup:.2f}, score: {current_score:.2f} -> {new_score:.2f}"
                )

        self.score_matrix[op_type.value][strategy_id] = new_score

        # 自动保存
        self.save_scores()

    def get_score(self, op_type: OpType, strategy_id: str) -> float:
        """获取指定算子类型和策略的得分"""
        if op_type == OpType.UNKNOWN or op_type.value not in self.score_matrix:
            return 0.0
        return self.score_matrix[op_type.value].get(strategy_id, 0.0)

    def get_all_scores(self, op_type: OpType) -> Dict[str, float]:
        """获取指定算子类型的所有策略得分"""
        if op_type == OpType.UNKNOWN or op_type.value not in self.score_matrix:
            return {}
        return self.score_matrix[op_type.value].copy()

    def reset_scores(self, op_type: Optional[OpType] = None):
        """
        重置得分矩阵

        Args:
            op_type: 如果指定，只重置该算子类型的得分；否则重置所有
        """
        if op_type and op_type != OpType.UNKNOWN:
            if op_type.value in self.score_matrix:
                strategy_ids = self.strategy_manager.get_strategy_ids()
                self.score_matrix[op_type.value] = {
                    strategy_id: 0.0 for strategy_id in strategy_ids
                }
                self.logger.info(f"Reset scores for {op_type.value}")
        else:
            self._init_score_matrix()
            self.logger.info("Reset all scores")

        self.save_scores()
