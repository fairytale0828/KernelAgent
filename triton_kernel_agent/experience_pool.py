# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""经验池管理器 - 存储和管理高性能 kernel 经验"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

from .op_classifier import OpType


class ExperienceEntry:
    """经验条目"""
    
    def __init__(
        self,
        op_type: str,
        strategy_id: str,
        best_speedup: float,
        kernel_code: str,
        tile_config: Optional[Dict[str, Any]] = None,
        shared_mem_layout: Optional[Dict[str, Any]] = None,
        hardware_sig: Optional[Dict[str, Any]] = None,
        problem_info: Optional[Dict[str, Any]] = None,
    ):
        self.id = self._generate_id(op_type, strategy_id, best_speedup)
        self.op_type = op_type
        self.strategy_id = strategy_id
        self.best_speedup = best_speedup
        self.kernel_code = kernel_code
        self.tile_config = tile_config or {}
        self.shared_mem_layout = shared_mem_layout or {}
        self.hardware_sig = hardware_sig or {}
        self.problem_info = problem_info or {}
        self.created_at = datetime.now().isoformat()
    
    def _generate_id(self, op_type: str, strategy_id: str, speedup: float) -> str:
        """生成唯一ID"""
        content = f"{op_type}_{strategy_id}_{speedup:.4f}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "op_type": self.op_type,
            "strategy_id": self.strategy_id,
            "best_speedup": self.best_speedup,
            "kernel_code": self.kernel_code,
            "tile_config": self.tile_config,
            "shared_mem_layout": self.shared_mem_layout,
            "hardware_sig": self.hardware_sig,
            "problem_info": self.problem_info,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceEntry":
        """从字典创建"""
        entry = cls(
            op_type=data["op_type"],
            strategy_id=data["strategy_id"],
            best_speedup=data["best_speedup"],
            kernel_code=data["kernel_code"],
            tile_config=data.get("tile_config"),
            shared_mem_layout=data.get("shared_mem_layout"),
            hardware_sig=data.get("hardware_sig"),
            problem_info=data.get("problem_info"),
        )
        entry.id = data.get("id", entry.id)
        entry.created_at = data.get("created_at", entry.created_at)
        return entry


class ExperiencePool:
    """经验池管理器"""
    
    def __init__(self, pool_file: Optional[Path] = None):
        """
        初始化经验池
        
        Args:
            pool_file: 经验池文件路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 经验池文件
        if pool_file is None:
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            self.pool_file = data_dir / "experience_pool.json"
        else:
            self.pool_file = Path(pool_file)
            self.pool_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 经验池数据: {bucket_key: [ExperienceEntry, ...]}
        self.pool: Dict[str, List[ExperienceEntry]] = {}
        
        # 加载现有经验
        self._load_pool()
    
    def _get_bucket_key(self, op_type: str, hardware_sig: Optional[Dict] = None) -> str:
        """
        获取桶的键
        
        Args:
            op_type: 算子类型
            hardware_sig: 硬件签名
            
        Returns:
            桶键，格式: "op_type:hardware"
        """
        if hardware_sig and "arch" in hardware_sig:
            return f"{op_type}:{hardware_sig['arch']}"
        return f"{op_type}:default"
    
    def _load_pool(self):
        """从文件加载经验池"""
        if not self.pool_file.exists():
            self.logger.info("Experience pool file not found, starting fresh")
            return
        
        try:
            with open(self.pool_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 转换为 ExperienceEntry 对象
            for bucket_key, entries_data in data.items():
                self.pool[bucket_key] = [
                    ExperienceEntry.from_dict(entry_data)
                    for entry_data in entries_data
                ]
            
            total_entries = sum(len(entries) for entries in self.pool.values())
            self.logger.info(
                f"Loaded experience pool: {len(self.pool)} buckets, "
                f"{total_entries} total entries"
            )
        except Exception as e:
            self.logger.error(f"Failed to load experience pool: {e}")
            self.pool = {}
    
    def save_pool(self):
        """保存经验池到文件"""
        try:
            # 转换为可序列化的字典
            data = {
                bucket_key: [entry.to_dict() for entry in entries]
                for bucket_key, entries in self.pool.items()
            }
            
            with open(self.pool_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved experience pool to {self.pool_file}")
        except Exception as e:
            self.logger.error(f"Failed to save experience pool: {e}")
    
    def add_experience(
        self,
        op_type: str,
        strategy_id: str,
        best_speedup: float,
        kernel_code: str,
        tile_config: Optional[Dict[str, Any]] = None,
        shared_mem_layout: Optional[Dict[str, Any]] = None,
        hardware_sig: Optional[Dict[str, Any]] = None,
        problem_info: Optional[Dict[str, Any]] = None,
    ) -> ExperienceEntry:
        """
        添加经验到池中
        
        Args:
            op_type: 算子类型
            strategy_id: 策略ID
            best_speedup: 最佳加速比
            kernel_code: kernel 代码
            tile_config: tile 配置
            shared_mem_layout: 共享内存布局
            hardware_sig: 硬件签名
            problem_info: 问题信息
            
        Returns:
            创建的经验条目
        """
        entry = ExperienceEntry(
            op_type=op_type,
            strategy_id=strategy_id,
            best_speedup=best_speedup,
            kernel_code=kernel_code,
            tile_config=tile_config,
            shared_mem_layout=shared_mem_layout,
            hardware_sig=hardware_sig,
            problem_info=problem_info,
        )
        
        bucket_key = self._get_bucket_key(op_type, hardware_sig)
        if bucket_key not in self.pool:
            self.pool[bucket_key] = []
        
        self.pool[bucket_key].append(entry)
        self.save_pool()
        
        self.logger.info(
            f"Added experience {entry.id} to bucket {bucket_key}: "
            f"strategy={strategy_id}, speedup={best_speedup:.2f}x"
        )
        
        return entry
    
    def get_experiences(
        self,
        op_type: str,
        hardware_sig: Optional[Dict] = None,
        min_speedup: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[ExperienceEntry]:
        """
        获取经验列表
        
        Args:
            op_type: 算子类型
            hardware_sig: 硬件签名
            min_speedup: 最小加速比过滤
            top_k: 返回前k个最佳经验
            
        Returns:
            经验列表
        """
        bucket_key = self._get_bucket_key(op_type, hardware_sig)
        entries = self.pool.get(bucket_key, [])
        
        # 过滤
        if min_speedup is not None:
            entries = [e for e in entries if e.best_speedup >= min_speedup]
        
        # 排序
        entries = sorted(entries, key=lambda e: e.best_speedup, reverse=True)
        
        # 限制数量
        if top_k is not None:
            entries = entries[:top_k]
        
        return entries
    
    def can_create_fusion(
        self,
        op_type: str,
        hardware_sig: Optional[Dict] = None,
        min_parents: int = 2,
    ) -> bool:
        """
        判断是否可以创建融合 Worker
        
        Args:
            op_type: 算子类型
            hardware_sig: 硬件签名
            min_parents: 最少需要的父代数量
            
        Returns:
            是否可以创建融合
        """
        experiences = self.get_experiences(op_type, hardware_sig)
        return len(experiences) >= min_parents
    
    def select_fusion_parents(
        self,
        op_type: str,
        hardware_sig: Optional[Dict] = None,
        num_parents: int = 2,
    ) -> List[ExperienceEntry]:
        """
        选择融合的父代经验
        
        Args:
            op_type: 算子类型
            hardware_sig: 硬件签名
            num_parents: 父代数量
            
        Returns:
            父代经验列表
        """
        experiences = self.get_experiences(op_type, hardware_sig, top_k=num_parents)
        return experiences
    
    def get_bucket_count(self, op_type: str, hardware_sig: Optional[Dict] = None) -> int:
        """获取指定桶中的经验数量"""
        bucket_key = self._get_bucket_key(op_type, hardware_sig)
        return len(self.pool.get(bucket_key, []))
