"""
Nsys性能分析工具
用于对Triton kernel进行nsys profiling并解析性能瓶颈
"""

import json
import logging
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class NsysProfiler:
    """Nsys性能分析器，用于分析Triton kernel的性能瓶颈"""

    def __init__(self, nsys_path: Optional[str] = None):
        """
        初始化Nsys分析器

        Args:
            nsys_path: nsys可执行文件路径，如果为None则从PATH查找
        """
        self.nsys_path = nsys_path or shutil.which("nsys")
        if not self.nsys_path:
            logger.warning(
                "nsys not found in PATH. Nsys profiling will be disabled. "
                "Please install NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems"
            )
            self.available = False
        else:
            self.available = True
            logger.info(f"Using nsys at: {self.nsys_path}")

    def profile_kernel(
        self,
        test_file: Path,
        workdir: Path,
        output_dir: Path,
        round_num: int,
        warmup_runs: int = 3,
        profile_runs: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        对kernel进行nsys profiling

        Args:
            test_file: 测试文件路径
            workdir: 工作目录
            output_dir: 输出目录
            round_num: 轮次编号
            warmup_runs: 预热运行次数
            profile_runs: profiling运行次数

        Returns:
            包含profiling结果的字典，失败返回None
        """
        if not self.available:
            return None

        try:
            # 生成nsys报告文件路径
            nsys_report = output_dir / \
                f"nsys_report_round_{round_num}.nsys-rep"
            nsys_stats = output_dir / f"nsys_stats_round_{round_num}.txt"

            # 首先尝试使用gpu-metrics-device（如果支持）
            # 如果不支持，回退到基本配置
            cmd_variants = [
                # 尝试1: 使用gpu-metrics-device all（如果GPU支持）
                [
                    self.nsys_path,
                    "profile",
                    "--output", str(nsys_report),
                    "--force-overwrite", "true",
                    "--trace", "cuda,nvtx",
                    "--cuda-memory-usage", "true",
                    "--gpu-metrics-device", "all",
                    sys.executable,
                    str(test_file),
                ],
                # 尝试2: 使用gpu-metrics-device 0（单个GPU）
                [
                    self.nsys_path,
                    "profile",
                    "--output", str(nsys_report),
                    "--force-overwrite", "true",
                    "--trace", "cuda,nvtx",
                    "--cuda-memory-usage", "true",
                    "--gpu-metrics-device", "0",
                    sys.executable,
                    str(test_file),
                ],
                # 尝试3: 不使用gpu-metrics-device（基本配置）
                [
                    self.nsys_path,
                    "profile",
                    "--output", str(nsys_report),
                    "--force-overwrite", "true",
                    "--trace", "cuda,nvtx",
                    "--cuda-memory-usage", "true",
                    sys.executable,
                    str(test_file),
                ],
            ]

            logger.info(f"Running nsys profiling for round {round_num}...")
            result = None
            last_error = None

            # 尝试不同的命令变体
            for i, cmd in enumerate(cmd_variants):
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(workdir),
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5分钟超时
                    )

                    if result.returncode == 0:
                        logger.info(
                            f"nsys profiling succeeded with variant {i+1}"
                        )
                        break
                    else:
                        last_error = result.stderr
                        if i < len(cmd_variants) - 1:
                            logger.debug(
                                f"nsys variant {i+1} failed, trying next variant..."
                            )
                except subprocess.TimeoutExpired:
                    logger.warning(f"nsys variant {i+1} timed out")
                    if i < len(cmd_variants) - 1:
                        continue
                    else:
                        raise
                except Exception as e:
                    logger.debug(f"nsys variant {i+1} error: {e}")
                    if i < len(cmd_variants) - 1:
                        continue
                    else:
                        raise

            if result is None or result.returncode != 0:
                logger.warning(
                    f"nsys profiling failed after trying all variants: {last_error[:500] if last_error else 'Unknown error'}"
                )
                return None

            # 生成统计报告（文本格式，更易解析）
            stats_txt_cmd = [
                self.nsys_path,
                "stats",
                "--report", "gputrace",
                "--format", "table",  # 使用table格式而不是csv
                str(nsys_report),
            ]

            stats_txt_result = subprocess.run(
                stats_txt_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            stats_txt_file = None
            if stats_txt_result.returncode == 0:
                stats_txt_file = output_dir / \
                    f"nsys_stats_round_{round_num}.txt"
                stats_txt_file.write_text(stats_txt_result.stdout)
            else:
                logger.warning("Failed to generate nsys stats text report")

            # 生成统计报告（CSV格式，作为备用）
            stats_csv_cmd = [
                self.nsys_path,
                "stats",
                "--report", "gputrace",
                "--format", "csv",
                str(nsys_report),
            ]

            stats_csv_result = subprocess.run(
                stats_csv_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            stats_csv_file = None
            if stats_csv_result.returncode == 0:
                stats_csv_file = output_dir / \
                    f"nsys_stats_round_{round_num}.csv"
                stats_csv_file.write_text(stats_csv_result.stdout)

            # 解析性能瓶颈（使用文本输出）
            bottlenecks = self._parse_bottlenecks(
                nsys_report,
                stats_txt_result.stdout if stats_txt_result.returncode == 0 else "",
                stats_csv_result.stdout if stats_csv_result.returncode == 0 else ""
            )

            return {
                "success": True,
                "nsys_report": str(nsys_report),
                "nsys_stats_txt": str(stats_txt_file) if stats_txt_file else None,
                "nsys_stats_csv": str(stats_csv_file) if stats_csv_file else None,
                "bottlenecks": bottlenecks,
                "round": round_num,
            }

        except subprocess.TimeoutExpired:
            logger.error("nsys profiling timed out")
            return None
        except Exception as e:
            logger.error(f"nsys profiling failed: {e}")
            return None

    def _create_profile_script(
        self, test_file: Path, warmup_runs: int, profile_runs: int
    ) -> str:
        """创建用于profiling的Python脚本"""
        return f"""#!/usr/bin/env python3
import sys
import torch
import time
from pathlib import Path

# 导入kernel和测试
sys.path.insert(0, str(Path("{test_file}").parent))

from kernel import kernel_function
import test_kernel

# 准备测试输入
test_inputs = test_kernel.get_test_inputs()

# 预热
for _ in range({warmup_runs}):
    with torch.no_grad():
        result = kernel_function(*test_inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Profiling运行
for _ in range({profile_runs}):
    with torch.no_grad():
        result = kernel_function(*test_inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
"""

    def _parse_bottlenecks(
        self,
        nsys_report: Path,
        stats_txt: str,
        stats_csv: str
    ) -> Dict[str, Any]:
        """
        从nsys报告中解析性能瓶颈

        Args:
            nsys_report: nsys报告文件路径
            stats_txt: nsys stats文本输出
            stats_csv: nsys stats CSV输出

        Returns:
            包含瓶颈信息的字典
        """
        bottlenecks = {
            "memory_bandwidth_utilization": None,
            "compute_utilization": None,
            "gpu_utilization": None,
            "warp_efficiency": None,
            "occupancy": None,
            "memory_throughput": None,
            "kernel_duration": None,
            "top_kernels": [],
            "bottleneck_summary": [],
        }

        try:
            # 从文本stats输出中提取信息（主要方法）
            if stats_txt:
                txt_info = self._parse_stats_output(stats_txt)
                bottlenecks.update(txt_info)

            # 尝试从SQLite数据库提取信息（如果存在）
            # nsys stats会自动生成.sqlite文件
            sqlite_file = nsys_report.parent / f"{nsys_report.stem}.sqlite"
            if sqlite_file.exists():
                db_info = self._parse_nsys_database(sqlite_file)
                bottlenecks.update(db_info)

        except Exception as e:
            logger.warning(f"Failed to parse bottlenecks: {e}")

        return bottlenecks

    def _parse_nsys_database(self, sqlite_file: Path) -> Dict[str, Any]:
        """从nsys SQLite数据库中解析信息"""
        info = {}
        try:
            conn = sqlite3.connect(str(sqlite_file))
            cursor = conn.cursor()

            # 查询CUDA kernel信息
            try:
                cursor.execute("""
                    SELECT 
                        name,
                        AVG(duration) as avg_duration,
                        SUM(duration) as total_duration,
                        COUNT(*) as count
                    FROM CUPTI_ACTIVITY_KIND_KERNEL
                    GROUP BY name
                    ORDER BY total_duration DESC
                    LIMIT 10
                """)
                kernels = cursor.fetchall()
                if kernels:
                    info["top_kernels"] = [
                        {
                            "name": k[0],
                            "avg_duration_us": k[1],
                            "total_duration_us": k[2],
                            "count": k[3],
                        }
                        for k in kernels
                    ]
            except sqlite3.OperationalError as e:
                logger.debug(f"Could not query kernel table: {e}")

            # 查询内存传输信息
            try:
                cursor.execute("""
                    SELECT 
                        AVG(bytes) as avg_bytes,
                        SUM(bytes) as total_bytes,
                        COUNT(*) as count
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY
                """)
                memcpy = cursor.fetchone()
                if memcpy and memcpy[0]:
                    info["memory_transfer"] = {
                        "avg_bytes": memcpy[0],
                        "total_bytes": memcpy[1],
                        "count": memcpy[2],
                    }
            except sqlite3.OperationalError as e:
                logger.debug(f"Could not query memcpy table: {e}")

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to parse nsys SQLite database: {e}")

        return info

    def _parse_stats_output(self, stats_output: str) -> Dict[str, Any]:
        """从nsys stats输出中解析信息"""
        info = {}
        bottleneck_summary = []

        # 解析GPU利用率（多种可能的格式）
        gpu_util_patterns = [
            r"GPU Utilization[:\s]+([\d.]+)%",
            r"GPU\s+Util[:\s]+([\d.]+)%",
            r"Utilization[:\s]+([\d.]+)%",
        ]

        for pattern in gpu_util_patterns:
            gpu_util_match = re.search(pattern, stats_output, re.IGNORECASE)
            if gpu_util_match:
                gpu_util = float(gpu_util_match.group(1)) / 100.0
                info["gpu_utilization"] = gpu_util
                if gpu_util < 0.5:
                    bottleneck_summary.append(
                        f"GPU利用率较低 ({gpu_util*100:.1f}%)，可能存在内存带宽瓶颈或计算资源未充分利用"
                    )
                break

        # 解析内存带宽
        mem_bandwidth_patterns = [
            r"Memory Bandwidth[:\s]+([\d.]+)\s*GB/s",
            r"Mem Bandwidth[:\s]+([\d.]+)\s*GB/s",
            r"Bandwidth[:\s]+([\d.]+)\s*GB/s",
        ]

        for pattern in mem_bandwidth_patterns:
            mem_bandwidth_match = re.search(
                pattern, stats_output, re.IGNORECASE)
            if mem_bandwidth_match:
                mem_bandwidth = float(mem_bandwidth_match.group(1))
                info["memory_bandwidth_gbs"] = mem_bandwidth
                if mem_bandwidth < 100:
                    bottleneck_summary.append(
                        f"内存带宽较低 ({mem_bandwidth:.2f} GB/s)，可能存在内存访问瓶颈"
                    )
                break

        # 解析kernel执行时间（查找包含时间信息的行）
        kernel_time_patterns = [
            r"([\w.]+)\s+([\d.]+)\s*ms",
            r"([\w.]+)\s+([\d.]+)\s*µs",
        ]

        for pattern in kernel_time_patterns:
            kernel_time_matches = re.findall(pattern, stats_output)
            if kernel_time_matches:
                # 转换微秒到毫秒
                kernel_times = []
                for name, time_str in kernel_time_matches[:5]:
                    time_val = float(time_str)
                    if "µs" in stats_output[stats_output.find(f"{name} {time_str}"):stats_output.find(f"{name} {time_str}")+50]:
                        time_val = time_val / 1000.0  # 微秒转毫秒
                    kernel_times.append({"name": name, "time_ms": time_val})
                if kernel_times:
                    info["kernel_times"] = kernel_times
                break

        # 如果没有找到任何指标，尝试从输出中提取基本信息
        if not info and stats_output:
            # 查找包含"Time"或"Duration"的行
            time_lines = [line for line in stats_output.split('\n')
                          if 'time' in line.lower() or 'duration' in line.lower()]
            if time_lines:
                info["raw_stats_lines"] = time_lines[:10]

        # 生成瓶颈摘要
        if bottleneck_summary:
            info["bottleneck_summary"] = bottleneck_summary
        else:
            info["bottleneck_summary"] = ["未发现明显性能瓶颈"]

        return info

    def generate_optimization_suggestions(
        self, bottlenecks: Dict[str, Any]
    ) -> str:
        """
        基于瓶颈信息生成优化建议

        Args:
            bottlenecks: 瓶颈信息字典

        Returns:
            优化建议文本
        """
        suggestions = []

        # 内存带宽瓶颈
        if bottlenecks.get("memory_bandwidth_utilization") is not None:
            mem_util = bottlenecks["memory_bandwidth_utilization"]
            if mem_util < 0.5:
                suggestions.append(
                    "内存带宽利用率低，建议："
                    "- 使用向量化内存访问（tl.load/storage with vectorized offsets）"
                    "- 增加block size以提高内存合并"
                    "- 考虑使用shared memory进行数据重用"
                )

        # GPU利用率低
        if bottlenecks.get("gpu_utilization") is not None:
            gpu_util = bottlenecks["gpu_utilization"]
            if gpu_util < 0.5:
                suggestions.append(
                    "GPU利用率低，建议："
                    "- 增加occupancy（调整block size和num_warps）"
                    "- 减少kernel启动开销（合并多个操作）"
                    "- 优化内存访问模式以减少stall"
                )

        # Warp效率
        if bottlenecks.get("warp_efficiency") is not None:
            warp_eff = bottlenecks["warp_efficiency"]
            if warp_eff < 0.8:
                suggestions.append(
                    "Warp效率低，建议："
                    "- 减少分支发散（branch divergence）"
                    "- 优化条件判断逻辑"
                    "- 使用masked operations代替条件分支"
                )

        # 内存传输瓶颈
        if bottlenecks.get("memory_transfer"):
            suggestions.append(
                "检测到大量内存传输，建议："
                "- 减少host-device数据传输"
                "- 使用in-place操作"
                "- 优化数据布局以减少传输"
            )

        if not suggestions:
            suggestions.append("性能表现良好，可考虑微调优化")

        return "\n".join(suggestions)
