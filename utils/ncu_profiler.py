"""
NCU (NVIDIA Nsight Compute) 性能分析工具
用于对Triton kernel进行详细的kernel级别性能分析
"""

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class NcuProfiler:
    """NCU性能分析器，用于分析Triton kernel的详细性能瓶颈"""

    def __init__(self, ncu_path: Optional[str] = None):
        """
        初始化NCU分析器

        Args:
            ncu_path: ncu可执行文件路径，如果为None则从PATH查找
        """
        self.ncu_path = ncu_path or shutil.which("ncu")
        if not self.ncu_path:
            logger.warning(
                "ncu not found in PATH. NCU profiling will be disabled. "
                "Please install NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute"
            )
            self.available = False
        else:
            self.available = True
            logger.info(f"Using ncu at: {self.ncu_path}")
            # 在__init__中测试NCU
            if self.available:
                try:
                    test_result = subprocess.run(
                        [self.ncu_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if test_result.returncode == 0:
                        logger.info(
                            f"NCU version: {test_result.stdout.strip()}")
                except Exception as e:
                    logger.warning(f"Failed to get NCU version: {e}")

    def profile_kernel(
        self,
        test_file: Path,
        workdir: Path,
        output_dir: Path,
        round_num: int,
        warmup_runs: int = 3,
        profile_runs: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        对kernel进行NCU profiling

        Args:
            test_file: 测试文件路径
            workdir: 工作目录
            output_dir: 输出目录
            round_num: 轮次编号
            warmup_runs: 预热运行次数
            profile_runs: profiling运行次数（NCU通常只分析1次）

        Returns:
            包含profiling结果的字典，失败返回None
        """
        if not self.available:
            return None

        try:
            # 生成NCU报告文件路径
            ncu_csv = output_dir / f"ncu_stats_round_{round_num}.csv"
            ncu_txt = output_dir / f"ncu_stats_round_{round_num}.txt"

            # NCU命令：使用正确的参数格式
            # 尝试多种命令变体，因为不同版本的NCU参数可能不同
            cmd_variants = [
                # 方案1: 使用--set default和--export
                [
                    self.ncu_path,
                    "--set", "default",
                    "--launch-skip", str(warmup_runs),
                    "--launch-count", str(profile_runs),
                    "--export", str(ncu_csv),
                    "--force-overwrite",
                    sys.executable,
                    str(test_file),
                ],
                # 方案2: 使用--set speed-of-light（更快的指标集）
                [
                    self.ncu_path,
                    "--set", "speed-of-light",
                    "--launch-skip", str(warmup_runs),
                    "--launch-count", str(profile_runs),
                    "--export", str(ncu_csv),
                    "--force-overwrite",
                    sys.executable,
                    str(test_file),
                ],
                # 方案3: 不使用--set，让NCU自动选择
                [
                    self.ncu_path,
                    "--launch-skip", str(warmup_runs),
                    "--launch-count", str(profile_runs),
                    "--export", str(ncu_csv),
                    "--force-overwrite",
                    sys.executable,
                    str(test_file),
                ],
                # 方案4: 最简单的命令，只收集基本指标
                [
                    self.ncu_path,
                    "--launch-skip", str(warmup_runs),
                    "--launch-count", str(profile_runs),
                    "--force-overwrite",
                    sys.executable,
                    str(test_file),
                ],
                # 方案5: 基础模式 - 只收集基本信息（不需要性能计数器）
                [
                    self.ncu_path,
                    "--set", "none",  # 不收集性能计数器
                    "--print-summary", "per-kernel",
                    "--launch-skip", str(warmup_runs),
                    "--launch-count", str(profile_runs),
                    sys.executable,
                    str(test_file),
                ],
            ]

            logger.info(f"Running NCU profiling for round {round_num}...")
            result = None
            last_error = None

            # 尝试不同的命令变体
            for i, cmd in enumerate(cmd_variants):
                try:
                    logger.debug(f"Trying NCU variant {i+1}: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        cwd=str(workdir),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode == 0:
                        logger.info(
                            f"NCU profiling succeeded with variant {i+1}")
                        break
                    else:
                        last_error = result.stderr or result.stdout or "Unknown error"
                        if i < len(cmd_variants) - 1:
                            logger.debug(
                                f"NCU variant {i+1} failed (return code {result.returncode}), trying next variant..."
                            )
                            if result.stderr:
                                logger.debug(f"Error: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"NCU variant {i+1} timed out")
                    if i < len(cmd_variants) - 1:
                        continue
                    else:
                        raise
                except Exception as e:
                    logger.debug(f"NCU variant {i+1} error: {e}")
                    if i < len(cmd_variants) - 1:
                        continue
                    else:
                        raise

            if result is None or result.returncode != 0:
                error_msg = last_error[:500] if last_error else "Unknown error"

                # 检查是否是权限错误
                if "ERR_NVGPUCTRPERM" in error_msg or "permission" in error_msg.lower():
                    logger.warning(
                        f"NCU profiling failed due to permission error. "
                        f"Falling back to basic performance metrics from test output."
                    )
                    # 降级：从测试输出中提取基本性能信息
                    return self._fallback_to_basic_metrics(test_file, workdir, output_dir, round_num)
                else:
                    logger.warning(
                        f"NCU profiling failed: {error_msg}"
                    )
                return None

            # 保存文本输出
            if result.stdout:
                ncu_txt.write_text(result.stdout)
                logger.debug(f"Saved NCU output to {ncu_txt}")

            # 检查CSV文件是否生成
            if not ncu_csv.exists() and result.returncode == 0:
                logger.warning(f"NCU CSV file not generated: {ncu_csv}")

            # 解析NCU输出
            bottlenecks = self._parse_bottlenecks(
                result.stdout,
                ncu_csv if ncu_csv.exists() else None
            )

            return {
                "success": True,
                "ncu_csv": str(ncu_csv) if ncu_csv.exists() else None,
                "ncu_txt": str(ncu_txt) if ncu_txt.exists() else None,
                "ncu_output": result.stdout,
                "bottlenecks": bottlenecks,
                "round": round_num,
            }

        except subprocess.TimeoutExpired:
            logger.error("NCU profiling timed out")
            return None
        except Exception as e:
            logger.error(f"NCU profiling failed: {e}", exc_info=True)
            return None

    def _fallback_to_basic_metrics(
        self,
        test_file: Path,
        workdir: Path,
        output_dir: Path,
        round_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        当NCU权限不足时，降级到从测试输出中提取基本性能指标
        """
        try:
            # 运行测试并捕获输出
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            # 从stdout中解析性能信息
            bottlenecks = {
                "occupancy": None,
                "warp_efficiency": None,
                "memory_throughput": None,
                "compute_throughput": None,
                "memory_bandwidth_utilization": None,
                "stall_reasons": {},
                "top_kernels": [],
                "bottleneck_summary": ["NCU权限不足，使用基础性能指标"],
            }

            # 解析测试输出中的性能信息
            stdout = result.stdout
            if "PyTorch time:" in stdout and "Triton time:" in stdout:
                # 提取性能基准测试结果
                pytorch_match = re.search(
                    r"PyTorch time:\s*([\d.]+)\s*ms", stdout)
                triton_match = re.search(
                    r"Triton time:\s*([\d.]+)\s*ms", stdout)
                speedup_match = re.search(r"Speedup:\s*([\d.]+)\s*x", stdout)

                if pytorch_match and triton_match:
                    pytorch_time = float(pytorch_match.group(1))
                    triton_time = float(triton_match.group(1))
                    speedup = float(speedup_match.group(
                        1)) if speedup_match else pytorch_time / triton_time

                    bottlenecks["basic_performance"] = {
                        "pytorch_time_ms": pytorch_time,
                        "triton_time_ms": triton_time,
                        "speedup": speedup,
                    }

                    if speedup < 1.0:
                        bottlenecks["bottleneck_summary"].append(
                            f"性能低于PyTorch参考实现 (speedup={speedup:.2f}x)，需要优化"
                        )

            return {
                "success": True,
                "ncu_csv": None,
                "ncu_txt": None,
                "ncu_output": stdout,
                "bottlenecks": bottlenecks,
                "round": round_num,
                "fallback_mode": True,  # 标记为降级模式
            }
        except Exception as e:
            logger.warning(f"Fallback metrics extraction failed: {e}")
            return None

    def _parse_bottlenecks(
        self,
        ncu_output: str,
        ncu_csv: Optional[Path]
    ) -> Dict[str, Any]:
        """
        从NCU输出中解析性能瓶颈

        Args:
            ncu_output: NCU标准输出
            ncu_csv: NCU CSV报告文件路径

        Returns:
            包含瓶颈信息的字典
        """
        bottlenecks = {
            "occupancy": None,
            "warp_efficiency": None,
            "memory_throughput": None,
            "compute_throughput": None,
            "memory_bandwidth_utilization": None,
            "stall_reasons": {},
            "top_kernels": [],
            "bottleneck_summary": [],
        }

        try:
            # 从标准输出中解析关键指标
            if ncu_output:
                output_info = self._parse_ncu_output(ncu_output)
                bottlenecks.update(output_info)

            # 从CSV文件中解析详细信息
            if ncu_csv and ncu_csv.exists():
                csv_info = self._parse_ncu_csv(ncu_csv)
                bottlenecks.update(csv_info)

        except Exception as e:
            logger.warning(f"Failed to parse bottlenecks: {e}")

        return bottlenecks

    def _parse_ncu_output(self, output: str) -> Dict[str, Any]:
        """从NCU标准输出中解析信息"""
        info = {}
        bottleneck_summary = []

        # 解析Occupancy
        occupancy_match = re.search(
            r"Occupancy[:\s]+([\d.]+)%", output, re.IGNORECASE
        )
        if occupancy_match:
            occupancy = float(occupancy_match.group(1)) / 100.0
            info["occupancy"] = occupancy
            if occupancy < 0.5:
                bottleneck_summary.append(
                    f"Occupancy较低 ({occupancy*100:.1f}%)，建议增加block size或调整num_warps"
                )

        # 解析Warp Efficiency
        warp_eff_match = re.search(
            r"Warp\s+Efficiency[:\s]+([\d.]+)%", output, re.IGNORECASE
        )
        if warp_eff_match:
            warp_eff = float(warp_eff_match.group(1)) / 100.0
            info["warp_efficiency"] = warp_eff
            if warp_eff < 0.8:
                bottleneck_summary.append(
                    f"Warp效率较低 ({warp_eff*100:.1f}%)，存在分支发散或内存访问不连续问题"
                )

        # 解析Memory Throughput
        mem_throughput_match = re.search(
            r"Memory\s+Throughput[:\s]+([\d.]+)\s*GB/s", output, re.IGNORECASE
        )
        if mem_throughput_match:
            mem_throughput = float(mem_throughput_match.group(1))
            info["memory_throughput"] = mem_throughput

        # 解析Compute Throughput
        compute_throughput_match = re.search(
            r"Compute\s+Throughput[:\s]+([\d.]+)", output, re.IGNORECASE
        )
        if compute_throughput_match:
            compute_throughput = float(compute_throughput_match.group(1))
            info["compute_throughput"] = compute_throughput

        # 解析Stall Reasons
        stall_pattern = r"([\w\s]+)[:\s]+([\d.]+)%"
        stall_matches = re.findall(stall_pattern, output)
        if stall_matches:
            stall_reasons = {}
            for reason, percentage in stall_matches:
                if "stall" in reason.lower() or "wait" in reason.lower():
                    stall_reasons[reason.strip()] = float(percentage)
            if stall_reasons:
                info["stall_reasons"] = stall_reasons
                # 找出主要的stall原因
                main_stall = max(stall_reasons.items(), key=lambda x: x[1])
                if main_stall[1] > 30:
                    bottleneck_summary.append(
                        f"主要停顿原因: {main_stall[0]} ({main_stall[1]:.1f}%)"
                    )

        # 生成瓶颈摘要
        if bottleneck_summary:
            info["bottleneck_summary"] = bottleneck_summary
        else:
            info["bottleneck_summary"] = ["未发现明显性能瓶颈"]

        return info

    def _parse_ncu_csv(self, csv_file: Path) -> Dict[str, Any]:
        """从NCU CSV文件中解析详细信息"""
        info = {}
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 查找包含kernel信息的行
            kernels = []
            for line in lines:
                if 'kernel' in line.lower() or 'function' in line.lower():
                    # 解析kernel信息
                    parts = line.split(',')
                    if len(parts) >= 3:
                        kernels.append({
                            "name": parts[0].strip(),
                            "duration_us": float(parts[1]) if parts[1].strip().replace('.', '').isdigit() else None,
                            "occupancy": float(parts[2]) if len(parts) > 2 and parts[2].strip().replace('.', '').isdigit() else None,
                        })

            if kernels:
                info["top_kernels"] = kernels[:10]

        except Exception as e:
            logger.warning(f"Failed to parse NCU CSV: {e}")

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

        # Occupancy低
        if bottlenecks.get("occupancy") is not None:
            occupancy = bottlenecks["occupancy"]
            if occupancy < 0.5:
                suggestions.append(
                    f"Occupancy较低 ({occupancy*100:.1f}%)，建议：\n"
                    "- 增加block size（BLOCK_SIZE）\n"
                    "- 调整num_warps参数\n"
                    "- 减少每个thread的寄存器使用\n"
                    "- 优化shared memory使用"
                )

        # Warp效率低
        if bottlenecks.get("warp_efficiency") is not None:
            warp_eff = bottlenecks["warp_efficiency"]
            if warp_eff < 0.8:
                suggestions.append(
                    f"Warp效率低 ({warp_eff*100:.1f}%)，建议：\n"
                    "- 减少分支发散（branch divergence）\n"
                    "- 优化条件判断逻辑，使用masked operations\n"
                    "- 确保内存访问模式连续（coalesced access）\n"
                    "- 使用tl.where代替if-else分支"
                )

        # 内存吞吐量低
        if bottlenecks.get("memory_throughput") is not None:
            mem_throughput = bottlenecks["memory_throughput"]
            if mem_throughput < 100:  # 假设理论带宽约为900GB/s
                suggestions.append(
                    f"内存吞吐量较低 ({mem_throughput:.2f} GB/s)，建议：\n"
                    "- 使用向量化内存访问（tl.load/storage with vectorized offsets）\n"
                    "- 增加block size以提高内存合并（memory coalescing）\n"
                    "- 使用tl.multiple_of和tl.max_contiguous提示编译器\n"
                    "- 考虑使用shared memory进行数据重用"
                )

        # Stall原因分析
        if bottlenecks.get("stall_reasons"):
            stall_reasons = bottlenecks["stall_reasons"]
            main_stall = max(stall_reasons.items(), key=lambda x: x[1])
            if main_stall[1] > 30:
                stall_type = main_stall[0].lower()
                if "memory" in stall_type:
                    suggestions.append(
                        f"内存停顿 ({main_stall[1]:.1f}%)，建议：\n"
                        "- 优化内存访问模式，减少bank conflicts\n"
                        "- 使用prefetch或异步内存操作\n"
                        "- 增加内存访问的并行度"
                    )
                elif "compute" in stall_type or "execution" in stall_type:
                    suggestions.append(
                        f"计算停顿 ({main_stall[1]:.1f}%)，建议：\n"
                        "- 优化计算逻辑，减少依赖链\n"
                        "- 使用更多的并行计算\n"
                        "- 考虑使用tensor cores（如果适用）"
                    )

        # 计算吞吐量分析
        if bottlenecks.get("compute_throughput") is not None:
            compute_throughput = bottlenecks["compute_throughput"]
            # 这里需要根据具体GPU的理论峰值来判断
            suggestions.append(
                f"计算吞吐量: {compute_throughput:.2f}\n"
                "建议检查计算密度和内存访问比例"
            )

        if not suggestions:
            suggestions.append("性能表现良好，可考虑微调优化")

        return "\n\n".join(suggestions)
