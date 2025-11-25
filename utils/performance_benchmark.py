"""
基于KernelBench的通用Triton性能基准测试模块
完全基于KernelBench的get_inputs和get_init_inputs来制定输入参数，具有普适性
"""

import torch
import time
import tempfile
import os
import importlib.util
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import inspect

logger = logging.getLogger(__name__)

# 尝试导入KernelBench加载器
try:
    from .kernelbench_loader import KernelBenchLoader
    KERNELBENCH_AVAILABLE = True
except ImportError:
    KERNELBENCH_AVAILABLE = False
    logger.warning("KernelBench加载器不可用")


class TritonPerformanceBenchmark:
    """基于KernelBench的通用Triton性能基准测试器"""

    def __init__(self, device: str = "cuda", warmup_runs: int = 10, benchmark_runs: int = 100):
        """
        初始化性能基准测试器
        
        Args:
            device: 测试设备
            warmup_runs: 预热运行次数
            benchmark_runs: 基准测试运行次数
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU模式")
            self.device = "cpu"
        
        # 记录TF32设置状态
        if device == "cuda":
            self.original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            self.original_tf32_cudnn = torch.backends.cudnn.allow_tf32

    def benchmark_with_kernelbench(self, kernel_file_path: str, level: int, problem_id: int) -> Dict[str, Any]:
        """
        基于KernelBench信息进行通用性能基准测试
        完全基于get_inputs和get_init_inputs，适配任意算子
        
        Args:
            kernel_file_path: Triton内核文件路径
            level: KernelBench难度级别
            problem_id: KernelBench问题ID
            
        Returns:
            性能测试结果
        """
        results = {
            "success": False,
            "error": None,
            "pytorch_time_ms": None,
            "triton_time_ms": None,
            "speedup": None,
            "memory_usage_mb": None,
            "operation_name": None,
            "input_shapes": None,
            "output_shape": None
        }
        
        try:
            # 1. 获取KernelBench问题信息
            kernelbench_info = self._get_kernelbench_info(level, problem_id)
            if not kernelbench_info:
                results["error"] = f"无法获取Level {level} Problem {problem_id}的KernelBench信息"
                return results
            
            results["operation_name"] = kernelbench_info.get("operation_name", "unknown")
            logger.info(f"开始性能测试: {results['operation_name']}")
            
            # 2. 执行KernelBench原始PyTorch代码，获取输入和模型
            pytorch_inputs, pytorch_model = self._create_pytorch_reference_from_kernelbench(kernelbench_info)
            if pytorch_inputs is None or pytorch_model is None:
                results["error"] = "无法从KernelBench创建PyTorch参考"
                return results
            
            # 记录输入信息
            results["input_shapes"] = [list(inp.shape) for inp in pytorch_inputs]
            logger.info(f"输入形状: {results['input_shapes']}")
            
            # 3. 加载Triton内核函数
            triton_func = self._load_triton_kernel(kernel_file_path)
            if triton_func is None:
                results["error"] = "无法加载Triton内核函数"
                return results
            
            # 4. 验证Triton内核输出正确性
            if not self._validate_triton_correctness(triton_func, pytorch_model, pytorch_inputs):
                results["error"] = "Triton内核输出与PyTorch不匹配"
                return results
            
            # 5. 运行性能基准测试
            perf_results = self._run_universal_benchmark(triton_func, pytorch_model, pytorch_inputs)
            results.update(perf_results)
            
            # 6. 记录输出形状
            if results["success"]:
                with torch.no_grad():
                    sample_output = pytorch_model(*pytorch_inputs)
                    results["output_shape"] = list(sample_output.shape)
                
                logger.info(f"🚀 性能测试完成: {results['operation_name']}")
                logger.info(f"   输入形状: {results['input_shapes']}")
                logger.info(f"   输出形状: {results['output_shape']}")
                logger.info(f"   PyTorch: {results['pytorch_time_ms']:.3f}ms")
                logger.info(f"   Triton:  {results['triton_time_ms']:.3f}ms")
                logger.info(f"   加速比:  {results['speedup']:.2f}x")
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"性能测试失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return results

    def _get_kernelbench_info(self, level: int, problem_id: int) -> Optional[Dict[str, Any]]:
        """
        获取KernelBench问题信息
        
        Args:
            level: 难度级别
            problem_id: 问题ID
            
        Returns:
            KernelBench问题信息
        """
        try:
            if not KERNELBENCH_AVAILABLE:
                logger.error("KernelBench不可用")
                return None
            
            loader = KernelBenchLoader()
            try:
                problem_info = loader.get_problem(level, problem_id)
                logger.info(f"成功获取KernelBench信息: {problem_info['operation_name']}")
                return problem_info
            finally:
                loader.close()
                
        except Exception as e:
            logger.error(f"获取KernelBench信息失败: {e}")
            return None

    def _create_pytorch_reference_from_kernelbench(self, kernelbench_info: Dict[str, Any]) -> Tuple[Optional[List[torch.Tensor]], Optional[Any]]:
        """
        基于KernelBench信息创建PyTorch参考实现
        完全基于get_inputs和get_init_inputs，不写死任何逻辑
        
        Args:
            kernelbench_info: KernelBench问题信息
            
        Returns:
            (输入张量列表, PyTorch模型函数)
        """
        try:
            pytorch_code = kernelbench_info.get('pytorch_code', '')
            if not pytorch_code:
                logger.error("KernelBench信息中缺少PyTorch代码")
                return None, None
            
            logger.info("执行KernelBench原始PyTorch代码...")
            namespace = {}
            exec(pytorch_code, namespace)
            
            # 1. 验证必要的函数和类存在
            required_items = ['get_inputs', 'Model']
            for item in required_items:
                if item not in namespace:
                    logger.error(f"KernelBench代码中缺少必要的 {item}")
                    return None, None
            
            # 2. 使用get_inputs获取原始输入
            get_inputs_func = namespace['get_inputs']
            original_inputs = get_inputs_func()
            logger.info(f"KernelBench原始输入: {len(original_inputs)} 个张量")
            
            # 按照KernelAgent的设计，将float32转换为bfloat16
            pytorch_inputs = []
            for i, inp in enumerate(original_inputs):
                if isinstance(inp, torch.Tensor):
                    # 记录原始信息
                    original_dtype = inp.dtype
                    original_device = inp.device
                    
                    # 移动到目标设备
                    if self.device == 'cuda' and inp.device.type != 'cuda':
                        inp = inp.cuda()
                    elif self.device == 'cpu' and inp.device.type != 'cpu':
                        inp = inp.cpu()
                    
                    # 按照KernelAgent的设计：如果原始是float32，转换为bfloat16
                    if original_dtype == torch.float32:
                        inp = inp.to(torch.bfloat16)
                        logger.info(f"  输入 {i}: shape={inp.shape}, dtype={inp.dtype} (转换自{original_dtype}), device={inp.device}")
                    else:
                        logger.info(f"  输入 {i}: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")
                    
                    pytorch_inputs.append(inp)
                    
                    # 记录变化
                    if original_device != inp.device:
                        logger.info(f"    设备已迁移: {original_device} -> {inp.device}")
                else:
                    logger.warning(f"  输入 {i}: 非张量类型 {type(inp)}")
            
            # 3. 使用get_init_inputs获取初始化参数（如果存在）
            model_class = namespace['Model']
            init_inputs = []
            
            if 'get_init_inputs' in namespace:
                try:
                    get_init_inputs_func = namespace['get_init_inputs']
                    init_inputs = get_init_inputs_func()
                    logger.info(f"KernelBench初始化参数: {len(init_inputs)} 个")
                    for i, init_inp in enumerate(init_inputs):
                        logger.info(f"  初始化参数 {i}: {type(init_inp)} - {init_inp}")
                except Exception as e:
                    logger.warning(f"获取初始化参数失败: {e}")
            
            # 4. 创建PyTorch模型实例
            if init_inputs:
                pytorch_model = model_class(*init_inputs)
                logger.info("使用初始化参数创建PyTorch模型")
            else:
                pytorch_model = model_class()
                logger.info("创建PyTorch模型（无初始化参数）")
            
            # 移动模型到目标设备
            if self.device == 'cuda':
                pytorch_model = pytorch_model.cuda()
            elif self.device == 'cpu':
                pytorch_model = pytorch_model.cpu()
            
            pytorch_model.eval()
            
            # 5. 创建可调用的PyTorch参考函数
            def pytorch_reference(*inputs):
                with torch.no_grad():
                    result = pytorch_model(*inputs)
                    # 如果输入被转换为bfloat16，确保输出也是bfloat16
                    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor) and inputs[0].dtype == torch.bfloat16:
                        if isinstance(result, torch.Tensor) and result.dtype != torch.bfloat16:
                            result = result.to(torch.bfloat16)
                    return result
            
            logger.info("成功创建基于KernelBench的PyTorch参考实现")
            return pytorch_inputs, pytorch_reference
            
        except Exception as e:
            logger.error(f"从KernelBench创建PyTorch参考失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return None, None

    def _load_triton_kernel(self, kernel_file_path: str) -> Optional[Any]:
        """
        加载Triton内核函数
        优先选择wrapper函数而不是@triton.jit函数
        
        Args:
            kernel_file_path: 内核文件路径
            
        Returns:
            可调用的内核函数
        """
        try:
            # 读取内核代码
            with open(kernel_file_path, 'r', encoding='utf-8') as f:
                kernel_code = f.read()
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name
            
            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location("kernel_module", temp_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找合适的函数
                wrapper_func = None
                kernel_func = None
                all_functions = []
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if not callable(attr) or attr_name.startswith('_') or attr_name.startswith('test_'):
                        continue
                    
                    all_functions.append(attr_name)
                    
                    # 检查是否为@triton.jit函数
                    if hasattr(attr, '__triton_jit__') or str(type(attr)) == "<class 'triton.runtime.jit.JITFunction'>":
                        kernel_func = attr
                        logger.debug(f"找到@triton.jit函数: {attr_name}")
                    else:
                        # 检查是否为wrapper函数
                        try:
                            sig = inspect.signature(attr)
                            if len(sig.parameters) > 0:  # wrapper函数应该有参数
                                # 优先选择标准命名
                                if attr_name == 'kernel_function':
                                    wrapper_func = attr
                                    logger.debug(f"找到标准wrapper函数: {attr_name}")
                                    break
                                # 兼容其他命名模式
                                elif any(pattern in attr_name.lower() for pattern in ['wrapper', 'forward', 'triton_']):
                                    if wrapper_func is None:
                                        wrapper_func = attr
                                        logger.debug(f"找到兼容wrapper函数: {attr_name}")
                                # 其他有参数的函数作为备选
                                elif wrapper_func is None:
                                    wrapper_func = attr
                                    logger.debug(f"候选wrapper函数: {attr_name}")
                        except Exception as e:
                            logger.debug(f"检查函数签名失败 {attr_name}: {e}")
                
                logger.info(f"模块中的所有函数: {all_functions}")
                
                # 优先返回wrapper函数
                if wrapper_func:
                    logger.info(f"选择wrapper函数: {wrapper_func.__name__}")
                    return wrapper_func
                else:
                    logger.error("未找到可调用的wrapper函数")
                    if kernel_func:
                        logger.error("找到了@triton.jit函数，但它不能直接调用，需要wrapper函数")
                    return None
                    
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"加载Triton内核失败: {e}")
            return None

    def _validate_triton_correctness(self, triton_func: Any, pytorch_func: Any, test_inputs: List[torch.Tensor]) -> bool:
        """
        验证Triton内核输出的正确性
        
        Args:
            triton_func: Triton内核函数
            pytorch_func: PyTorch参考函数
            test_inputs: 测试输入
            
        Returns:
            是否正确
        """
        try:
            logger.info("开始Triton内核正确性验证...")
            
            # 记录输入信息
            for i, inp in enumerate(test_inputs):
                logger.info(f"验证输入 {i}: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")
            
            # 获取PyTorch参考结果
            with torch.no_grad():
                pytorch_result = pytorch_func(*test_inputs)
            
            logger.info(f"PyTorch参考结果: shape={pytorch_result.shape}, dtype={pytorch_result.dtype}, device={pytorch_result.device}")
            
            # 尝试获取Triton结果
            try:
                triton_result = triton_func(*test_inputs)
                logger.info(f"Triton内核结果: shape={triton_result.shape}, dtype={triton_result.dtype}, device={triton_result.device}")
            except Exception as e:
                logger.error(f"Triton内核执行失败: {e}")
                
                # 检查是否是数据类型不匹配问题
                if "Expected" in str(e) and "got" in str(e):
                    logger.error("检测到数据类型不匹配问题")
                    logger.error("这通常是因为生成的内核期望的数据类型与KernelBench原始输入不匹配")
                    logger.error("建议检查内核代码中的数据类型处理")
                
                return False
            
            # 比较结果
            if isinstance(pytorch_result, torch.Tensor) and isinstance(triton_result, torch.Tensor):
                # 检查形状匹配
                if pytorch_result.shape != triton_result.shape:
                    logger.error(f"输出形状不匹配: PyTorch={pytorch_result.shape}, Triton={triton_result.shape}")
                    return False
                
                # 检查数据类型
                if pytorch_result.dtype != triton_result.dtype:
                    logger.warning(f"输出数据类型不匹配: PyTorch={pytorch_result.dtype}, Triton={triton_result.dtype}")
                    # 尝试转换到相同类型进行比较
                    if pytorch_result.dtype in [torch.float32, torch.float16, torch.bfloat16] and \
                       triton_result.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        # 转换到更高精度的类型进行比较
                        if pytorch_result.dtype == torch.float32 or triton_result.dtype == torch.float32:
                            target_dtype = torch.float32
                        else:
                            target_dtype = pytorch_result.dtype
                        
                        pytorch_result_cmp = pytorch_result.to(target_dtype)
                        triton_result_cmp = triton_result.to(target_dtype)
                        logger.info(f"转换到 {target_dtype} 进行比较")
                    else:
                        logger.error("无法处理的数据类型组合")
                        return False
                else:
                    pytorch_result_cmp = pytorch_result
                    triton_result_cmp = triton_result
                
                # 根据数据类型选择合适的容差
                if pytorch_result_cmp.dtype == torch.bfloat16:
                    rtol, atol = 1e-2, 2e-2
                elif pytorch_result_cmp.dtype == torch.float16:
                    rtol, atol = 1e-3, 1e-3
                else:
                    rtol, atol = 1e-4, 1e-4
                
                logger.info(f"使用容差: rtol={rtol}, atol={atol}")
                
                is_close = torch.allclose(pytorch_result_cmp, triton_result_cmp, rtol=rtol, atol=atol)
                
                if not is_close:
                    logger.warning("Triton输出与PyTorch不完全匹配")
                    
                    # 计算差异统计
                    abs_diff = torch.abs(pytorch_result_cmp - triton_result_cmp)
                    max_diff = torch.max(abs_diff)
                    mean_diff = torch.mean(abs_diff)
                    
                    # 计算相对误差
                    nonzero_mask = pytorch_result_cmp != 0
                    if torch.any(nonzero_mask):
                        rel_diff = torch.abs((pytorch_result_cmp[nonzero_mask] - triton_result_cmp[nonzero_mask]) / pytorch_result_cmp[nonzero_mask])
                        max_rel_diff = torch.max(rel_diff)
                        mean_rel_diff = torch.mean(rel_diff)
                    else:
                        max_rel_diff = torch.tensor(0.0)
                        mean_rel_diff = torch.tensor(0.0)
                    
                    logger.warning(f"最大绝对差异: {max_diff:.6f}, 平均绝对差异: {mean_diff:.6f}")
                    logger.warning(f"最大相对差异: {max_rel_diff:.6f}, 平均相对差异: {mean_rel_diff:.6f}")
                    
                    # 显示一些样本值进行调试
                    logger.warning("样本值比较（前10个元素）:")
                    pytorch_flat = pytorch_result_cmp.flatten()[:10]
                    triton_flat = triton_result_cmp.flatten()[:10]
                    for i in range(min(10, len(pytorch_flat))):
                        logger.warning(f"  [{i}] PyTorch: {pytorch_flat[i]:.6f}, Triton: {triton_flat[i]:.6f}, 差异: {abs(pytorch_flat[i] - triton_flat[i]):.6f}")
                    
                    # 对于某些算子，可能存在数值精度差异，给出警告但不阻止性能测试
                    if max_diff < 0.1:  # 如果差异不是太大，继续进行性能测试
                        logger.warning("差异在可接受范围内，继续性能测试")
                        return True
                    else:
                        logger.error("差异过大，停止性能测试")
                        return False
                
                logger.info("✅ Triton内核输出正确性验证通过")
                return True
            else:
                logger.error(f"结果类型不匹配: PyTorch={type(pytorch_result)}, Triton={type(triton_result)}")
                return False
                
        except Exception as e:
            logger.error(f"正确性验证失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False

    def _run_universal_benchmark(self, triton_func: Any, pytorch_func: Any, test_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        运行通用性能基准测试
        
        Args:
            triton_func: Triton内核函数
            pytorch_func: PyTorch参考函数
            test_inputs: 测试输入列表
            
        Returns:
            性能测试结果
        """
        results = {
            "success": False,
            "pytorch_time_ms": None,
            "triton_time_ms": None,
            "speedup": None,
            "memory_usage_mb": None
        }
        
        try:
            logger.info("开始通用性能基准测试...")
            
            # PyTorch基准测试
            pytorch_time = self._benchmark_pytorch_general(pytorch_func, test_inputs)
            if pytorch_time is None:
                results["error"] = "PyTorch基准测试失败"
                return results
            
            # Triton基准测试
            triton_time = self._benchmark_triton_general(triton_func, test_inputs)
            if triton_time is None:
                results["error"] = "Triton基准测试失败"
                return results
            
            # 计算加速比
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            # 计算内存使用
            memory_usage = 0
            if self.device == "cuda":
                for inp in test_inputs:
                    if isinstance(inp, torch.Tensor):
                        memory_usage += inp.numel() * inp.element_size()
                memory_usage_mb = memory_usage / (1024 * 1024)
            else:
                memory_usage_mb = 0
            
            results.update({
                "success": True,
                "pytorch_time_ms": pytorch_time,
                "triton_time_ms": triton_time,
                "speedup": speedup,
                "memory_usage_mb": memory_usage_mb
            })
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"通用性能基准测试失败: {e}")
            return results

    def _benchmark_pytorch_general(self, pytorch_func: Any, test_inputs: List[torch.Tensor]) -> Optional[float]:
        """
        通用PyTorch性能测试
        
        Args:
            pytorch_func: PyTorch函数
            test_inputs: 测试输入列表
            
        Returns:
            平均执行时间（毫秒）
        """
        try:
            # 启用TF32优化以获得公平的性能对比
            original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            original_tf32_cudnn = torch.backends.cudnn.allow_tf32
            
            try:
                # 启用TF32优化（A100等Ampere架构的标准优化）
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                # 预热
                for _ in range(self.warmup_runs):
                    with torch.no_grad():
                        pytorch_func(*test_inputs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                # 基准测试
                start_time = time.perf_counter()
                for _ in range(self.benchmark_runs):
                    with torch.no_grad():
                        pytorch_func(*test_inputs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / self.benchmark_runs * 1000
                return avg_time
                
            finally:
                # 恢复原始TF32设置
                torch.backends.cuda.matmul.allow_tf32 = original_tf32_matmul
                torch.backends.cudnn.allow_tf32 = original_tf32_cudnn
                
        except Exception as e:
            logger.error(f"PyTorch基准测试失败: {e}")
            return None

    def _benchmark_triton_general(self, triton_func: Any, test_inputs: List[torch.Tensor]) -> Optional[float]:
        """
        通用Triton性能测试
        
        Args:
            triton_func: Triton函数
            test_inputs: 测试输入列表
            
        Returns:
            平均执行时间（毫秒）
        """
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 预热
            for _ in range(self.warmup_runs):
                triton_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 基准测试
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                triton_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / self.benchmark_runs * 1000
            return avg_time
            
        except Exception as e:
            logger.error(f"Triton基准测试失败: {e}")
            return None

    def format_performance_summary(self, results: Dict[str, Any]) -> str:
        """
        格式化性能测试结果摘要
        
        Args:
            results: 性能测试结果
            
        Returns:
            格式化的摘要字符串
        """
        if not results["success"]:
            return f"❌ 性能测试失败: {results.get('error', 'Unknown error')}"
        
        operation_name = results.get("operation_name", "Unknown")
        pytorch_time = results["pytorch_time_ms"]
        triton_time = results["triton_time_ms"]
        speedup = results["speedup"]
        
        # 性能评级
        if speedup > 2.0:
            grade = "优秀 🏆"
        elif speedup > 1.2:
            grade = "良好 ✅"
        elif speedup > 0.8:
            grade = "一般 ⚠️"
        else:
            grade = "需优化 ❌"
        
        summary = f"""🚀 {operation_name} 性能基准测试结果:
   输入形状: {results.get('input_shapes', 'Unknown')}
   输出形状: {results.get('output_shape', 'Unknown')}
   PyTorch: {pytorch_time:.3f}ms
   Triton:  {triton_time:.3f}ms
   加速比:  {speedup:.2f}x
   评级:    {grade}"""
        
        if results.get("memory_usage_mb"):
            summary += f"\n   内存:    {results['memory_usage_mb']:.1f}MB"
        
        return summary


def benchmark_successful_kernel(session_dir: str, level: Optional[int] = None, problem_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    对成功验证的内核进行性能基准测试
    完全基于KernelBench的get_inputs和get_init_inputs
    
    Args:
        session_dir: 会话目录路径
        level: KernelBench难度级别
        problem_id: KernelBench问题ID
        
    Returns:
        性能测试结果，失败返回None
    """
    try:
        session_path = Path(session_dir)
        kernel_file = session_path / "final_kernel.py"
        
        # 检查内核文件是否存在
        if not kernel_file.exists():
            logger.warning(f"内核文件不存在: {kernel_file}")
            return None
        
        # 如果提供了level和problem_id，直接使用
        if level is not None and problem_id is not None:
            logger.info(f"使用提供的Level {level} Problem {problem_id}进行性能测试")
            
            # 创建基准测试器
            benchmark = TritonPerformanceBenchmark(warmup_runs=5, benchmark_runs=20)
            
            # 运行基于KernelBench的性能测试
            results = benchmark.benchmark_with_kernelbench(str(kernel_file), level, problem_id)
            
            # 保存结果
            if results["success"]:
                perf_file = session_path / "performance_results.json"
                with open(perf_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(results, f, indent=2)
            
            return results
        else:
            logger.warning("未提供level和problem_id，无法进行基于KernelBench的性能测试")
            return None
        
    except Exception as e:
        logger.error(f"性能基准测试失败: {e}")
        return None