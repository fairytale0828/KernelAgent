#!/usr/bin/env python3
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
"""
One-shot pipeline runner: extract → [algebraic rewrite → aggregate] → dispatch → compose.

Pipeline with --enable-algebraic-rewrite:
1. Extract: 细粒度子图提取
2. E-Graph: 代数优化，发现优化机会 (online_softmax, late_scaling 等)
3. Aggregate: 识别融合模式，聚合子图，保留优化提示
4. Dispatch: 根据聚合子图 + 优化提示生成融合 kernel
5. Compose: 组装最终代码

Usage:
  python -m Fuser.pipeline \
    --problem /abs/path/to/kernelbench_problem.py \
    --extract-model gpt-5 \
    --dispatch-model o4-mini \
    [--dispatch-jobs 1] \
    --compose-model o4-mini \
    --workers 4 --max-iters 5 \
    --llm-timeout-s 1200 --run-timeout-s 1200 \
    --out-root ./.fuse \
    [--verify] [--compose-max-iters 5] \
    [--enable-algebraic-rewrite] [--rewrite-top-k 5]

Writes all artifacts into the run directory created by the extractor. The final
composed kernel and composition summary live under <run_dir>/compose_out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .subgraph_extractor import extract_subgraphs_to_json
from .dispatch_kernel_agent import run as dispatch_run
from .compose_end_to_end import compose


def run_pipeline(
    problem_path: Path,
    extract_model: str,
    dispatch_model: Optional[str],
    compose_model: str,
    dispatch_jobs: int | str,
    workers: int,
    max_iters: int,
    llm_timeout_s: int,
    run_timeout_s: int,
    out_root: Optional[Path] = None,
    verify: bool = True,
    compose_max_iters: int = 5,
    # 代数重写参数
    enable_algebraic_rewrite: bool = False,
    rewrite_top_k: int = 5,
    rewrite_cost_model: str = "io_aware",
    rewrite_rule_set: str = "default",
    rewrite_time_budget_ms: int = 10000,
) -> dict:
    """
    运行完整的 Fuser pipeline

    Args:
        problem_path: 问题文件路径
        extract_model: 提取阶段使用的模型
        dispatch_model: dispatch 阶段使用的模型
        compose_model: compose 阶段使用的模型
        dispatch_jobs: 并行 dispatch 任务数
        workers: worker 数量
        max_iters: 最大迭代次数
        llm_timeout_s: LLM 超时时间
        run_timeout_s: 运行超时时间
        out_root: 输出根目录
        verify: 是否验证
        compose_max_iters: compose 最大迭代次数
        enable_algebraic_rewrite: 是否启用代数重写
        rewrite_top_k: 每个子图生成的候选数量
        rewrite_cost_model: 代价模型
        rewrite_rule_set: 规则集
        rewrite_time_budget_ms: 重写时间预算

    Returns:
        pipeline 结果字典
    """
    # Select default KernelAgent model if not provided: prefer GPT-5 for Level 2/3
    if dispatch_model is None:
        pp = str(problem_path)
        is_l2 = (
            ("/KernelBench/KernelBench/level2/" in pp)
            or ("/KernelBench/level2/" in pp)
            or ("level2/" in pp)
        )
        is_l3 = (
            ("/KernelBench/KernelBench/level3/" in pp)
            or ("/KernelBench/level3/" in pp)
            or ("level3/" in pp)
        )
        if is_l2 or is_l3:
            dispatch_model = "gpt-5"
        else:
            dispatch_model = "o4-mini"

    # Step 1: extract
    print("=" * 60)
    print("Step 1: Extracting subgraphs...")
    print("=" * 60)
    run_dir, subgraphs_path = extract_subgraphs_to_json(
        problem_path=problem_path,
        model_name=extract_model,
        workers=workers,
        max_iters=max_iters,
        llm_timeout_s=llm_timeout_s,
        run_timeout_s=run_timeout_s,
    )
    print(f"✓ Extracted subgraphs to: {subgraphs_path}")

    # Step 1.5: Algebraic Rewrite (optional)
    rewrite_stats = None
    if enable_algebraic_rewrite:
        print("\n" + "=" * 60)
        print("Step 1.5: Algebraic Rewriting (E-Graph)...")
        print("=" * 60)

        try:
            # 使用新的通用优化器
            from .egraph import AlgebraicOptimizer, OptimizerConfig, get_rule_set, list_rule_sets

            print(f"  Available rule sets: {list_rule_sets()}")
            print(f"  Using rule set: {rewrite_rule_set}")

            optimizer_config = OptimizerConfig(
                rule_set=rewrite_rule_set,
                top_k=rewrite_top_k,
                cost_model=rewrite_cost_model,
                time_budget_ms=rewrite_time_budget_ms,
                max_iterations=30,
                include_original=True,
                include_proof=True,
                deduplicate=True,
                expand_ops=True,
                expand_norms=True,
            )

            optimizer = AlgebraicOptimizer(optimizer_config)

            # 输出到新文件
            transformed_path = Path(run_dir) / "subgraphs_transformed.json"
            stats = optimizer.optimize_json(subgraphs_path, transformed_path)

            print(f"✓ Algebraic rewrite completed:")
            print(f"  Original subgraphs: {stats['original_count']}")
            print(f"  Generated variants: {stats['total_variants']}")
            print(f"  Total output: {stats['output_count']}")
            print(f"  Processing time: {stats['processing_time_ms']}ms")

            if stats.get("by_rule"):
                print(f"  Rules applied:")
                for rule_name, count in stats["by_rule"].items():
                    print(f"    - {rule_name}: {count}")

            # 使用重写后的文件
            subgraphs_path = transformed_path
            rewrite_stats = stats

        except ImportError as e:
            print(f"⚠ Algebraic rewrite not available: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing without rewrite...")
        except Exception as e:
            print(f"⚠ Algebraic rewrite failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with original subgraphs...")

    # Step 1.6: Aggregate (识别融合模式，聚合子图)
    aggregate_stats = None
    if enable_algebraic_rewrite:
        print("\n" + "=" * 60)
        print("Step 1.6: Aggregating subgraphs (pattern recognition)...")
        print("=" * 60)

        try:
            from .egraph import SubgraphAggregator

            aggregator = SubgraphAggregator()

            # 读取当前子图
            with open(subgraphs_path, 'r', encoding='utf-8') as f:
                current_subgraphs = json.load(f)

            # 执行聚合
            aggregated_subgraphs, agg_stats = aggregator.aggregate(
                current_subgraphs)

            # 输出到新文件
            aggregated_path = Path(run_dir) / "subgraphs_aggregated.json"
            with open(aggregated_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_subgraphs, f, indent=2)

            print(f"✓ Aggregation completed:")
            print(f"  Input subgraphs: {agg_stats['input_count']}")
            print(f"  Output subgraphs: {agg_stats['output_count']}")
            print(f"  Fusions applied: {agg_stats['fusions_applied']}")
            print(
                f"  Optimization hints preserved: {agg_stats.get('optimization_hints_preserved', 0)}")

            if agg_stats.get("patterns_matched"):
                print(f"  Fusion patterns matched:")
                for pattern_name, count in agg_stats["patterns_matched"].items():
                    print(f"    - {pattern_name}: {count}")

            # 显示聚合后的子图类型
            print(f"  Aggregated subgraph types:")
            type_counts = {}
            for sg in aggregated_subgraphs:
                sg_type = sg.get("type", "unknown")
                type_counts[sg_type] = type_counts.get(sg_type, 0) + 1
            for sg_type, count in type_counts.items():
                hints = ""
                # 检查是否有 codegen_hints
                for sg in aggregated_subgraphs:
                    if sg.get("type") == sg_type and sg.get("codegen_hints"):
                        hint_keys = list(sg["codegen_hints"].keys())[:3]
                        hints = f" (hints: {hint_keys})"
                        break
                print(f"    - {sg_type}: {count}{hints}")

            # 使用聚合后的文件
            subgraphs_path = aggregated_path
            aggregate_stats = agg_stats

        except ImportError as e:
            print(f"⚠ Aggregation not available: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing without aggregation...")
        except Exception as e:
            print(f"⚠ Aggregation failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with previous subgraphs...")

    # Step 2: dispatch to KernelAgent
    print("\n" + "=" * 60)
    print("Step 2: Dispatching to KernelAgent...")
    print("=" * 60)

    out_dir = Path(run_dir) / "kernels_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dispatch concurrency (support "auto")
    jobs_val: int
    if isinstance(dispatch_jobs, str) and dispatch_jobs.strip().lower() == "auto":
        try:
            with Path(subgraphs_path).open("r", encoding="utf-8") as f:
                _items = json.load(f)
            jobs_val = max(1, int(len(_items))) if isinstance(
                _items, list) else 1
        except Exception:
            jobs_val = 1
    else:
        try:
            jobs_val = max(1, int(dispatch_jobs))
        except Exception:
            jobs_val = 1

    summary_path = dispatch_run(
        subgraphs_path=Path(subgraphs_path),
        out_dir=out_dir,
        agent_model=dispatch_model,
        jobs=jobs_val,
    )
    print(f"✓ Dispatch completed: {summary_path}")

    # Step 3: compose end-to-end
    print("\n" + "=" * 60)
    print("Step 3: Composing end-to-end kernel...")
    print("=" * 60)

    compose_out = Path(run_dir) / "compose_out"
    compose_out.mkdir(parents=True, exist_ok=True)
    comp_res = compose(
        problem_path=problem_path,
        subgraphs_path=Path(subgraphs_path),
        kernels_summary_path=summary_path,
        out_dir=compose_out,
        model_name=compose_model,
        verify=verify,
        max_iters=compose_max_iters,
    )
    print(f"✓ Composition completed")

    result = {
        "run_dir": str(run_dir),
        "subgraphs": str(subgraphs_path),
        "kernels_summary": str(summary_path),
        "composition": comp_res,
    }

    if rewrite_stats:
        result["algebraic_rewrite"] = rewrite_stats

    if aggregate_stats:
        result["aggregate"] = aggregate_stats

    return result


def main(argv: Optional[list[str]] = None) -> int:
    # Load .env if present for OPENAI_API_KEY, proxies, etc.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="End-to-end pipeline: extract → [algebraic rewrite] → dispatch → compose"
    )

    # 基本参数
    p.add_argument("--problem", required=True,
                   help="Absolute path to the problem file")
    p.add_argument("--extract-model", default="gpt-5")
    p.add_argument(
        "--dispatch-model",
        default=None,
        help="KernelAgent model (default: gpt-5 for level2 problems, else o4-mini)",
    )
    p.add_argument(
        "--dispatch-jobs",
        type=str,
        default="2",
        help="Max concurrent KernelAgent subgraph tasks (default: 2); use 'auto' to match subgraph count",
    )
    p.add_argument("--compose-model", default="o4-mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=5,
                   help="Extractor iter budget")
    p.add_argument("--llm-timeout-s", type=int, default=1200)
    p.add_argument("--run-timeout-s", type=int, default=1200)
    p.add_argument("--out-root", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--compose-max-iters", type=int, default=5)

    # 代数重写参数
    p.add_argument(
        "--enable-algebraic-rewrite",
        action="store_true",
        help="Enable E-Graph based algebraic rewriting (generates multiple subgraph variants)"
    )
    p.add_argument(
        "--rewrite-top-k",
        type=int,
        default=5,
        help="Number of candidate variants per subgraph (default: 5)"
    )
    p.add_argument(
        "--rewrite-cost-model",
        choices=["io_aware", "flops", "balanced"],
        default="io_aware",
        help="Cost model for algebraic rewrite (default: io_aware)"
    )
    p.add_argument(
        "--rewrite-rule-set",
        choices=["default", "full", "minimal", "mlp", "attention"],
        default="default",
        help="Rule set for algebraic rewrite (default: G1-G4 enabled)"
    )
    p.add_argument(
        "--rewrite-time-budget-ms",
        type=int,
        default=10000,
        help="Time budget for algebraic rewrite in milliseconds (default: 10000)"
    )

    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    if not problem_path.is_file():
        print(f"problem not found: {problem_path}")
        return 2

    try:
        res = run_pipeline(
            problem_path=problem_path,
            extract_model=args.extract_model,
            dispatch_model=args.dispatch_model,
            compose_model=args.compose_model,
            dispatch_jobs=args.dispatch_jobs,
            workers=args.workers,
            max_iters=args.max_iters,
            llm_timeout_s=args.llm_timeout_s,
            run_timeout_s=args.run_timeout_s,
            out_root=Path(args.out_root) if args.out_root else None,
            verify=args.verify,
            compose_max_iters=args.compose_max_iters,
            # 代数重写参数
            enable_algebraic_rewrite=args.enable_algebraic_rewrite,
            rewrite_top_k=args.rewrite_top_k,
            rewrite_cost_model=args.rewrite_cost_model,
            rewrite_rule_set=args.rewrite_rule_set,
            rewrite_time_budget_ms=args.rewrite_time_budget_ms,
        )
        print("\n" + "=" * 60)
        print("Pipeline Result:")
        print("=" * 60)
        print(json.dumps(res, indent=2))
        return 0
    except SystemExit as e:
        try:
            return int(e.code) if e.code is not None else 1
        except Exception:
            try:
                import sys as _sys

                print(str(e), file=_sys.stderr)
            except Exception:
                pass
            return 1
    except Exception as e:
        print(f"pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
