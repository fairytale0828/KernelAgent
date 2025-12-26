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
Enhanced pipeline with Mirage-style algorithmic optimization.

Usage:
  python -m Fuser.pipeline_enhanced \
    --problem /abs/path/to/kernelbench_problem.py \
    --extract-model deepseek-chat \
    --dispatch-model deepseek-chat \
    --compose-model deepseek-chat \
    --workers 4 --max-iters 5 \
    --verify \
    --enable-algorithmic-rewrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .subgraph_extractor import extract_subgraphs_to_json
from .dispatch_kernel_agent import run as dispatch_run
from .compose_end_to_end import compose
from .algorithm_analyzer import AlgorithmPatternDetector
from .algorithm_rewriter import AlgorithmRewriter


def run_pipeline_with_algorithmic_optimization(
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
    enable_algorithmic_rewrite: bool = True,
    rewrite_benefit_threshold: float = 0.6,
) -> dict:
    """Enhanced pipeline with algorithmic optimization."""
    
    # Select default KernelAgent model if not provided
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
    
    print(f"🚀 Starting enhanced Fuser pipeline with algorithmic optimization")
    print(f"   Problem: {problem_path.name}")
    print(f"   Models: extract={extract_model}, dispatch={dispatch_model}, compose={compose_model}")
    print(f"   Algorithmic rewrite: {'enabled' if enable_algorithmic_rewrite else 'disabled'}")
    print()
    
    # Step 1: Extract subgraphs
    print("📊 Step 1: Extracting subgraphs...")
    run_dir, subgraphs_path = extract_subgraphs_to_json(
        problem_path=problem_path,
        model_name=extract_model,
        workers=workers,
        max_iters=max_iters,
        llm_timeout_s=llm_timeout_s,
        run_timeout_s=run_timeout_s,
    )
    print(f"   ✓ Subgraphs extracted to: {subgraphs_path}")
    
    # Step 1.5: Algorithmic Rewriting (NEW!)
    if enable_algorithmic_rewrite:
        print()
        print("🔬 Step 1.5: Analyzing for algorithmic optimization opportunities...")
        
        with open(subgraphs_path, 'r') as f:
            original_subgraphs = json.load(f)
        
        print(f"   Original subgraphs: {len(original_subgraphs)}")
        
        # Detect algorithmic patterns
        detector = AlgorithmPatternDetector()
        patterns = detector.analyze(original_subgraphs)
        
        # Filter by benefit threshold
        high_value_patterns = [
            p for p in patterns 
            if p.estimated_benefit >= rewrite_benefit_threshold
        ]
        
        if high_value_patterns:
            print(f"   🔍 Detected {len(high_value_patterns)} high-value optimization opportunities:")
            for p in high_value_patterns:
                print(f"      - {p.pattern_type}: {p.subgraph_ids}")
                print(f"        Benefit: {p.estimated_benefit:.2f}, Strategy: {p.rewrite_strategy}")
            
            # Apply rewrites
            print(f"   🔧 Applying algorithmic rewrites...")
            rewriter = AlgorithmRewriter()
            rewritten_subgraphs, metadata = rewriter.apply_rewrites(
                original_subgraphs, 
                high_value_patterns
            )
            
            # Save rewritten subgraphs
            rewritten_path = Path(run_dir) / "subgraphs_rewritten.json"
            with open(rewritten_path, 'w') as f:
                json.dump(rewritten_subgraphs, f, indent=2)
            
            # Save rewrite metadata
            metadata_path = Path(run_dir) / "algorithm_rewrite_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Use rewritten subgraphs
            subgraphs_path = rewritten_path
            
            print(f"   ✓ Applied {len(metadata['applied_patterns'])} algorithmic rewrites")
            print(f"   ✓ Eliminated {len(set(metadata['eliminated_subgraphs']))} intermediate subgraphs")
            print(f"   ✓ Created {len(metadata['new_subgraphs'])} fused subgraphs")
            print(f"   ✓ Final subgraph count: {len(rewritten_subgraphs)}")
        else:
            print(f"   ℹ️  No high-value optimization opportunities found (threshold: {rewrite_benefit_threshold})")
    
    # Step 2: Dispatch to KernelAgent
    print()
    print("⚙️  Step 2: Dispatching subgraphs to KernelAgent...")
    out_dir = Path(run_dir) / "kernels_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve dispatch concurrency
    jobs_val: int
    if isinstance(dispatch_jobs, str) and dispatch_jobs.strip().lower() == "auto":
        try:
            with Path(subgraphs_path).open("r", encoding="utf-8") as f:
                _items = json.load(f)
            jobs_val = max(1, int(len(_items))) if isinstance(_items, list) else 1
        except Exception:
            jobs_val = 1
    else:
        try:
            jobs_val = max(1, int(dispatch_jobs))
        except Exception:
            jobs_val = 1
    
    print(f"   Parallel jobs: {jobs_val}")
    
    summary_path = dispatch_run(
        subgraphs_path=Path(subgraphs_path),
        out_dir=out_dir,
        agent_model=dispatch_model,
        jobs=jobs_val,
    )
    print(f"   ✓ Kernels generated: {summary_path}")
    
    # Step 3: Compose end-to-end
    print()
    print("🔗 Step 3: Composing end-to-end kernel...")
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
    
    print(f"   ✓ Composition complete")
    if verify:
        if comp_res.get("verify_passed"):
            print(f"   ✅ Verification PASSED")
        else:
            print(f"   ❌ Verification FAILED: {comp_res.get('verify_reason', 'unknown')}")
    
    # Final summary
    print()
    print("=" * 60)
    print("📋 Pipeline Summary:")
    print(f"   Run directory: {run_dir}")
    print(f"   Subgraphs: {subgraphs_path}")
    print(f"   Kernels: {summary_path}")
    print(f"   Composed kernel: {comp_res.get('composed_path', 'N/A')}")
    if enable_algorithmic_rewrite and high_value_patterns:
        print(f"   Algorithmic optimizations applied: {len(metadata['applied_patterns'])}")
    print("=" * 60)
    
    return {
        "run_dir": str(run_dir),
        "subgraphs": str(subgraphs_path),
        "kernels_summary": str(summary_path),
        "composition": comp_res,
        "algorithmic_rewrite": {
            "enabled": enable_algorithmic_rewrite,
            "patterns_detected": len(patterns) if enable_algorithmic_rewrite else 0,
            "patterns_applied": len(metadata['applied_patterns']) if enable_algorithmic_rewrite and high_value_patterns else 0,
        } if enable_algorithmic_rewrite else None
    }


def main(argv: Optional[list[str]] = None) -> int:
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    
    p = argparse.ArgumentParser(
        description="Enhanced pipeline with algorithmic optimization: extract → rewrite → dispatch → compose"
    )
    p.add_argument("--problem", required=True, help="Absolute path to the problem file")
    p.add_argument("--extract-model", default="deepseek-chat", help="Model for subgraph extraction")
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
    p.add_argument("--compose-model", default="deepseek-chat", help="Model for composition")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    p.add_argument("--max-iters", type=int, default=5, help="Extractor iteration budget")
    p.add_argument("--llm-timeout-s", type=int, default=1200, help="LLM timeout in seconds")
    p.add_argument("--run-timeout-s", type=int, default=1200, help="Run timeout in seconds")
    p.add_argument("--out-root", default=None, help="Output root directory")
    p.add_argument("--verify", action="store_true", help="Verify composed kernel")
    p.add_argument("--compose-max-iters", type=int, default=5, help="Composition max iterations")
    p.add_argument(
        "--enable-algorithmic-rewrite",
        action="store_true",
        default=True,
        help="Enable Mirage-style algorithmic rewriting (default: True)"
    )
    p.add_argument(
        "--no-algorithmic-rewrite",
        action="store_false",
        dest="enable_algorithmic_rewrite",
        help="Disable algorithmic rewriting"
    )
    p.add_argument(
        "--rewrite-benefit-threshold",
        type=float,
        default=0.6,
        help="Minimum benefit threshold for applying rewrites (0-1, default: 0.6)"
    )
    args = p.parse_args(argv)
    
    problem_path = Path(args.problem).resolve()
    if not problem_path.is_file():
        print(f"❌ Problem file not found: {problem_path}", file=sys.stderr)
        return 2
    
    try:
        res = run_pipeline_with_algorithmic_optimization(
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
            enable_algorithmic_rewrite=args.enable_algorithmic_rewrite,
            rewrite_benefit_threshold=args.rewrite_benefit_threshold,
        )
        print()
        print("📄 Full result JSON:")
        print(json.dumps(res, indent=2))
        return 0
    except SystemExit as e:
        try:
            return int(e.code) if e.code is not None else 1
        except Exception:
            return 1
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
