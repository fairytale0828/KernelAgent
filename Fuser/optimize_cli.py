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
CLI for LLM-First optimization discovery.

Usage:
  python -m Fuser.optimize_cli --problem /path/to/problem.py \\
      --model gpt-5 \\
      --workers 4 \\
      --plans-per-worker 3 \\
      --refinements-per-plan 3 \\
      --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .config import OrchestratorConfig, new_run_id
from .optimizing_orchestrator import OptimizingOrchestrator
from .paths import ensure_abs_regular_file, make_run_dirs, PathSafetyError


def run_optimization(
    problem_path: Path,
    model: str,
    workers: int = 4,
    plans_per_worker: int = 3,
    refinements_per_plan: int = 3,
    llm_timeout_s: int = 1200,
    run_timeout_s: int = 1200,
    stream_mode: str = "all",
    verify: bool = False,
    out_root: Optional[Path] = None,
) -> dict:
    """
    Run LLM-First optimization discovery.
    
    Args:
        problem_path: Path to problem file
        model: Model name for LLM calls
        workers: Number of parallel workers
        plans_per_worker: Number of plans each worker tries
        refinements_per_plan: Max refinements per plan
        llm_timeout_s: Timeout for LLM calls
        run_timeout_s: Timeout for code execution
        stream_mode: Streaming mode (all|winner|none)
        verify: Whether to verify final result
        out_root: Optional output root directory
        
    Returns:
        Dictionary with results
    """
    # Create configuration
    cfg = OrchestratorConfig(
        problem_path=problem_path,
        model=model,
        workers=workers,
        max_iters=refinements_per_plan,  # Used as max refinements per plan
        llm_timeout_s=llm_timeout_s,
        run_timeout_s=run_timeout_s,
        stream_mode=stream_mode,
        store_responses=False,
        isolated=False,
        deny_network=False,
        enable_reasoning_extras=True,
    )
    
    # Create run directories
    run_id = new_run_id()
    if out_root:
        base_dir = out_root
    else:
        # Use .fuse directory (consistent with Fuser pipeline)
        # If you want to use triton_kernel_logs instead, uncomment the next line:
        # base_dir = Path.cwd() / "triton_kernel_logs"
        base_dir = Path.cwd() / ".fuse"
    base_dir.mkdir(exist_ok=True, parents=True)
    
    dirs = make_run_dirs(base_dir, run_id)
    
    # Save configuration
    (dirs["orchestrator"] / "config.json").write_text(
        cfg.to_json(), encoding="utf-8"
    )
    
    print(f"Run directory: {dirs['run_dir']}")
    print("Initializing orchestrator...")
    sys.stdout.flush()
    
    # Run orchestrator
    orchestrator = OptimizingOrchestrator(
        cfg=cfg,
        run_dir=dirs["run_dir"],
        workers_dir=dirs["workers"],
        orchestrator_dir=dirs["orchestrator"],
        num_plans_per_worker=plans_per_worker,
        max_refinements_per_plan=refinements_per_plan,
    )
    
    print("Starting worker processes...")
    sys.stdout.flush()
    
    summary = orchestrator.run()
    
    print("Orchestrator completed.")
    sys.stdout.flush()
    
    result = {
        "run_id": run_id,
        "run_dir": str(dirs["run_dir"]),
        "success": summary.winner_worker_id is not None,
        "winner_worker_id": summary.winner_worker_id,
        "artifact_path": summary.artifact_path,
        "reason": summary.reason,
    }
    
    # Optional verification
    if verify and summary.artifact_path:
        print("\n=== Verification ===")
        print("Extracting and verifying final result...")
        
        import tarfile
        from .runner import run_candidate
        
        # Extract code
        verify_dir = dirs["run_dir"] / "verification"
        verify_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(summary.artifact_path, "r:gz") as tf:
            tf.extractall(verify_dir)
        
        code_path = verify_dir / "code.py"
        if code_path.exists():
            # Run verification
            run_result = run_candidate(
                artifacts_code_path=code_path,
                run_root=verify_dir / "run",
                timeout_s=run_timeout_s,
                isolated=False,
                deny_network=False,
            )
            
            result["verification"] = {
                "passed": run_result.passed,
                "reason": run_result.reason,
                "validator": run_result.validator_used,
            }
            
            if run_result.passed:
                print("✓ Verification PASSED")
            else:
                print(f"✗ Verification FAILED: {run_result.reason}")
        else:
            print("✗ No code.py found in artifact")
            result["verification"] = {"passed": False, "reason": "no code.py"}
    
    return result


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="LLM-First optimization discovery for complex operators"
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="Path to problem file (KernelBench-style PyTorch code)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name for LLM calls (default: gpt-5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--plans-per-worker",
        type=int,
        default=3,
        help="Number of plans each worker tries (default: 3)",
    )
    parser.add_argument(
        "--refinements-per-plan",
        type=int,
        default=3,
        help="Max refinement iterations per plan (default: 3)",
    )
    parser.add_argument(
        "--llm-timeout-s",
        type=int,
        default=1200,
        help="Timeout for LLM calls in seconds (default: 1200)",
    )
    parser.add_argument(
        "--run-timeout-s",
        type=int,
        default=1200,
        help="Timeout for code execution in seconds (default: 1200)",
    )
    parser.add_argument(
        "--stream-mode",
        choices=["all", "winner", "none"],
        default="all",
        help="Streaming mode for output (default: all)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify final result after optimization",
    )
    parser.add_argument(
        "--out-root",
        help="Output root directory (default: ./.fuse)",
    )
    
    args = parser.parse_args(argv)
    
    # Validate problem path (convert relative to absolute if needed)
    try:
        problem_input = Path(args.problem)
        if not problem_input.is_absolute():
            problem_input = Path.cwd() / problem_input
        problem_path = ensure_abs_regular_file(str(problem_input))
    except PathSafetyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    
    # Parse out_root
    out_root = Path(args.out_root) if args.out_root else None
    
    print("=" * 60)
    print("LLM-First Optimization Discovery")
    print("=" * 60)
    print(f"Problem: {problem_path}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Plans per worker: {args.plans_per_worker}")
    print(f"Refinements per plan: {args.refinements_per_plan}")
    print("=" * 60)
    print()
    print("Starting optimization...")
    sys.stdout.flush()
    
    try:
        result = run_optimization(
            problem_path=problem_path,
            model=args.model,
            workers=args.workers,
            plans_per_worker=args.plans_per_worker,
            refinements_per_plan=args.refinements_per_plan,
            llm_timeout_s=args.llm_timeout_s,
            run_timeout_s=args.run_timeout_s,
            stream_mode=args.stream_mode,
            verify=args.verify,
            out_root=out_root,
        )
        
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60)
        
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
