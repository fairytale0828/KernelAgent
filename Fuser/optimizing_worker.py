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
Optimizing Worker that uses LLM-First optimization discovery.

This worker replaces the traditional single-prompt approach with a multi-agent
system that explores multiple optimization strategies.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any, Callable, List, Dict

from .config import WorkerConfig
from .llm_optimizer import LLMOptimizer, OptimizationPlan
from .runner import run_candidate
from .logging_utils import setup_file_logger
from .dedup import register_digest
from .code_extractor import sha256_of_code


@dataclass
class OptimizationResult:
    """Result from optimizing a single plan."""
    plan_id: str
    success: bool
    code: Optional[str]
    iterations: int
    best_speedup: float
    error_msg: Optional[str]
    run_dir: Path


class OptimizingWorker:
    """
    Worker that discovers and implements optimizations using LLM agents.
    
    This worker:
    1. Uses Planner to generate multiple optimization plans
    2. Selects top K plans to implement
    3. For each plan:
       - Uses Coder to generate implementation
       - Uses Critic to audit before running
       - Runs and benchmarks
       - Uses Refiner to fix errors or improve performance
    4. Returns the best result
    """
    
    def __init__(
        self,
        cfg: WorkerConfig,
        problem_path: Path,
        winner_queue: Any,
        cancel_event: Any,
        on_delta: Optional[Callable[[str], None]] = None,
        num_plans_to_try: int = 3,
        max_refinements_per_plan: int = 3,
    ):
        """
        Initialize the optimizing worker.
        
        Args:
            cfg: Worker configuration
            problem_path: Path to problem file
            winner_queue: Queue to report success
            cancel_event: Event to signal cancellation
            on_delta: Optional callback for streaming output
            num_plans_to_try: Number of top plans to implement (default: 3)
            max_refinements_per_plan: Max refinement iterations per plan (default: 3)
        """
        self.cfg = cfg
        self.problem_path = problem_path
        self.winner_queue = winner_queue
        self.cancel_event = cancel_event
        self.on_delta = on_delta
        self.num_plans_to_try = num_plans_to_try
        self.max_refinements_per_plan = max_refinements_per_plan
        
        self.logger = setup_file_logger(
            cfg.workspace_dir / "logs" / "optimizing_worker.log",
            name=f"optimizing-worker-{cfg.worker_id}"
        )
        
        # Setup directories
        self.dirs = self._ensure_dirs(cfg.workspace_dir)
        
        # Initialize LLM optimizer
        self.optimizer = LLMOptimizer(
            model=cfg.model,
            timeout_s=cfg.llm_timeout_s,
            enable_reasoning_extras=cfg.enable_reasoning_extras,
        )
    
    def _ensure_dirs(self, base: Path) -> Dict[str, Path]:
        """Create necessary directories."""
        dirs = {
            "input": base / "input",
            "planning": base / "planning",
            "implementations": base / "implementations",
            "runs": base / "runs",
            "logs": base / "logs",
            "artifacts": base / "artifacts",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs
    
    def run(self) -> None:
        """Run the optimization discovery process."""
        self.logger.info(f"Worker {self.cfg.worker_id} starting optimization discovery")
        print(f"[{self.cfg.worker_id}] Starting optimization discovery", flush=True)
        
        # Read problem code
        problem_code = self.problem_path.read_text(encoding="utf-8")
        
        # Save problem
        (self.dirs["input"] / "problem.py").write_text(problem_code, encoding="utf-8")
        
        print(f"[{self.cfg.worker_id}] Problem loaded, starting planning phase", flush=True)
        
        # Phase 1: Planning - Generate or load shared optimization plans
        self.logger.info("Phase 1: Getting optimization plans")
        print(f"[{self.cfg.worker_id}] Calling Planner LLM...", flush=True)
        
        # Check if shared plans are available
        shared_plans_path = getattr(self, 'shared_plans_path', None)
        if shared_plans_path and Path(shared_plans_path).exists():
            # Load shared plans
            print(f"[{self.cfg.worker_id}] Loading shared plans from {shared_plans_path}", flush=True)
            try:
                import json
                from .llm_optimizer import PlannerOutput, OptimizationPlan
                with open(shared_plans_path, 'r') as f:
                    plans_data = json.load(f)
                
                # Reconstruct PlannerOutput
                plans = []
                for plan_data in plans_data.get("plans", []):
                    plan = OptimizationPlan(**plan_data)
                    plans.append(plan)
                
                planner_output = PlannerOutput(
                    problem_fingerprint=plans_data.get("problem_fingerprint", ""),
                    detected_patterns=plans_data.get("detected_patterns", []),
                    plans=plans,
                    recommendations=plans_data.get("recommendations", {}),
                )
                
                print(f"[{self.cfg.worker_id}] Loaded {len(planner_output.plans)} shared plans", flush=True)
                self.logger.info(f"Loaded {len(planner_output.plans)} shared plans")
            except Exception as e:
                print(f"[{self.cfg.worker_id}] Failed to load shared plans: {e}, generating independently", flush=True)
                self.logger.warning(f"Failed to load shared plans: {e}, generating independently")
                shared_plans_path = None
        
        if not shared_plans_path or not Path(shared_plans_path).exists():
            # Generate plans independently
            try:
                planner_output = self.optimizer.plan(
                    problem_code=problem_code,
                    output_dir=self.dirs["planning"],
                )
                print(f"[{self.cfg.worker_id}] Planner completed, generated {len(planner_output.plans)} plans", flush=True)
                self.logger.info(f"Generated {len(planner_output.plans)} plans")
                self.logger.info(f"Detected patterns: {planner_output.detected_patterns}")
            except Exception as e:
                print(f"[{self.cfg.worker_id}] Planning failed: {e}", flush=True)
                self.logger.error(f"Planning failed: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Select top K plans to implement
        # Strategy: Different workers should try different plans for better exploration
        recommended_ids = planner_output.recommendations.get("pick_first", [])
        plans_to_try = []
        
        # Get worker index from worker_id (e.g., "worker_01" -> 0)
        try:
            worker_idx = int(self.cfg.worker_id.split('_')[-1]) - 1
        except:
            worker_idx = 0
        
        # Strategy 1: If we have recommendations, distribute them across workers
        if recommended_ids:
            # Each worker starts from a different offset in the recommended list
            start_idx = (worker_idx * self.num_plans_to_try) % len(recommended_ids)
            for i in range(self.num_plans_to_try):
                plan_id = recommended_ids[(start_idx + i) % len(recommended_ids)]
                plan = next((p for p in planner_output.plans if p.plan_id == plan_id), None)
                if plan and plan not in plans_to_try:
                    plans_to_try.append(plan)
        
        # Strategy 2: Fill remaining slots with non-recommended plans
        # Each worker gets a different subset
        remaining_slots = self.num_plans_to_try - len(plans_to_try)
        if remaining_slots > 0:
            non_recommended = [p for p in planner_output.plans if p not in plans_to_try]
            start_idx = (worker_idx * remaining_slots) % max(len(non_recommended), 1)
            for i in range(remaining_slots):
                if non_recommended:
                    idx = (start_idx + i) % len(non_recommended)
                    plans_to_try.append(non_recommended[idx])
        
        self.logger.info(
            f"Worker {self.cfg.worker_id} selected {len(plans_to_try)} plans: "
            f"{[p.plan_id for p in plans_to_try]}"
        )
        print(
            f"[{self.cfg.worker_id}] Selected plans: {[p.plan_id for p in plans_to_try]}",
            flush=True
        )
        
        # Phase 2: Implementation - Try each plan
        results: List[OptimizationResult] = []
        
        for plan_idx, plan in enumerate(plans_to_try):
            if self.cancel_event.is_set():
                self.logger.info("Cancellation requested, stopping")
                break
            
            self.logger.info(f"Phase 2.{plan_idx + 1}: Implementing plan {plan.plan_id} ({plan.tag})")
            self.logger.info(f"  Idea: {plan.high_level_idea}")
            
            result = self._implement_plan(plan, plan_idx)
            results.append(result)
            
            if result.success:
                self.logger.info(f"Plan {plan.plan_id} succeeded with speedup {result.best_speedup:.2f}x")
            else:
                self.logger.info(f"Plan {plan.plan_id} failed: {result.error_msg}")
        
        # Phase 3: Selection - Pick the best result
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            # Sort by speedup (higher is better)
            best_result = max(successful_results, key=lambda r: r.best_speedup)
            self.logger.info(f"Best result: plan {best_result.plan_id} with speedup {best_result.best_speedup:.2f}x")
            
            # Report winner
            winner_data = {
                "worker_id": self.cfg.worker_id,
                "plan_id": best_result.plan_id,
                "iter": best_result.iterations,
                "speedup": best_result.best_speedup,
                "validator": "run_tests",
                "runs_dir": str(best_result.run_dir),
                "artifacts_dir": str(self.dirs["artifacts"]),
                "code": best_result.code,
            }
            
            # Save best result
            (self.dirs["artifacts"] / "best_code.py").write_text(best_result.code, encoding="utf-8")
            (self.dirs["artifacts"] / "best_result.json").write_text(
                json.dumps(winner_data, indent=2), encoding="utf-8"
            )
            
            try:
                self.winner_queue.put(winner_data, timeout=0.1)
            except Exception as e:
                self.logger.error(f"Failed to report winner: {e}")
        else:
            self.logger.warning("No successful implementations found")
    
    def _implement_plan(self, plan: OptimizationPlan, plan_idx: int) -> OptimizationResult:
        """
        Implement a single optimization plan.
        
        Returns:
            OptimizationResult with success status and metrics
        """
        plan_dir = self.dirs["implementations"] / f"plan_{plan_idx}_{plan.plan_id}"
        plan_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plan
        (plan_dir / "plan.json").write_text(
            json.dumps(asdict(plan), indent=2), encoding="utf-8"
        )
        
        problem_code = self.problem_path.read_text(encoding="utf-8")
        
        previous_code = None
        error_context = None
        perf_context = None
        best_speedup = 0.0
        
        for iteration in range(self.max_refinements_per_plan):
            if self.cancel_event.is_set():
                break
            
            iter_dir = plan_dir / f"iteration_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"  Iteration {iteration + 1}/{self.max_refinements_per_plan}")
            
            # Generate or refine code
            if iteration == 0:
                # First iteration: use Coder
                try:
                    code = self.optimizer.code(
                        problem_code=problem_code,
                        selected_plan=plan,
                        output_dir=iter_dir,
                    )
                except Exception as e:
                    self.logger.error(f"Coder failed: {e}")
                    return OptimizationResult(
                        plan_id=plan.plan_id,
                        success=False,
                        code=None,
                        iterations=iteration + 1,
                        best_speedup=0.0,
                        error_msg=f"Coder failed: {e}",
                        run_dir=plan_dir,
                    )
            else:
                # Subsequent iterations: use Refiner
                try:
                    code = self.optimizer.refine(
                        problem_code=problem_code,
                        selected_plan=plan,
                        previous_code=previous_code,
                        error_context=error_context,
                        perf_context=perf_context,
                        output_dir=iter_dir,
                    )
                except Exception as e:
                    self.logger.error(f"Refiner failed: {e}")
                    return OptimizationResult(
                        plan_id=plan.plan_id,
                        success=False,
                        code=previous_code,
                        iterations=iteration + 1,
                        best_speedup=best_speedup,
                        error_msg=f"Refiner failed: {e}",
                        run_dir=plan_dir,
                    )
            
            # Save code
            (iter_dir / "code.py").write_text(code, encoding="utf-8")
            
            # Check for duplicates
            sha = sha256_of_code(code)
            status, owner = register_digest(
                self.cfg.shared_digests_dir, sha, self.cfg.worker_id, iteration
            )
            if status == "duplicate_cross_worker":
                self.logger.info(f"  Duplicate code (owner={owner}), skipping")
                continue
            if status == "duplicate_same_worker":
                self.logger.info(f"  Duplicate code in same worker, skipping")
                continue
            
            # Audit with Critic (only on first iteration to save time)
            if iteration == 0:
                try:
                    critic_output = self.optimizer.critique(
                        selected_plan=plan,
                        candidate_code=code,
                        output_dir=iter_dir,
                    )
                    
                    if critic_output.verdict == "fail" and critic_output.critical_issues:
                        self.logger.warning(f"  Critic found critical issues: {critic_output.critical_issues}")
                        # Continue anyway, but log the issues
                except Exception as e:
                    self.logger.warning(f"Critic failed: {e}")
            
            # Run the code
            run_root = self.dirs["runs"] / f"plan_{plan_idx}" / f"iteration_{iteration}"
            run_root.mkdir(parents=True, exist_ok=True)
            
            try:
                run_result = run_candidate(
                    artifacts_code_path=iter_dir / "code.py",
                    run_root=run_root,
                    timeout_s=self.cfg.run_timeout_s,
                    isolated=self.cfg.isolated,
                    deny_network=self.cfg.deny_network,
                    cancel_event=self.cancel_event,
                )
            except Exception as e:
                self.logger.error(f"  Run failed: {e}")
                error_context = f"Exception during run: {e}"
                previous_code = code
                continue
            
            # Check result
            if run_result.passed:
                # Extract speedup from stdout (if microbench was run)
                speedup = self._extract_speedup(run_result.stdout_path)
                if speedup > best_speedup:
                    best_speedup = speedup
                
                self.logger.info(f"  SUCCESS! Speedup: {speedup:.2f}x")
                
                return OptimizationResult(
                    plan_id=plan.plan_id,
                    success=True,
                    code=code,
                    iterations=iteration + 1,
                    best_speedup=speedup,
                    error_msg=None,
                    run_dir=run_root,
                )
            else:
                # Build error context for refinement
                stdout_tail = self._tail_text(run_result.stdout_path, 2000)
                stderr_tail = self._tail_text(run_result.stderr_path, 2000)
                
                error_context = f"RUN_FAIL: {run_result.reason}\n\nSTDOUT_TAIL:\n{stdout_tail}\n\nSTDERR_TAIL:\n{stderr_tail}"
                
                # Extract performance context if available
                speedup = self._extract_speedup(run_result.stdout_path)
                if speedup > 0:
                    perf_context = f"Partial speedup: {speedup:.2f}x (but correctness failed)"
                else:
                    perf_context = "No performance data available"
                
                previous_code = code
                self.logger.info(f"  Failed: {run_result.reason}")
        
        # All iterations exhausted without success
        return OptimizationResult(
            plan_id=plan.plan_id,
            success=False,
            code=previous_code,
            iterations=self.max_refinements_per_plan,
            best_speedup=best_speedup,
            error_msg="Max refinements reached without success",
            run_dir=plan_dir,
        )
    
    def _tail_text(self, path: Path, max_bytes: int) -> str:
        """Read tail of a file."""
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                take = min(size, max_bytes)
                f.seek(size - take)
                return f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""
    
    def _extract_speedup(self, stdout_path: Path) -> float:
        """Extract speedup from microbench output."""
        try:
            stdout = stdout_path.read_text(encoding="utf-8")
            
            # Look for patterns like "Speedup: 2.5x" or "speedup=2.5"
            import re
            patterns = [
                r"[Ss]peedup[:\s=]+(\d+\.?\d*)",
                r"(\d+\.?\d*)[xX]\s+faster",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, stdout)
                if match:
                    return float(match.group(1))
            
            # Default to 1.0 if no speedup found but test passed
            return 1.0
        except Exception:
            return 0.0
