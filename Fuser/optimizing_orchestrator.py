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
Optimizing Orchestrator for LLM-First optimization discovery.

This orchestrator coordinates multiple OptimizingWorkers that use the
Planner/Coder/Critic/Refiner agent system to discover optimizations.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import signal
import sys
import tarfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from queue import Empty
from typing import Any, Dict

from dotenv import load_dotenv

from .config import OrchestratorConfig, ResultSummary, WorkerConfig
from .logging_utils import redact, setup_file_logger


def _optimizing_worker_process_main(
    cfg_payload: Dict[str, Any],
    problem_path: str,
    winner_queue: Any,
    cancel_event: Any,
    console_q: Any,
) -> None:
    """Worker process entry point for optimizing worker."""
    import sys
    print(f"[Worker {cfg_payload['worker_id']}] Process started", flush=True)
    
    from pathlib import Path as _P
    from .config import WorkerConfig as _WC
    from .optimizing_worker import OptimizingWorker as _Worker

    # Load environment variables in worker process
    load_dotenv()
    
    print(f"[Worker {cfg_payload['worker_id']}] Imports completed", flush=True)

    # Rehydrate dataclass
    wcfg = _WC(
        run_id=cfg_payload["run_id"],
        worker_id=cfg_payload["worker_id"],
        variant_index=cfg_payload["variant_index"],
        model=cfg_payload["model"],
        max_iters=cfg_payload["max_iters"],
        llm_timeout_s=cfg_payload["llm_timeout_s"],
        run_timeout_s=cfg_payload["run_timeout_s"],
        store_responses=cfg_payload["store_responses"],
        isolated=cfg_payload["isolated"],
        deny_network=cfg_payload["deny_network"],
        enable_reasoning_extras=cfg_payload["enable_reasoning_extras"],
        stream_dir=_P("."),
        workspace_dir=_P(cfg_payload["workspace_dir"]),
        shared_digests_dir=_P(cfg_payload["shared_digests_dir"]),
    )

    def _on_delta(s: str) -> None:
        try:
            console_q.put_nowait(s)
        except Exception:
            pass

    # Get optimization parameters
    num_plans_to_try = cfg_payload.get("num_plans_to_try", 3)
    max_refinements_per_plan = cfg_payload.get("max_refinements_per_plan", 3)
    shared_plans_path = cfg_payload.get("shared_plans_path")

    print(f"[Worker {cfg_payload['worker_id']}] Starting optimization worker", flush=True)
    
    try:
        worker = _Worker(
            cfg=wcfg,
            problem_path=_P(problem_path),
            winner_queue=winner_queue,
            cancel_event=cancel_event,
            on_delta=_on_delta,
            num_plans_to_try=num_plans_to_try,
            max_refinements_per_plan=max_refinements_per_plan,
        )
        # Set shared_plans_path as attribute
        if shared_plans_path:
            worker.shared_plans_path = shared_plans_path
        worker.run()
        print(f"[Worker {cfg_payload['worker_id']}] Completed successfully", flush=True)
    except Exception as e:
        print(f"[Worker {cfg_payload['worker_id']}] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


class OptimizingOrchestrator:
    """
    Orchestrator for LLM-First optimization discovery.
    
    This orchestrator:
    1. Spawns multiple workers
    2. Each worker independently:
       - Generates optimization plans
       - Implements top K plans
       - Benchmarks and refines
    3. Collects and ranks all results
    4. Returns the best optimization
    """

    def __init__(
        self,
        cfg: OrchestratorConfig,
        run_dir: Path,
        workers_dir: Path,
        orchestrator_dir: Path,
        num_plans_per_worker: int = 3,
        max_refinements_per_plan: int = 3,
    ) -> None:
        """
        Initialize the optimizing orchestrator.
        
        Args:
            cfg: Orchestrator configuration
            run_dir: Run directory
            workers_dir: Workers directory
            orchestrator_dir: Orchestrator directory
            num_plans_per_worker: Number of plans each worker tries (default: 3)
            max_refinements_per_plan: Max refinements per plan (default: 3)
        """
        self.cfg = cfg
        self.run_dir = run_dir
        self.workers_dir = workers_dir
        self.orchestrator_dir = orchestrator_dir
        self.num_plans_per_worker = num_plans_per_worker
        self.max_refinements_per_plan = max_refinements_per_plan
        
        self.logger = setup_file_logger(orchestrator_dir / "orchestrator.log")
        self.cancel_event = mp.Event()
        self.winner_queue: mp.Queue[Any] = mp.Queue(maxsize=10)  # Allow multiple winners
        self.console_threads: list[threading.Thread] = []
        self._stop_console = threading.Event()
        self._stream_mode = cfg.stream_mode

    def _make_worker_cfg(self, idx: int) -> WorkerConfig:
        """Create worker configuration."""
        worker_id = f"worker_{idx + 1:02d}"
        wdir = self.workers_dir / worker_id
        wdir.mkdir(parents=True, exist_ok=True)
        digests_dir = self.run_dir / "shared" / "digests"
        return WorkerConfig(
            run_id=str(self.run_dir.name),
            worker_id=worker_id,
            variant_index=idx % 4,
            model=self.cfg.model,
            max_iters=self.cfg.max_iters,
            llm_timeout_s=self.cfg.llm_timeout_s,
            run_timeout_s=self.cfg.run_timeout_s,
            store_responses=self.cfg.store_responses,
            isolated=self.cfg.isolated,
            deny_network=self.cfg.deny_network,
            enable_reasoning_extras=self.cfg.enable_reasoning_extras,
            stream_dir=self.orchestrator_dir / "stream.log",
            workspace_dir=wdir,
            shared_digests_dir=digests_dir,
        )

    def _start_console_mux(self, queues: dict[str, mp.Queue[str]]) -> None:
        """Start console multiplexer for streaming output."""
        if self._stream_mode == "none":
            return

        def loop() -> None:
            last_flush: dict[str, float] = {k: 0.0 for k in queues}
            stream_path = self.orchestrator_dir / "stream.log"
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            f = stream_path.open("a", encoding="utf-8")
            try:
                while not self._stop_console.is_set():
                    now = time.time()
                    for wid, q in queues.items():
                        if now - last_flush[wid] < 0.05:
                            continue
                        chunks: list[str] = []
                        while True:
                            try:
                                delta = q.get_nowait()
                                chunks.append(delta)
                            except Empty:
                                break
                        if chunks:
                            last_flush[wid] = now
                            text = redact("".join(chunks))
                            try:
                                f.write(f"[{wid}] " + text)
                                f.flush()
                            except Exception:
                                pass
                            if self._stream_mode == "all":
                                try:
                                    print(f"[{wid}] ", end="", flush=False)
                                    print(text, end="", flush=True)
                                except Exception:
                                    pass
                    time.sleep(0.02)
            finally:
                try:
                    f.close()
                except Exception:
                    pass

        t = threading.Thread(target=loop, name="console-mux", daemon=True)
        t.start()
        self.console_threads.append(t)

    def _stop_console_mux(self) -> None:
        """Stop console multiplexer."""
        self._stop_console.set()
        for t in self.console_threads:
            t.join(timeout=1.0)

    def run(self) -> ResultSummary:
        """
        Run the optimizing orchestrator.
        
        Returns:
            ResultSummary with best optimization result
        """
        # Signal handling
        is_main_thread = threading.current_thread() is threading.main_thread()
        if is_main_thread:
            def _sig_handler(signum: int, frame: Any) -> None:
                self.logger.info("received signal %s; canceling", signum)
                self.cancel_event.set()

            old_int = signal.signal(signal.SIGINT, _sig_handler)
            old_term = signal.signal(signal.SIGTERM, _sig_handler)
        else:
            old_int = old_term = None

        try:
            # Per-worker output queues
            delta_queues: dict[str, mp.Queue[str]] = {}
            procs: list[mp.Process] = []

            self.logger.info(f"Starting {self.cfg.workers} worker processes")
            print(f"Spawning {self.cfg.workers} workers...")
            sys.stdout.flush()

            # Generate shared plans once (instead of per-worker)
            shared_plans_path = None
            if self.cfg.workers > 1:
                print("Generating shared optimization plans...")
                sys.stdout.flush()
                try:
                    from .llm_optimizer import LLMOptimizer
                    optimizer = LLMOptimizer(
                        model=self.cfg.model,
                        timeout_s=self.cfg.llm_timeout_s,
                        enable_reasoning_extras=self.cfg.enable_reasoning_extras,
                    )
                    problem_code = self.cfg.problem_path.read_text(encoding="utf-8")
                    planner_output = optimizer.plan(
                        problem_code=problem_code,
                        output_dir=self.orchestrator_dir / "shared_planning",
                    )
                    # Save shared plans
                    shared_plans_path = self.orchestrator_dir / "shared_plans.json"
                    import json
                    from dataclasses import asdict
                    shared_plans_path.write_text(
                        json.dumps(asdict(planner_output), indent=2), encoding="utf-8"
                    )
                    print(f"Generated {len(planner_output.plans)} shared plans")
                    sys.stdout.flush()
                except Exception as e:
                    self.logger.warning(f"Failed to generate shared plans: {e}, workers will generate independently")
                    print(f"Warning: Failed to generate shared plans, workers will generate independently")
                    sys.stdout.flush()

            for i in range(self.cfg.workers):
                wcfg = self._make_worker_cfg(i)
                dq: mp.Queue[str] = mp.Queue(maxsize=256)
                delta_queues[wcfg.worker_id] = dq

                cfg_payload = {
                    "run_id": wcfg.run_id,
                    "worker_id": wcfg.worker_id,
                    "variant_index": wcfg.variant_index,
                    "model": wcfg.model,
                    "max_iters": wcfg.max_iters,
                    "llm_timeout_s": wcfg.llm_timeout_s,
                    "run_timeout_s": wcfg.run_timeout_s,
                    "store_responses": wcfg.store_responses,
                    "isolated": wcfg.isolated,
                    "deny_network": wcfg.deny_network,
                    "enable_reasoning_extras": wcfg.enable_reasoning_extras,
                    "workspace_dir": str(wcfg.workspace_dir),
                    "shared_digests_dir": str(wcfg.shared_digests_dir),
                    "num_plans_to_try": self.num_plans_per_worker,
                    "max_refinements_per_plan": self.max_refinements_per_plan,
                    "shared_plans_path": str(shared_plans_path) if shared_plans_path else None,
                }
                
                p = mp.Process(
                    target=_optimizing_worker_process_main,
                    name=wcfg.worker_id,
                    args=(
                        cfg_payload,
                        str(self.cfg.problem_path),
                        self.winner_queue,
                        self.cancel_event,
                        dq,
                    ),
                )
                p.start()
                procs.append(p)
                self.logger.info(f"Started worker {i} (PID: {p.pid})")
                print(f"  Worker {i} started (PID: {p.pid})")
                sys.stdout.flush()

            print(f"All {len(procs)} workers started, waiting for results...")
            sys.stdout.flush()
            self._start_console_mux(delta_queues)

            # Collect all results
            winners: list[dict[str, Any]] = []
            canceled_by_signal = False

            while True:
                if self.cancel_event.is_set():
                    canceled_by_signal = True
                    break
                try:
                    winner = self.winner_queue.get(timeout=0.1)
                    winners.append(winner)
                    self.logger.info(
                        "Worker %s completed with plan %s: speedup=%.2fx",
                        winner.get("worker_id"),
                        winner.get("plan_id"),
                        winner.get("speedup", 0.0),
                    )
                except Empty:
                    # Check if all workers finished
                    if all(not p.is_alive() for p in procs):
                        break

            self._stop_console_mux()

            # Wait for all workers
            for p in procs:
                if p.is_alive():
                    try:
                        p.join(timeout=10.0)
                    except Exception:
                        pass
                    if p.is_alive():
                        self.logger.warning(f"Worker {p.name} did not finish, terminating")
                        try:
                            p.terminate()
                            p.join(timeout=2.0)
                        except Exception:
                            pass

            # Collect remaining results
            while not self.winner_queue.empty():
                try:
                    winner = self.winner_queue.get_nowait()
                    winners.append(winner)
                except Empty:
                    break

            # Select best result by speedup
            winner: dict[str, Any] | None = None
            if winners:
                winner = max(winners, key=lambda w: w.get("speedup", 0.0))
                self.logger.info(
                    "Selected best result: worker=%s plan=%s speedup=%.2fx (from %d total)",
                    winner.get("worker_id"),
                    winner.get("plan_id"),
                    winner.get("speedup", 0.0),
                    len(winners),
                )

            reason = (
                "canceled"
                if winner is None and canceled_by_signal
                else (
                    f"pass with plan {winner.get('plan_id')} speedup={winner.get('speedup', 0.0):.2f}x"
                    if winner is not None
                    else "no_passing_solution"
                )
            )
            
            summary = ResultSummary(
                run_id=str(self.run_dir.name),
                winner_worker_id=winner.get("worker_id") if winner else None,
                artifact_path=None,
                reason=reason,
            )

            if winner is not None:
                try:
                    # Package the best result
                    result_tar = self.run_dir / "result.tar.gz"
                    with tarfile.open(result_tar, "w:gz") as tf:
                        # Include the best code
                        code_path = Path(winner["artifacts_dir"]) / "best_code.py"
                        if code_path.exists():
                            tf.add(str(code_path), arcname="code.py")
                        
                        # Include runs directory
                        runs_dir = Path(winner.get("runs_dir", ""))
                        if runs_dir.is_dir():
                            tf.add(str(runs_dir), arcname="runs")
                    
                    summary.artifact_path = str(result_tar)
                    
                    # Save winner info
                    win_dir = self.orchestrator_dir / "winner"
                    win_dir.mkdir(parents=True, exist_ok=True)
                    (win_dir / "worker_id.txt").write_text(
                        winner["worker_id"], encoding="utf-8"
                    )
                    (win_dir / "plan_id.txt").write_text(
                        winner.get("plan_id", "unknown"), encoding="utf-8"
                    )
                    (win_dir / "speedup.txt").write_text(
                        f"{winner.get('speedup', 0.0):.4f}", encoding="utf-8"
                    )
                except Exception as e:
                    self.logger.error("packaging failed: %s", e)
                    summary.reason = f"packaging_failure: {e}"

            (self.orchestrator_dir / "summary.json").write_text(
                json.dumps(asdict(summary), indent=2), encoding="utf-8"
            )
            return summary
        finally:
            if is_main_thread:
                assert old_int is not None and old_term is not None
                signal.signal(signal.SIGINT, old_int)
                signal.signal(signal.SIGTERM, old_term)
