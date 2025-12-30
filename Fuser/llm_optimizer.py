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
LLM-First Optimization System for Fuser.

This module implements a 4-agent system (Planner/Coder/Critic/Refiner) that
discovers algorithmic optimizations, algebraic transforms, and fusion strategies
for complex operators.
"""

from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .event_adapter import EventAdapter
from .code_extractor import extract_single_python_file
from .runner import run_candidate


@dataclass
class OptimizationPlan:
    """Represents a single optimization plan."""
    plan_id: str
    tag: str  # baseline|algebraic_rewrite|algorithmic_variant|fusion_aggressive|memory_saving
    high_level_idea: str
    subgraphs: List[Dict[str, Any]]
    algebraic_transforms: List[Dict[str, Any]]
    algorithmic_variants: List[Dict[str, Any]]
    kernel_plan: Dict[str, Any]
    expected_tradeoffs: Dict[str, Any]
    acceptance_tests: List[str]


@dataclass
class PlannerOutput:
    """Output from the Planner agent."""
    problem_fingerprint: str
    detected_patterns: List[str]
    plans: List[OptimizationPlan]
    recommendations: Dict[str, Any]


@dataclass
class CriticOutput:
    """Output from the Critic agent."""
    verdict: str  # pass|fail
    critical_issues: List[str]
    noncritical_issues: List[str]
    suggested_patches: List[Dict[str, str]]
    risk_level: str  # low|med|high


class LLMOptimizer:
    """
    LLM-First optimizer that discovers and implements optimizations.
    
    This class coordinates 4 agents:
    - Planner: Generates multiple optimization plans
    - Coder: Implements a selected plan
    - Critic: Audits code before execution
    - Refiner: Fixes errors and improves performance
    """
    
    def __init__(
        self,
        model: str,
        timeout_s: int = 1200,
        enable_reasoning_extras: bool = True,
    ):
        """
        Initialize the LLM optimizer.
        
        Args:
            model: Model name for LLM calls
            timeout_s: Timeout for LLM calls
            enable_reasoning_extras: Whether to enable high reasoning effort
        """
        self.model = model
        self.timeout_s = timeout_s
        self.enable_reasoning_extras = enable_reasoning_extras
    
    def plan(
        self,
        problem_code: str,
        subgraphs_json: Optional[str] = None,
        trace_summary: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> PlannerOutput:
        """
        Generate multiple optimization plans using the Planner agent.
        
        Args:
            problem_code: Original problem file content
            subgraphs_json: Optional subgraphs JSON from extraction
            trace_summary: Optional trace summary
            output_dir: Optional directory to save outputs
            
        Returns:
            PlannerOutput with multiple plans
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("Building Planner prompt...")
        system_prompt = "You are a senior GPU compiler+kernel engineer. You must output ONLY valid JSON (no markdown, no prose)."
        
        user_prompt = self._build_planner_prompt(
            problem_code=problem_code,
            subgraphs_json=subgraphs_json,
            trace_summary=trace_summary,
        )
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "planner_prompt.txt").write_text(user_prompt, encoding="utf-8")
        
        logger.info(f"Calling LLM with model: {self.model}")
        print(f"  Calling Planner LLM (model: {self.model}, timeout: {self.timeout_s}s)...", flush=True)
        
        # Call LLM
        jsonl_path = output_dir / "planner.stream.jsonl" if output_dir else Path("/tmp/planner.jsonl")
        adapter = EventAdapter(
            model=self.model,
            store_responses=False,
            timeout_s=self.timeout_s,
            jsonl_path=jsonl_path,
        )
        
        extras = {}
        if self.enable_reasoning_extras:
            extras["reasoning"] = {"effort": "high"}
            extras["text"] = {"format": {"type": "text"}}
        
        logger.info("Starting LLM stream...")
        print(f"  LLM stream starting...", flush=True)
        
        result = adapter.stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extras=extras,
        )
        
        logger.info("LLM stream completed")
        print(f"  LLM stream completed", flush=True)
        
        output_text = result.get("output_text", "")
        
        if output_dir:
            (output_dir / "planner_response.txt").write_text(output_text, encoding="utf-8")
        
        logger.info("Parsing JSON response...")
        print(f"  Parsing JSON response...", flush=True)
        
        # Parse JSON response
        plans_data = self._extract_json_from_response(output_text)
        
        if output_dir:
            (output_dir / "planner_output.json").write_text(
                json.dumps(plans_data, indent=2), encoding="utf-8"
            )
        
        logger.info("Planner completed successfully")
        print(f"  Planner completed successfully", flush=True)
        
        return self._parse_planner_output(plans_data)
    
    def code(
        self,
        problem_code: str,
        selected_plan: OptimizationPlan,
        previous_attempt: Optional[str] = None,
        error_context: Optional[str] = None,
        perf_context: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> str:
        """
        Generate code implementation using the Coder agent.
        
        Args:
            problem_code: Original problem file content
            selected_plan: Selected optimization plan
            previous_attempt: Optional previous code attempt
            error_context: Optional error context from previous run
            perf_context: Optional performance context
            output_dir: Optional directory to save outputs
            
        Returns:
            Generated code as string
        """
        system_prompt = "Return ONE runnable Python file only, as a single fenced ```python block. No prose."
        
        user_prompt = self._build_coder_prompt(
            problem_code=problem_code,
            selected_plan=selected_plan,
            previous_attempt=previous_attempt,
            error_context=error_context,
            perf_context=perf_context,
        )
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "coder_prompt.txt").write_text(user_prompt, encoding="utf-8")
        
        # Call LLM
        jsonl_path = output_dir / "coder.stream.jsonl" if output_dir else Path("/tmp/coder.jsonl")
        adapter = EventAdapter(
            model=self.model,
            store_responses=False,
            timeout_s=self.timeout_s,
            jsonl_path=jsonl_path,
        )
        
        extras = {}
        if self.enable_reasoning_extras:
            extras["reasoning"] = {"effort": "high"}
            extras["text"] = {"format": {"type": "text"}}
        
        result = adapter.stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extras=extras,
        )
        
        output_text = result.get("output_text", "")
        
        if output_dir:
            (output_dir / "coder_response.txt").write_text(output_text, encoding="utf-8")
        
        # Extract code
        extracted = extract_single_python_file(output_text)
        code = extracted.code
        
        if output_dir:
            (output_dir / "coder_output.py").write_text(code, encoding="utf-8")
        
        return code
    
    def critique(
        self,
        selected_plan: OptimizationPlan,
        candidate_code: str,
        output_dir: Optional[Path] = None,
    ) -> CriticOutput:
        """
        Audit code using the Critic agent.
        
        Args:
            selected_plan: Selected optimization plan
            candidate_code: Code to audit
            output_dir: Optional directory to save outputs
            
        Returns:
            CriticOutput with audit results
        """
        system_prompt = "You are a strict code auditor. Output ONLY JSON (no markdown, no prose)."
        
        user_prompt = self._build_critic_prompt(
            selected_plan=selected_plan,
            candidate_code=candidate_code,
        )
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "critic_prompt.txt").write_text(user_prompt, encoding="utf-8")
        
        # Call LLM
        jsonl_path = output_dir / "critic.stream.jsonl" if output_dir else Path("/tmp/critic.jsonl")
        adapter = EventAdapter(
            model=self.model,
            store_responses=False,
            timeout_s=self.timeout_s,
            jsonl_path=jsonl_path,
        )
        
        extras = {}
        if self.enable_reasoning_extras:
            extras["text"] = {"format": {"type": "text"}}
        
        result = adapter.stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extras=extras,
        )
        
        output_text = result.get("output_text", "")
        
        if output_dir:
            (output_dir / "critic_response.txt").write_text(output_text, encoding="utf-8")
        
        # Parse JSON response
        critic_data = self._extract_json_from_response(output_text)
        
        if output_dir:
            (output_dir / "critic_output.json").write_text(
                json.dumps(critic_data, indent=2), encoding="utf-8"
            )
        
        return self._parse_critic_output(critic_data)
    
    def refine(
        self,
        problem_code: str,
        selected_plan: OptimizationPlan,
        previous_code: str,
        error_context: str,
        perf_context: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> str:
        """
        Refine code using the Refiner agent.
        
        Args:
            problem_code: Original problem file content
            selected_plan: Selected optimization plan
            previous_code: Previous code attempt
            error_context: Error context from previous run
            perf_context: Optional performance context
            output_dir: Optional directory to save outputs
            
        Returns:
            Refined code as string
        """
        system_prompt = "Return ONE runnable Python file only, as a single fenced ```python block. No prose."
        
        user_prompt = self._build_refiner_prompt(
            problem_code=problem_code,
            selected_plan=selected_plan,
            previous_code=previous_code,
            error_context=error_context,
            perf_context=perf_context,
        )
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "refiner_prompt.txt").write_text(user_prompt, encoding="utf-8")
        
        # Call LLM
        jsonl_path = output_dir / "refiner.stream.jsonl" if output_dir else Path("/tmp/refiner.jsonl")
        adapter = EventAdapter(
            model=self.model,
            store_responses=False,
            timeout_s=self.timeout_s,
            jsonl_path=jsonl_path,
        )
        
        extras = {}
        if self.enable_reasoning_extras:
            extras["reasoning"] = {"effort": "high"}
            extras["text"] = {"format": {"type": "text"}}
        
        result = adapter.stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extras=extras,
        )
        
        output_text = result.get("output_text", "")
        
        if output_dir:
            (output_dir / "refiner_response.txt").write_text(output_text, encoding="utf-8")
        
        # Extract code
        extracted = extract_single_python_file(output_text)
        code = extracted.code
        
        if output_dir:
            (output_dir / "refiner_output.py").write_text(code, encoding="utf-8")
        
        return code
    
    def _build_planner_prompt(
        self,
        problem_code: str,
        subgraphs_json: Optional[str],
        trace_summary: Optional[str],
    ) -> str:
        """Build the Planner agent prompt."""
        prompt_parts = []
        
        prompt_parts.append("""You will be given a PyTorch problem file (KernelBench-style). Your job is NOT to write final code yet.
Your job is to propose MULTIPLE optimization plans that the system can later implement and benchmark.

GOAL
- Maximize the chance of discovering algorithmic-level optimizations (e.g., FlashAttention-style streaming softmax, algebraic reorderings like RMSNorm/GEMM epilogue scaling).
- Minimize reliance on strict compiler rules: you may assume we will verify correctness by executing a self-test against a PyTorch reference.
- However, your plans must still be mechanically plausible and shape-consistent.

HARD REQUIREMENTS
1) Output MUST be a single JSON object matching the provided schema exactly.
2) You MUST propose 4-7 distinct plans.
3) Diversity constraint:
   - At least 1 plan must be a baseline / simplest correct approach.
   - At least 1 plan must involve an algorithmic variant (e.g., streaming/online softmax for attention; split-k; fusion across softmax; streaming reduction).
   - At least 1 plan must involve an algebraic transform (e.g., move scaling, fold gamma into weights, epilogue row-scale).
   - If you detect an attention-like pattern, at least 1 plan MUST be "flash_attention_streaming" style.
   - If you detect RMSNorm+GEMM or LayerNorm+GEMM, at least 1 plan MUST propose moving scaling/gamma to GEMM epilogue or folding into weights.
4) Each plan must include:
   - A subgraph breakdown (SG0, SG1, ...) with explicit input/output/weight shapes (use symbols only if absolutely necessary, but prefer concrete dims from problem).
   - A list of algebraic_transforms (can be empty for baseline).
   - A list of algorithmic_variants (can be empty for baseline).
   - A kernel_plan with a schedule_search_space (reasonable discrete sets).
   - Expected tradeoffs and risks (numerical stability, precision).
5) Assume the system will later:
   - Generate Triton code for selected plans,
   - Run run_tests() for correctness,
   - Run microbench to rank plans.

So DO NOT worry about proving equivalence; focus on proposing strong candidates with clear assumptions.

OUTPUT JSON SCHEMA (must follow exactly):
{
  "problem_fingerprint": "<string>",
  "detected_patterns": ["attention|rmsnorm_gemm|gemm_epilogue|conv_chain|reduce|pointwise_chain|other..."],
  "plans": [
    {
      "plan_id": "P1",
      "tag": "baseline|algebraic_rewrite|algorithmic_variant|fusion_aggressive|memory_saving",
      "high_level_idea": "<one sentence>",
      "subgraphs": [
        {
          "sg_id": "SG0",
          "ops_summary": "<string>",
          "inputs": [{"name":"x","shape":"[...]", "dtype":"float16|float32|bf16", "layout":"NCHW|NHWC|contiguous|strided"}],
          "outputs": [{"name":"y","shape":"[...]", "dtype":"..."}],
          "weights": [{"name":"W","shape":"[...]", "dtype":"..."}],
          "notes": "<shape/stride/broadcast notes>"
        }
      ],
      "algebraic_transforms": [
        {
          "name": "<transform_name>",
          "before": "<math expression>",
          "after": "<math expression>",
          "assumptions": ["broadcast condition...", "dtype tolerance..."],
          "risk": "<fp16 error/numerical stability risk>"
        }
      ],
      "algorithmic_variants": [
        {
          "name": "flash_attention_streaming|naive_attention|online_softmax|splitk_gemm|row_scale_epilogue|...",
          "where": "SGx",
          "notes": "<core invariants/numerical stability strategy>"
        }
      ],
      "kernel_plan": {
        "kernels": [
          {
            "kernel_id": "K0",
            "covers_subgraphs": ["SG0","SG1"],
            "launches": 1,
            "intermediate_buffers": [{"name":"tmp0","shape":"[...]", "lifetime":"within_kernel|between_kernels"}],
            "schedule_search_space": {
              "BLOCK_M": [16, 32, 64, 128],
              "BLOCK_N": [16, 32, 64, 128],
              "num_warps": [2, 4, 8],
              "num_stages": [1, 2, 3, 4]
            }
          }
        ]
      },
      "expected_tradeoffs": {
        "launch_count": "<int or range>",
        "memory_io": "<qualitative>",
        "numerical_stability": "<qualitative>",
        "implementation_risk": "<low|med|high>"
      },
      "acceptance_tests": ["allclose vs reference rtol/atol ...", "metamorphic test ideas ..."]
    }
  ],
  "recommendations": {
    "pick_first": ["P2","P3"],
    "why": "<string>"
  }
}

INPUTS
""")
        
        prompt_parts.append(f"- PROBLEM_FILE_CONTENT:\n```python\n{problem_code}\n```\n")
        
        if subgraphs_json:
            prompt_parts.append(f"\n- SUBGRAPHS_JSON:\n```json\n{subgraphs_json}\n```\n")
        
        if trace_summary:
            prompt_parts.append(f"\n- TRACE_SUMMARY:\n```json\n{trace_summary}\n```\n")
        
        prompt_parts.append("\nNow output ONLY the JSON object.")
        
        return "\n".join(prompt_parts)
    
    def _build_coder_prompt(
        self,
        problem_code: str,
        selected_plan: OptimizationPlan,
        previous_attempt: Optional[str],
        error_context: Optional[str],
        perf_context: Optional[str],
    ) -> str:
        """Build the Coder agent prompt."""
        prompt_parts = []
        
        prompt_parts.append("""You are implementing ONE selected optimization plan as Triton code.

HARD REQUIREMENTS (non-negotiable)
1) Output MUST be ONE complete Python file, fenced as a single ```python block.
2) The file MUST define:
   - kernel_function(...) : the primary entry point, same semantic as the original model forward for the given shapes.
   - run_tests() : compares kernel_function output with a PyTorch reference derived from the original problem file and prints 'PASS' then exits(0) on success.
   - microbench() : runs kernel_function multiple times and prints timing statistics (at least median or avg).
3) kernel_function MUST compute final outputs using Triton kernels ONLY.
   - DO NOT use torch.nn / torch.nn.functional / torch.* elementwise/matmul/conv/softmax/etc in kernel_function.
   - Using torch is allowed ONLY for: allocating tensors, reference computation inside run_tests, and timing utilities.
4) No network access. No file I/O beyond current directory. No extra deps beyond: torch, triton, triton.language as tl, and stdlib.
5) Deterministic: set seeds where relevant.

NUMERICAL CHECK
- For fp32: allclose rtol<=1e-3, atol<=1e-3
- For fp16/bf16: allow up to rtol<=2e-2, atol<=2e-2
- If your plan uses online softmax/streaming, implement numerically stable log-sum-exp style accumulation.

INPUTS
""")
        
        prompt_parts.append(f"A) ORIGINAL PROBLEM FILE (reference + helpers):\n```python\n{problem_code}\n```\n")
        
        plan_json = json.dumps(asdict(selected_plan), indent=2)
        prompt_parts.append(f"\nB) SELECTED PLAN JSON (you must follow this plan; if you must deviate, explain ONLY in code comments):\n```json\n{plan_json}\n```\n")
        
        if previous_attempt:
            prompt_parts.append(f"\nOPTIONAL: PRIOR FAILED ATTEMPT:\n```python\n{previous_attempt}\n```\n")
        
        if error_context:
            prompt_parts.append(f"\nOPTIONAL: ERROR_CONTEXT:\n{error_context}\n")
        
        if perf_context:
            prompt_parts.append(f"\nOPTIONAL: PERF_CONTEXT:\n{perf_context}\n")
        
        prompt_parts.append("""
IMPLEMENTATION GUIDELINES
- You may implement as 1 kernel or multiple kernels, but aim to minimize launches.
- If the plan includes schedule_search_space, pick a reasonable default and implement autotune if you can do so succinctly.
- Avoid common Triton pitfalls:
  * Do not use tl.broadcast on Python scalars. Use scalar constants directly (e.g., tl.maximum(x, 0.0)).
  * Use tl.load/tl.store masks for tails.
  * Keep BLOCK sizes power-of-two when possible.
- If attention pattern:
  * Prefer streaming/online softmax with m_i and l_i accumulators (FlashAttention-style).
  * Handle causal mask if needed.
- If RMSNorm/LN + GEMM:
  * If moving scaling to epilogue, implement row-wise scale applied to GEMM output.
  * If folding gamma into weights, do it in a pre-processing step ONLY if weights are constant; otherwise do it inside kernel carefully.

OUTPUT FORMAT
- Return only one fenced Python code block. No prose outside.
""")
        
        return "\n".join(prompt_parts)
    
    def _build_critic_prompt(
        self,
        selected_plan: OptimizationPlan,
        candidate_code: str,
    ) -> str:
        """Build the Critic agent prompt."""
        prompt_parts = []
        
        prompt_parts.append("""Audit the following candidate code for policy compliance and likely runtime issues.

HARD CHECKS
1) kernel_function must not call torch math ops (torch.relu, torch.matmul, torch.softmax, torch.nn, F.* etc).
   Torch is allowed only for allocations and inside run_tests reference path.
2) Shape consistency: tensor shapes used in tl.load/tl.store and pointer arithmetic must match plan's shapes.
3) Triton pitfalls: scalar broadcast misuse, missing masks, wrong strides, wrong grid, race conditions.
4) If the plan claims online softmax, check numerical stability and invariants (m_i, l_i).
5) Confirm run_tests prints 'PASS' and exits 0 on success.

INPUTS
""")
        
        plan_json = json.dumps(asdict(selected_plan), indent=2)
        prompt_parts.append(f"- SELECTED PLAN JSON:\n```json\n{plan_json}\n```\n")
        
        prompt_parts.append(f"\n- CANDIDATE CODE:\n```python\n{candidate_code}\n```\n")
        
        prompt_parts.append("""
OUTPUT JSON SCHEMA
{
  "verdict": "pass|fail",
  "critical_issues": [ "<string>", ... ],
  "noncritical_issues": [ "<string>", ... ],
  "suggested_patches": [
    { "target": "<function_or_line_hint>", "patch": "<textual instruction>" }
  ],
  "risk_level": "low|med|high"
}

Now output ONLY the JSON.
""")
        
        return "\n".join(prompt_parts)
    
    def _build_refiner_prompt(
        self,
        problem_code: str,
        selected_plan: OptimizationPlan,
        previous_code: str,
        error_context: str,
        perf_context: Optional[str],
    ) -> str:
        """Build the Refiner agent prompt."""
        prompt_parts = []
        
        prompt_parts.append("""You previously generated a Triton implementation but it failed correctness or performance targets.
You must fix it while keeping semantics identical to the PyTorch reference.

HARD REQUIREMENTS (same as before)
- Output ONE complete Python file as a single ```python block.
- kernel_function must be Triton-only for final computation (no torch math).
- run_tests must print PASS and exit(0) on success.
- Deterministic, no network, no extra deps.

WHAT CHANGED
You now have:
- ERROR_CONTEXT (compiler/runtime errors, stdout/stderr tail)
- PERF_CONTEXT (timing, launch count, suspected bottlenecks)

Use them to revise the implementation. If performance is poor, you may:
- Change fusion boundaries (merge or split kernels),
- Switch to a different algorithmic variant described in the plan family,
- Adjust schedule (BLOCK sizes, warps, stages),
- Reduce intermediate buffers / improve memory coalescing.

INPUTS
""")
        
        prompt_parts.append(f"A) ORIGINAL PROBLEM FILE:\n```python\n{problem_code}\n```\n")
        
        plan_json = json.dumps(asdict(selected_plan), indent=2)
        prompt_parts.append(f"\nB) TARGET PLAN (you may adjust within the same family if necessary; keep plan intent):\n```json\n{plan_json}\n```\n")
        
        prompt_parts.append(f"\nC) ERROR_CONTEXT:\n{error_context}\n")
        
        if perf_context:
            prompt_parts.append(f"\nD) PERF_CONTEXT:\n{perf_context}\n")
        
        prompt_parts.append(f"\nE) PREVIOUS ATTEMPT CODE:\n```python\n{previous_code}\n```\n")
        
        prompt_parts.append("""
ADDITIONAL RULES
- If you used tl.broadcast on scalars, remove it.
- If you see shape mismatch, fix pointer arithmetic and grid mapping first.
- If correctness fails for fp16/bf16, consider accumulation in fp32 where appropriate.
- If attention online softmax unstable, enforce log-sum-exp stability with running m_i and l_i.

OUTPUT FORMAT
Return only one fenced Python block with the corrected full file.
""")
        
        return "\n".join(prompt_parts)
    
    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON code block
        json_block_pattern = r"```(?:json)?\s*\n(.*?)```"
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        
        if matches:
            json_text = matches[-1].strip()
        else:
            # Try to find JSON object directly
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_text = text[first_brace:last_brace + 1]
            else:
                raise ValueError("No JSON found in response")
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nText: {json_text[:500]}")
    
    def _parse_planner_output(self, data: Dict[str, Any]) -> PlannerOutput:
        """Parse planner JSON output into PlannerOutput dataclass."""
        plans = []
        for plan_data in data.get("plans", []):
            plan = OptimizationPlan(
                plan_id=plan_data.get("plan_id", ""),
                tag=plan_data.get("tag", ""),
                high_level_idea=plan_data.get("high_level_idea", ""),
                subgraphs=plan_data.get("subgraphs", []),
                algebraic_transforms=plan_data.get("algebraic_transforms", []),
                algorithmic_variants=plan_data.get("algorithmic_variants", []),
                kernel_plan=plan_data.get("kernel_plan", {}),
                expected_tradeoffs=plan_data.get("expected_tradeoffs", {}),
                acceptance_tests=plan_data.get("acceptance_tests", []),
            )
            plans.append(plan)
        
        return PlannerOutput(
            problem_fingerprint=data.get("problem_fingerprint", ""),
            detected_patterns=data.get("detected_patterns", []),
            plans=plans,
            recommendations=data.get("recommendations", {}),
        )
    
    def _parse_critic_output(self, data: Dict[str, Any]) -> CriticOutput:
        """Parse critic JSON output into CriticOutput dataclass."""
        return CriticOutput(
            verdict=data.get("verdict", "fail"),
            critical_issues=data.get("critical_issues", []),
            noncritical_issues=data.get("noncritical_issues", []),
            suggested_patches=data.get("suggested_patches", []),
            risk_level=data.get("risk_level", "high"),
        )
