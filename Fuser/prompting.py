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
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

# Prompt rendering for the Fuse orchestrator
# - Deterministic, stateless per iteration
# - Four wording variants (0..3)
# - System + User messages only (Responses API input schema)


VARIANT_WORDINGS: tuple[str, str, str, str, str] = (
    "Rewrite the model into fusable subgraph modules with explicit input/output shapes.",
    "Refactor into fusion-friendly submodules and specify exact tensor shapes.",
    "Decompose into subgraphs suitable for fusion and document all shape contracts.",
    "Split into fusable modules and clearly state each input/output shape contract.",
    "Each fused subgraph must be its own nn.Module (no inline nn.* ops at top level).",
)

BASE_DEVELOPER_PROMPT = (
    "You are an expert PyTorch engineer focused on inference-only graph fusion.\n\n"
    "Hard requirements:\n"
    "- Return ONE runnable Python file, fenced as a single ```python block.\n"
    "- Each fused subgraph must be represented by its own nn.Module class with a clearly documented forward; do not leave raw nn.* ops inline in the top-level Model.\n"
    "- Include a function run_tests() that validates numerical equivalence to the original using helpers in the problem file. "
    "On success, run_tests() must print 'PASS' and exit(0).\n"
    "- If you cannot implement run_tests(), then at minimum print the exact sentinel ALL_TESTS_PASSED and exit(0) when tests succeed.\n"
    "- No network or file I/O outside the current directory. Avoid extra dependencies.\n"
    "- Deterministic: set seeds where relevant.\n\n"
    "Fusion guidance:\n"
    "- You may apply algebraic rearrangements that are strictly equivalent for the given shapes/dtypes (associativity, distributivity, scalar factor moves). Only use transformations that preserve numerical semantics for the stated precision.\n"
    "- Prefer fusion strategies that reduce intermediate tensors and kernel launches; document shape contracts for every subgraph.\n"
    "- Detect scaled dot-product attention patterns and fuse the full block (QKV linears, splits/reshapes, scaled QK^T, masking, activation/gating, applying V, head merge) into a single attention subgraph whenever feasible.\n"
    "- Only decompose attention into smaller subgraphs when you are certain fusion is impossible.\n"
    "- If you use a non-obvious algebraic rearrangement, include a short comment in the code explaining the equivalence.\n\n"
    "Search guidance:\n"
    "- If ERROR_CONTEXT is present, revise the fusion strategy or subgraph boundaries, not just constants.\n"
    "- If no ERROR_CONTEXT, still prefer the simplest valid fusion that minimizes intermediates.\n\n"
    "Iteration contract:\n"
    "- On each attempt, re-emit the entire single-file solution.\n"
    "- When ERROR_CONTEXT is provided, carefully analyze and fix issues, then re-emit the whole file.\n"
)

SYSTEM_PROMPT = "Return a single runnable Python file only."


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    extras: dict[str, Any]


def _variant_line(idx: int) -> str:
    i = idx % len(VARIANT_WORDINGS)
    return VARIANT_WORDINGS[i]


def _detect_focus(problem_file_content: str) -> list[str]:
    text = problem_file_content.lower()
    focus: list[str] = []
    if "attention" in text or "scaled_dot_product" in text or "softmax" in text:
        focus.append("If attention is present, prefer a single fused attention subgraph.")
    if "rmsnorm" in text or "rms_norm" in text:
        focus.append(
            "If RMSNorm is present, consider moving scalar factors when algebraically equivalent."
        )
    if "lora" in text or "low_rank" in text:
        focus.append(
            "If LoRA is present, consider fusing linear + low-rank updates when equivalent."
        )
    return focus


def _format_focus_hints(problem_file_content: str) -> str:
    hints = _detect_focus(problem_file_content)
    if not hints:
        return ""
    lines = ["FUSION_FOCUS:"]
    lines.extend(f"- {h}" for h in hints)
    return "\n".join(lines)


def build_user_prompt(
    attempt_index: int,
    problem_file_content: str,
    error_context: Optional[str],
    variant_index: int,
) -> str:
    parts: list[str] = []
    parts.append(_variant_line(variant_index))
    parts.append("")
    parts.append(BASE_DEVELOPER_PROMPT)
    parts.append("")
    focus = _format_focus_hints(problem_file_content)
    if focus:
        parts.append(focus)
        parts.append("")
    parts.append(f"ATTEMPT: {attempt_index}")
    if error_context:
        parts.append("")
        parts.append("ERROR_CONTEXT:")
        parts.append(error_context.strip())
    parts.append("")
    parts.append("PROBLEM_FILE_CONTENT:")
    parts.append(problem_file_content)
    return "\n".join(parts)


def render_prompt(
    problem_path: Path,
    variant_index: int,
    attempt_index: int,
    error_context: Optional[str],
    enable_reasoning_extras: bool,
    seed: Optional[int] = None,
    model_name: Optional[str] = None,
) -> RenderedPrompt:
    """Render system+user prompts and extras for the Responses API (deterministic)."""
    content = problem_path.read_text(encoding="utf-8")
    user = build_user_prompt(
        attempt_index=attempt_index,
        problem_file_content=content,
        error_context=error_context,
        variant_index=variant_index,
    )
    extras: dict[str, Any] = {}
    if seed is not None:
        extras["seed"] = seed
    if enable_reasoning_extras:
        # Use high reasoning effort for GPT-5 per policy
        extras["reasoning"] = {"effort": "high"}
        # Align with Responses API text options for clearer outputs
        text_options: dict[str, Any] = {"format": {"type": "text"}}
        if model_name:
            if model_name.startswith("gpt-5"):
                text_options["verbosity"] = "high"
            elif model_name.startswith("o4-mini"):
                text_options["verbosity"] = "medium"
        extras["text"] = text_options
    return RenderedPrompt(system=SYSTEM_PROMPT, user=user, extras=extras)
