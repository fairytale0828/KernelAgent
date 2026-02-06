"""
E-Graph Algebraic Rewriting Engine for Fuser.

This module provides equality saturation-based optimization for subgraph fusion.
It applies mathematically rigorous algebraic transformations to generate
equivalent subgraph variants, then selects optimal implementations based on
a cost model.

Pipeline:
1. Parse subgraphs.json
2. Expand high-level ops (rms_norm, linear, etc.) to low-level algebra ops
   using torch.fx or AST parsing
3. Build E-Graph from expanded ops
4. Apply algebraic rewrite rules (commutativity, associativity, etc.)
5. Extract Top-K candidates using cost model
6. Output transformed_subgraphs.json

Usage:
    from Fuser.egraph import AlgebraicRewriter, RewriteConfig

    config = RewriteConfig(expand_ops=True, expand_norms=True)
    rewriter = AlgebraicRewriter(config)
    result = rewriter.rewrite("subgraphs.json", "output.json")
"""

from .ir import (
    Dtype,
    TensorMeta,
    OpNode,
    SubgraphIR,
    parse_subgraph_json,
    ir_to_json,
)
from .egraph import EGraph, EClass, ENode
from .cost_model import CostModel, IOAwareCostModel
from .extractor import Extractor
from .rewriter import AlgebraicRewriter, RewriteConfig, RewriteResult
from .rules import RuleSet, Rule
from .fx_expander import (
    FXExpander,
    ExpandedGraph,
    LowLevelOp,
    expand_subgraph_json,
)
from .pattern_match import (
    RewriteRule,
    EMatchEngine,
    ConstraintChecker,
    parse_pattern,
)
from .optimizer import (
    AlgebraicOptimizer,
    OptimizerConfig,
    OptimizationResult,
    optimize_subgraph,
    optimize_json_file,
)
from .rules.rule_library import (
    get_all_rules,
    get_rule_set,
    list_rule_sets,
    RuleCategory,
)
from .subgraph_aggregator import (
    SubgraphAggregator,
    FusionPattern,
    OptimizationHintExtractor,
    aggregate_subgraphs,
    aggregate_json_file,
    FUSION_PATTERNS,
)

__all__ = [
    # IR
    "Dtype",
    "TensorMeta",
    "OpNode",
    "SubgraphIR",
    "parse_subgraph_json",
    "ir_to_json",
    # E-Graph
    "EGraph",
    "EClass",
    "ENode",
    # Cost Model
    "CostModel",
    "IOAwareCostModel",
    # Extractor
    "Extractor",
    # Rewriter (旧版)
    "AlgebraicRewriter",
    "RewriteConfig",
    "RewriteResult",
    # Rules
    "RuleSet",
    "Rule",
    # FX Expander
    "FXExpander",
    "ExpandedGraph",
    "LowLevelOp",
    "expand_subgraph_json",
    # Pattern Matching (新版)
    "RewriteRule",
    "EMatchEngine",
    "ConstraintChecker",
    "parse_pattern",
    # Optimizer (新版)
    "AlgebraicOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "optimize_subgraph",
    "optimize_json_file",
    # Rule Library
    "get_all_rules",
    "get_rule_set",
    "list_rule_sets",
    "RuleCategory",
    # Subgraph Aggregator
    "SubgraphAggregator",
    "FusionPattern",
    "OptimizationHintExtractor",
    "aggregate_subgraphs",
    "aggregate_json_file",
    "FUSION_PATTERNS",
]
