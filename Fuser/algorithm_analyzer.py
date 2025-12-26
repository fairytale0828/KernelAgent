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
Algorithm-level pattern detection for Mirage-style optimizations.
Analyzes subgraphs to identify fusion opportunities.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class AlgorithmPattern:
    """Detected algorithmic optimization pattern"""
    pattern_type: str  # "norm_gemm", "streaming_attention", "scale_fusion", etc.
    subgraph_ids: List[str]  # Involved subgraph IDs
    fusion_opportunity: Dict[str, Any]  # Fusion opportunity description
    estimated_benefit: float  # Estimated benefit (0-1)
    rewrite_strategy: str  # Rewrite strategy description


class AlgorithmPatternDetector:
    """Detects Mirage-style algorithmic optimization opportunities"""
    
    def __init__(self):
        self.patterns = [
            self._detect_norm_gemm_pattern,
            self._detect_streaming_attention_pattern,
            self._detect_scale_absorption_pattern,
            self._detect_broadcast_sharing_pattern,
            self._detect_epilogue_fusion_pattern,
        ]
    
    def analyze(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
        """Analyze subgraphs and identify algorithmic optimization patterns"""
        detected_patterns = []
        
        for detector in self.patterns:
            try:
                patterns = detector(subgraphs)
                detected_patterns.extend(patterns)
            except Exception as e:
                print(f"Warning: Pattern detector failed: {e}")
                continue
        
        # Sort by benefit
        detected_patterns.sort(key=lambda p: p.estimated_benefit, reverse=True)
        return detected_patterns
    
    def _detect_norm_gemm_pattern(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
        """Detect RMSNorm/LayerNorm + GEMM pattern"""
        patterns = []
        
        # Case 1: Check for norm + gemm within a single subgraph (NEW!)
        for sg in subgraphs:
            ops_list = sg.get("ops", [])
            
            if len(ops_list) >= 2:
                # Extract op names correctly (handle dict objects)
                op_names = []
                for op in ops_list:
                    if isinstance(op, dict):
                        op_names.append(op.get("op", ""))
                    elif isinstance(op, str):
                        op_names.append(op)
                    else:
                        op_names.append(str(op))
                
                # Look for norm followed by gemm
                for i in range(len(op_names) - 1):
                    curr_op = op_names[i].lower()
                    next_op = op_names[i + 1].lower()
                    
                    if self._is_norm_op(curr_op) and self._is_gemm_op(next_op):
                        norm_type = self._get_norm_type_from_op(curr_op)
                        patterns.append(AlgorithmPattern(
                            pattern_type="norm_gemm_fusion",
                            subgraph_ids=[sg["id"]],
                            fusion_opportunity={
                                "norm_type": norm_type,
                                "gemm_shape": self._extract_gemm_shape(sg),
                                "fusion_point": "reduction_into_gemm_loop",
                                "intermediate_eliminated": sg.get("output_shape", []),
                                "intra_subgraph": True,  # Mark as within-subgraph fusion
                                "norm_op_index": i,
                                "gemm_op_index": i + 1
                            },
                            estimated_benefit=0.85,  # High benefit
                            rewrite_strategy="fuse_norm_reduction_into_gemm_accumulation"
                        ))
        
        # Case 2: Check for norm + gemm across two separate subgraphs (original logic)
        for i, sg in enumerate(subgraphs):
            ops_list = sg.get("ops", [])
            
            # Extract op names correctly
            op_names = []
            for op in ops_list:
                if isinstance(op, dict):
                    op_names.append(op.get("op", ""))
                elif isinstance(op, str):
                    op_names.append(op)
                else:
                    op_names.append(str(op))
            
            # Find norm followed by gemm in next subgraph
            if self._has_norm(op_names) and i + 1 < len(subgraphs):
                next_sg = subgraphs[i + 1]
                next_ops_list = next_sg.get("ops", [])
                
                # Extract next op names
                next_op_names = []
                for op in next_ops_list:
                    if isinstance(op, dict):
                        next_op_names.append(op.get("op", ""))
                    elif isinstance(op, str):
                        next_op_names.append(op)
                    else:
                        next_op_names.append(str(op))
                
                if self._has_gemm(next_op_names):
                    # Check data dependency
                    if self._is_producer_consumer(sg, next_sg):
                        norm_type = self._get_norm_type(op_names)
                        patterns.append(AlgorithmPattern(
                            pattern_type="norm_gemm_fusion",
                            subgraph_ids=[sg["id"], next_sg["id"]],
                            fusion_opportunity={
                                "norm_type": norm_type,
                                "gemm_shape": self._extract_gemm_shape(next_sg),
                                "fusion_point": "reduction_into_gemm_loop",
                                "intermediate_eliminated": sg.get("output_shape", []),
                                "intra_subgraph": False  # Mark as cross-subgraph fusion
                            },
                            estimated_benefit=0.85,
                            rewrite_strategy="fuse_norm_reduction_into_gemm_accumulation"
                        ))
        
        return patterns
    
    def _detect_streaming_attention_pattern(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
        """Detect streamable Attention pattern (QK^T -> Softmax -> @V)"""
        patterns = []
        
        for i in range(len(subgraphs) - 2):
            # Look for QK^T -> Softmax -> @V sequence
            if self._is_attention_sequence(subgraphs[i:i+3]):
                qk_sg, softmax_sg, v_sg = subgraphs[i:i+3]
                
                # Check if K/V dimension can be streamed
                if self._can_stream_kv(qk_sg, v_sg):
                    patterns.append(AlgorithmPattern(
                        pattern_type="streaming_attention",
                        subgraph_ids=[sg["id"] for sg in [qk_sg, softmax_sg, v_sg]],
                        fusion_opportunity={
                            "qk_shape": qk_sg.get("output_shape", []),
                            "streaming_dim": self._get_streaming_dim(qk_sg),
                            "block_size": self._suggest_block_size(qk_sg),
                            "running_state": ["max", "sum"]
                        },
                        estimated_benefit=0.95,  # Very high benefit (FlashAttention-level)
                        rewrite_strategy="streaming_softmax_with_running_state"
                    ))
        
        return patterns
    
    def _detect_scale_absorption_pattern(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
        """Detect scale/bias operations that can be absorbed"""
        patterns = []
        
        for i, sg in enumerate(subgraphs):
            ops_list = sg.get("ops", [])
            
            if self._has_scale_or_bias(ops_list):
                # Look backward for GEMM
                if i > 0:
                    prev_ops_list = subgraphs[i-1].get("ops", [])
                    prev_op_names = [op.get("op", "") if isinstance(op, dict) else str(op) for op in prev_ops_list]
                    
                    if self._has_gemm(prev_op_names):
                        patterns.append(AlgorithmPattern(
                            pattern_type="scale_absorption_backward",
                            subgraph_ids=[subgraphs[i-1]["id"], sg["id"]],
                            fusion_opportunity={
                                "absorption_target": "gemm_weight",
                                "scale_value": self._extract_scale(sg),
                                "can_fold_compile_time": self._is_static_scale(sg)
                            },
                            estimated_benefit=0.6,
                            rewrite_strategy="absorb_scale_into_weight_or_epilogue"
                        ))
                
                # Look forward for GEMM
                if i + 1 < len(subgraphs):
                    next_ops_list = subgraphs[i+1].get("ops", [])
                    next_op_names = [op.get("op", "") if isinstance(op, dict) else str(op) for op in next_ops_list]
                    
                    if self._has_gemm(next_op_names):
                        patterns.append(AlgorithmPattern(
                            pattern_type="scale_absorption_forward",
                            subgraph_ids=[sg["id"], subgraphs[i+1]["id"]],
                            fusion_opportunity={
                                "absorption_target": "gemm_epilogue",
                                "scale_value": self._extract_scale(sg)
                            },
                            estimated_benefit=0.65,
                            rewrite_strategy="absorb_scale_into_epilogue"
                        ))
        
        return patterns
    
    def _detect_broadcast_sharing_pattern(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
        """Detect GQA-style broadcast/repeat that can be converted to sharing"""
        patterns = []
        
        for sg in subgraphs:
            ops = sg.get("ops", [])
            
            for op in ops:
                if op.get("op") in ["repeat", "broadcast", "expand"]:
                    repeat_factor = self._get_repeat_factor(op)
                    if repeat_factor > 1:
                        patterns.append(AlgorithmPattern(
                            pattern_type="broadcast_to_sharing",
                            subgraph_ids=[sg["id"]],
                            fusion_opportunity={
                                "repeat_factor": repeat_factor,
                                "shared_dim": op.get("dim"),
                                "original_shape": sg.get("input_shape", []),
                                "expanded_shape": sg.get("output_shape", [])
                            },
                            estimated_benefit=0.7,
                            rewrite_strategy="replace_broadcast_with_block_sharing"
                        ))
        
        return patterns
    
    def _detect_epilogue_fusion_pattern(self, subgraphs: List[Dict]) -> List[AlgorithmPattern]:
        """Detect pointwise chains that can be fused into GEMM epilogue"""
        patterns = []
        
        for i, sg in enumerate(subgraphs):
            ops_list = sg.get("ops", [])
            op_names = [op.get("op", "") if isinstance(op, dict) else str(op) for op in ops_list]
            
            if not self._has_gemm(op_names):
                continue
            
            # Look forward for pointwise operation chain
            epilogue_chain = []
            j = i + 1
            while j < len(subgraphs) and self._is_pointwise(subgraphs[j]):
                epilogue_chain.append(subgraphs[j])
                j += 1
                if len(epilogue_chain) >= 5:  # Max 5 ops
                    break
            
            if len(epilogue_chain) >= 2:  # At least 2 ops worth it
                epilogue_ops = []
                for e in epilogue_chain:
                    e_ops = e.get("ops", [])
                    for op in e_ops:
                        if isinstance(op, dict):
                            epilogue_ops.append(op.get("op", ""))
                        else:
                            epilogue_ops.append(str(op))
                
                patterns.append(AlgorithmPattern(
                    pattern_type="epilogue_fusion",
                    subgraph_ids=[sg["id"]] + [e["id"] for e in epilogue_chain],
                    fusion_opportunity={
                        "gemm_id": sg["id"],
                        "epilogue_ops": epilogue_ops,
                        "chain_length": len(epilogue_chain)
                    },
                    estimated_benefit=0.75,
                    rewrite_strategy="fuse_pointwise_chain_into_gemm_epilogue"
                ))
        
        return patterns
    
    # Helper methods
    def _has_norm(self, ops: List[str]) -> bool:
        norm_ops = ["layer_norm", "rms_norm", "group_norm", "batch_norm", "rmsnorm", "layernorm"]
        return any(op.lower() in norm_ops for op in ops if op)
    
    def _has_gemm(self, ops: List[str]) -> bool:
        gemm_ops = ["gemm", "matmul", "linear", "bmm"]
        return any(op.lower() in gemm_ops for op in ops if op)
    
    def _is_norm_op(self, op_name: str) -> bool:
        """Check if an operation is a normalization operation"""
        if not op_name:
            return False
        norm_ops = ["layer_norm", "rms_norm", "group_norm", "batch_norm", "rmsnorm", "layernorm"]
        return op_name.lower() in norm_ops
    
    def _is_gemm_op(self, op_name: str) -> bool:
        """Check if an operation is a GEMM operation"""
        if not op_name:
            return False
        gemm_ops = ["gemm", "matmul", "linear", "bmm"]
        return op_name.lower() in gemm_ops
    
    def _get_norm_type_from_op(self, op_name: str) -> str:
        """Get norm type from operation name"""
        if not op_name:
            return "unknown_norm"
        op_lower = op_name.lower()
        if "rms" in op_lower:
            return "rms_norm"
        elif "layer" in op_lower:
            return "layer_norm"
        elif "group" in op_lower:
            return "group_norm"
        elif "batch" in op_lower:
            return "batch_norm"
        return "unknown_norm"
    
    def _is_pointwise(self, sg: Dict) -> bool:
        ops_list = sg.get("ops", [])
        op_names = []
        for op in ops_list:
            if isinstance(op, dict):
                op_names.append(op.get("op", "").lower())
            elif isinstance(op, str):
                op_names.append(op.lower())
            else:
                op_names.append(str(op).lower())
        
        pointwise_ops = ["relu", "gelu", "sigmoid", "tanh", "add", "mul", "div", "sub", "silu"]
        return all(op in pointwise_ops for op in op_names if op) and len(op_names) > 0
    
    def _get_norm_type(self, ops: List[str]) -> str:
        for op in ops:
            op_lower = op.lower()
            if "rms" in op_lower:
                return "rms_norm"
            elif "layer" in op_lower:
                return "layer_norm"
            elif "group" in op_lower:
                return "group_norm"
            elif "batch" in op_lower:
                return "batch_norm"
        return "unknown_norm"
    
    def _extract_gemm_shape(self, sg: Dict) -> Dict[str, Any]:
        input_shape = sg.get("input_shape", [])
        output_shape = sg.get("output_shape", [])
        weights = sg.get("weights_fused", {})
        
        return {
            "input": input_shape,
            "output": output_shape,
            "weight": weights.get("weight", weights.get("linear_weight", []))
        }
    
    def _is_producer_consumer(self, sg1: Dict, sg2: Dict) -> bool:
        """Check if sg1's output feeds into sg2's input"""
        out_shape = sg1.get("output_shape", [])
        in_shape = sg2.get("input_shape", [])
        
        # Simple shape matching
        if not out_shape or not in_shape:
            return False
        
        return out_shape == in_shape
    
    def _is_attention_sequence(self, subgraphs: List[Dict]) -> bool:
        """Check if subgraphs form QK^T -> Softmax -> @V pattern"""
        if len(subgraphs) < 3:
            return False
        
        # Extract op names safely
        def get_op_names(sg):
            ops_list = sg.get("ops", [])
            names = []
            for op in ops_list:
                if isinstance(op, dict):
                    names.append(op.get("op", "").lower())
                elif isinstance(op, str):
                    names.append(op.lower())
                else:
                    names.append(str(op).lower())
            return names
        
        ops1 = get_op_names(subgraphs[0])
        ops2 = get_op_names(subgraphs[1])
        ops3 = get_op_names(subgraphs[2])
        
        has_matmul_1 = any("matmul" in op or "bmm" in op for op in ops1)
        has_softmax = any("softmax" in op for op in ops2)
        has_matmul_2 = any("matmul" in op or "bmm" in op for op in ops3)
        
        return has_matmul_1 and has_softmax and has_matmul_2
    
    def _can_stream_kv(self, qk_sg: Dict, v_sg: Dict) -> bool:
        """Check if K/V dimension can be streamed"""
        # Simplified check: if sequence dimension is large enough
        qk_shape = qk_sg.get("output_shape", [])
        if len(qk_shape) >= 2:
            seq_len = qk_shape[-1] if isinstance(qk_shape[-1], int) else 512
            return seq_len >= 128  # Worth streaming if seq_len >= 128
        return False
    
    def _get_streaming_dim(self, sg: Dict) -> int:
        """Get the dimension to stream over"""
        return -1  # Typically the last dimension (K/V sequence length)
    
    def _suggest_block_size(self, sg: Dict) -> int:
        """Suggest block size for streaming"""
        shape = sg.get("output_shape", [])
        if len(shape) >= 2:
            seq_len = shape[-1] if isinstance(shape[-1], int) else 512
            # Suggest block size as power of 2, typically 64 or 128
            if seq_len >= 512:
                return 128
            elif seq_len >= 256:
                return 64
        return 64
    
    def _has_scale_or_bias(self, ops: List[Dict]) -> bool:
        """Check if ops contain scale or bias operations"""
        for op in ops:
            op_name = op.get("op", "").lower()
            if op_name in ["mul", "div", "add", "sub"]:
                return True
        return False
    
    def _extract_scale(self, sg: Dict) -> Any:
        """Extract scale value from subgraph"""
        # Simplified: return placeholder
        return {"type": "dynamic", "shape": sg.get("output_shape", [])}
    
    def _is_static_scale(self, sg: Dict) -> bool:
        """Check if scale is compile-time constant"""
        # Simplified: assume dynamic for now
        return False
    
    def _get_repeat_factor(self, op: Dict) -> int:
        """Get repeat factor from repeat/broadcast op"""
        return op.get("repeat_factor", op.get("repeats", 1))
