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
Algorithm-level rewriter based on Mirage principles.
Rewrites subgraphs to enable algorithmic fusion.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
from .algorithm_analyzer import AlgorithmPattern


class AlgorithmRewriter:
    """Mirage-style algorithmic rewriter"""
    
    def __init__(self):
        self.rewrite_strategies = {
            "norm_gemm_fusion": self._rewrite_norm_gemm,
            "streaming_attention": self._rewrite_streaming_attention,
            "scale_absorption_backward": self._rewrite_scale_absorption,
            "scale_absorption_forward": self._rewrite_scale_absorption,
            "epilogue_fusion": self._rewrite_epilogue_fusion,
            "broadcast_to_sharing": self._rewrite_broadcast_sharing,
        }
    
    def apply_rewrites(
        self, 
        subgraphs: List[Dict], 
        patterns: List[AlgorithmPattern]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Apply algorithmic rewrites, return new subgraph list and metadata"""
        
        rewritten_subgraphs = []
        rewrite_metadata = {
            "applied_patterns": [],
            "eliminated_subgraphs": [],
            "new_subgraphs": []
        }
        
        # Sort by benefit, apply greedily
        patterns_sorted = sorted(patterns, key=lambda p: p.estimated_benefit, reverse=True)
        applied_ids = set()
        
        for pattern in patterns_sorted:
            # Check if already covered by another rewrite
            if any(sid in applied_ids for sid in pattern.subgraph_ids):
                continue
            
            # Apply rewrite strategy
            strategy_fn = self.rewrite_strategies.get(pattern.pattern_type)
            if strategy_fn:
                try:
                    new_sg = strategy_fn(subgraphs, pattern)
                    if new_sg:
                        rewritten_subgraphs.append(new_sg)
                        applied_ids.update(pattern.subgraph_ids)
                        rewrite_metadata["applied_patterns"].append({
                            "pattern": pattern.pattern_type,
                            "original_ids": pattern.subgraph_ids,
                            "new_id": new_sg["id"],
                            "estimated_benefit": pattern.estimated_benefit
                        })
                        rewrite_metadata["eliminated_subgraphs"].extend(pattern.subgraph_ids)
                        rewrite_metadata["new_subgraphs"].append(new_sg["id"])
                except Exception as e:
                    print(f"Warning: Failed to apply rewrite {pattern.pattern_type}: {e}")
                    continue
        
        # Add unrewritten subgraphs
        for sg in subgraphs:
            if sg["id"] not in applied_ids:
                rewritten_subgraphs.append(sg)
        
        return rewritten_subgraphs, rewrite_metadata
    
    def _rewrite_norm_gemm(self, subgraphs: List[Dict], pattern: AlgorithmPattern) -> Dict:
        """Rewrite Norm + GEMM as fused algorithm"""
        
        # Check if this is intra-subgraph fusion (single subgraph with multiple ops)
        is_intra_subgraph = pattern.fusion_opportunity.get("intra_subgraph", False)
        
        if is_intra_subgraph:
            # Case 1: Single subgraph containing both norm and gemm ops
            if len(pattern.subgraph_ids) != 1:
                return None
            
            orig_sg = self._find_subgraph(subgraphs, pattern.subgraph_ids[0])
            if not orig_sg:
                return None
            
            # Extract input shape (use 'inputs' if available, otherwise 'input_shape')
            input_shape = orig_sg.get("input_shape", [])
            if not input_shape and "inputs" in orig_sg:
                inputs = orig_sg.get("inputs", [])
                if isinstance(inputs, list) and len(inputs) > 0:
                    input_shape = inputs[0]
            
            # Construct fused subgraph description
            fused_sg = {
                "id": f"fused_norm_gemm_{orig_sg['id']}",
                "type": "algorithmic_fusion",
                "algorithm_type": "norm_gemm_fused",
                "data_layout": orig_sg.get("data_layout", "row_major"),
                "dtype": orig_sg.get("dtype", "float32"),
                
                # Input/output shapes
                "input_shape": input_shape,
                "output_shape": orig_sg.get("output_shape", []),
                
                # Fused operation description (algorithm-level)
                "ops": [
                    {
                        "op": "fused_norm_gemm",
                        "norm_type": pattern.fusion_opportunity.get("norm_type", "rms_norm"),
                        "gemm_shape": pattern.fusion_opportunity.get("gemm_shape", {}),
                        "algorithm_description": (
                            "Fuse normalization reduction into GEMM accumulation loop. "
                            "Compute mean(x^2) or mean/var while accumulating GEMM result. "
                            "Apply normalization in epilogue."
                        )
                    }
                ],
                
                # Weight information (from original subgraph)
                "weights_fused": orig_sg.get("weights_fused", {}),
                "weights_original": orig_sg.get("weights_original", {}),
                
                # Algorithm-level metadata
                "algorithm_metadata": {
                    "fusion_type": "reduction_into_main_loop",
                    "intermediate_eliminated": pattern.fusion_opportunity.get("intermediate_eliminated", []),
                    "memory_saved_bytes": self._estimate_memory_saved_single(orig_sg),
                    "expected_speedup": 1.5,
                    "intra_subgraph": True,
                    "original_ops_count": len(orig_sg.get("ops", []))
                },
                
                # Generation hints for KernelAgent
                "generation_hints": {
                    "triton_strategy": "single_kernel_with_running_reduction",
                    "key_optimizations": [
                        "Maintain running sum for normalization in registers",
                        "Fuse normalization scale into GEMM epilogue",
                        "Avoid materializing intermediate normalized tensor",
                        "Use shared memory efficiently for both GEMM and reduction"
                    ],
                    "reference_algorithm": "Similar to FlashAttention's running statistics",
                    "norm_type": pattern.fusion_opportunity.get("norm_type", "rms_norm")
                },
                
                # Original subgraph for reference
                "original_subgraph": orig_sg
            }
            
            return fused_sg
        
        else:
            # Case 2: Two separate subgraphs (cross-subgraph fusion)
            if len(pattern.subgraph_ids) != 2:
                return None
            
            norm_sg = self._find_subgraph(subgraphs, pattern.subgraph_ids[0])
            gemm_sg = self._find_subgraph(subgraphs, pattern.subgraph_ids[1])
            
            if not norm_sg or not gemm_sg:
                return None
            
            # Construct fused subgraph description
            fused_sg = {
                "id": f"fused_norm_gemm_{norm_sg['id']}_{gemm_sg['id']}",
                "type": "algorithmic_fusion",
                "algorithm_type": "norm_gemm_fused",
                "data_layout": gemm_sg.get("data_layout", "row_major"),
                "dtype": gemm_sg.get("dtype", "float32"),
                
                # Input/output shapes
                "input_shape": norm_sg.get("input_shape", []),
                "output_shape": gemm_sg.get("output_shape", []),
                
                # Fused operation description (algorithm-level)
                "ops": [
                    {
                        "op": "fused_norm_gemm",
                        "norm_type": pattern.fusion_opportunity.get("norm_type", "rms_norm"),
                        "gemm_shape": pattern.fusion_opportunity.get("gemm_shape", {}),
                        "algorithm_description": (
                            "Fuse normalization reduction into GEMM accumulation loop. "
                            "Compute mean(x^2) or mean/var while accumulating GEMM result. "
                            "Apply normalization in epilogue."
                        )
                    }
                ],
                
                # Weight information (merged)
                "weights_fused": {
                    **norm_sg.get("weights_fused", {}),
                    **gemm_sg.get("weights_fused", {})
                },
                
                # Algorithm-level metadata
                "algorithm_metadata": {
                    "fusion_type": "reduction_into_main_loop",
                    "intermediate_eliminated": pattern.fusion_opportunity.get("intermediate_eliminated", []),
                    "memory_saved_bytes": self._estimate_memory_saved(norm_sg, gemm_sg),
                    "expected_speedup": 1.5,
                    "intra_subgraph": False
                },
                
                # Generation hints for KernelAgent
                "generation_hints": {
                    "triton_strategy": "single_kernel_with_running_reduction",
                    "key_optimizations": [
                        "Maintain running sum for normalization in registers",
                        "Fuse normalization scale into GEMM epilogue",
                        "Avoid materializing intermediate normalized tensor",
                        "Use shared memory efficiently for both GEMM and reduction"
                    ],
                    "reference_algorithm": "Similar to FlashAttention's running statistics",
                    "norm_type": pattern.fusion_opportunity.get("norm_type", "rms_norm")
                },
                
                # Original subgraphs for reference
                "original_subgraphs": {
                    "norm": norm_sg,
                    "gemm": gemm_sg
                }
            }
            
            return fused_sg
    
    def _rewrite_streaming_attention(self, subgraphs: List[Dict], pattern: AlgorithmPattern) -> Dict:
        """Rewrite as streaming Attention algorithm"""
        qk_sg, softmax_sg, v_sg = [
            self._find_subgraph(subgraphs, sid) 
            for sid in pattern.subgraph_ids
        ]
        
        if not all([qk_sg, softmax_sg, v_sg]):
            return None
        
        fused_sg = {
            "id": f"streaming_attention_{qk_sg['id']}",
            "type": "algorithmic_fusion",
            "algorithm_type": "streaming_attention",
            "data_layout": "BHSD",  # Batch, Heads, Seq, Dim
            "dtype": qk_sg.get("dtype", "float32"),
            
            "input_shape": {
                "Q": qk_sg.get("input_shape", []),
                "K": qk_sg.get("inputs", [[]])[1] if "inputs" in qk_sg else None,
                "V": v_sg.get("inputs", [[]])[1] if "inputs" in v_sg else None
            },
            "output_shape": v_sg.get("output_shape", []),
            
            "ops": [
                {
                    "op": "streaming_attention",
                    "block_size_kv": pattern.fusion_opportunity.get("block_size", 64),
                    "algorithm_description": (
                        "Streaming attention with block-wise K/V processing. "
                        "Maintain running max and sum for online softmax. "
                        "Accumulate output incrementally without materializing full attention matrix."
                    )
                }
            ],
            
            "algorithm_metadata": {
                "fusion_type": "streaming_reduction",
                "block_size": pattern.fusion_opportunity.get("block_size", 64),
                "running_state": pattern.fusion_opportunity.get("running_state", ["max", "sum"]),
                "memory_complexity": "O(block_size) instead of O(seq_len^2)",
                "expected_speedup": 3.0
            },
            
            "generation_hints": {
                "triton_strategy": "block_wise_kv_iteration_with_running_state",
                "key_optimizations": [
                    "Tile K/V dimension into blocks",
                    "Maintain running max/sum in registers",
                    "Online softmax correction",
                    "Accumulate output on-the-fly"
                ],
                "reference_algorithm": "FlashAttention / FlashAttention-2"
            },
            
            "original_subgraphs": {
                "qk": qk_sg,
                "softmax": softmax_sg,
                "v": v_sg
            }
        }
        
        return fused_sg
    
    def _rewrite_epilogue_fusion(self, subgraphs: List[Dict], pattern: AlgorithmPattern) -> Dict:
        """Rewrite as GEMM with custom epilogue"""
        gemm_sg = self._find_subgraph(subgraphs, pattern.subgraph_ids[0])
        epilogue_sgs = [
            self._find_subgraph(subgraphs, sid) 
            for sid in pattern.subgraph_ids[1:]
        ]
        
        if not gemm_sg or not epilogue_sgs:
            return None
        
        fused_sg = {
            "id": f"gemm_with_epilogue_{gemm_sg['id']}",
            "type": "algorithmic_fusion",
            "algorithm_type": "gemm_custom_epilogue",
            "data_layout": gemm_sg.get("data_layout"),
            "dtype": gemm_sg.get("dtype"),
            
            "input_shape": gemm_sg.get("input_shape", []),
            "output_shape": epilogue_sgs[-1].get("output_shape", []),
            
            "ops": [
                {
                    "op": "gemm_with_custom_epilogue",
                    "gemm_shape": pattern.fusion_opportunity.get("gemm_shape", {}),
                    "epilogue_ops": pattern.fusion_opportunity.get("epilogue_ops", []),
                    "algorithm_description": (
                        f"GEMM with fused epilogue: {' -> '.join(pattern.fusion_opportunity.get('epilogue_ops', []))}. "
                        "All epilogue ops computed in registers before store."
                    )
                }
            ],
            
            "weights_fused": gemm_sg.get("weights_fused", {}),
            
            "algorithm_metadata": {
                "fusion_type": "epilogue_chain",
                "epilogue_length": len(epilogue_sgs),
                "memory_saved_bytes": sum(
                    self._estimate_tensor_size(sg.get("output_shape", [])) 
                    for sg in epilogue_sgs[:-1]
                ),
                "expected_speedup": 1.3
            },
            
            "generation_hints": {
                "triton_strategy": "gemm_with_register_epilogue",
                "key_optimizations": [
                    "Compute epilogue in registers after GEMM accumulation",
                    "Avoid intermediate stores",
                    "Vectorize epilogue ops if possible"
                ]
            },
            
            "original_subgraphs": {
                "gemm": gemm_sg,
                "epilogue": epilogue_sgs
            }
        }
        
        return fused_sg
    
    def _rewrite_scale_absorption(self, subgraphs: List[Dict], pattern: AlgorithmPattern) -> Dict:
        """Rewrite scale absorption (simplified)"""
        # For now, return None to skip this pattern
        # Can be implemented later if needed
        return None
    
    def _rewrite_broadcast_sharing(self, subgraphs: List[Dict], pattern: AlgorithmPattern) -> Dict:
        """Rewrite broadcast to sharing (simplified)"""
        # For now, return None to skip this pattern
        return None
    
    # Helper methods
    def _find_subgraph(self, subgraphs: List[Dict], sg_id: str) -> Optional[Dict]:
        """Find subgraph by ID"""
        for sg in subgraphs:
            if sg.get("id") == sg_id:
                return sg
        return None
    
    def _estimate_memory_saved(self, sg1: Dict, sg2: Dict) -> int:
        """Estimate memory saved by fusion (bytes) for cross-subgraph fusion"""
        # Simplified: estimate based on intermediate tensor size
        intermediate_shape = sg1.get("output_shape", [])
        dtype = sg1.get("dtype", "float32")
        
        if not intermediate_shape:
            return 0
        
        # Calculate tensor size
        size = 1
        for dim in intermediate_shape:
            if isinstance(dim, int):
                size *= dim
        
        # Bytes per element
        bytes_per_elem = 4 if dtype == "float32" else 2  # float32 or float16
        
        return size * bytes_per_elem
    
    def _estimate_memory_saved_single(self, sg: Dict) -> int:
        """Estimate memory saved by fusion (bytes) for intra-subgraph fusion"""
        # For intra-subgraph fusion, we save the intermediate tensor between ops
        # Use output_shape as an estimate of the intermediate tensor size
        intermediate_shape = sg.get("output_shape", [])
        dtype = sg.get("dtype", "float32")
        
        if not intermediate_shape:
            return 0
        
        # Calculate tensor size
        size = 1
        for dim in intermediate_shape:
            if isinstance(dim, int):
                size *= dim
        
        # Bytes per element
        bytes_per_elem = 4 if dtype == "float32" else 2  # float32 or float16
        
        # For norm+gemm, the intermediate is the normalized tensor
        # which has the same shape as the input
        input_shape = sg.get("input_shape", [])
        if not input_shape and "inputs" in sg:
            inputs = sg.get("inputs", [])
            if isinstance(inputs, list) and len(inputs) > 0:
                input_shape = inputs[0]
        
        if input_shape:
            size = 1
            for dim in input_shape:
                if isinstance(dim, int):
                    size *= dim
            return size * bytes_per_elem
        
        return size * bytes_per_elem
    
    def _estimate_tensor_size(self, shape: List) -> int:
        """Estimate tensor size in bytes"""
        if not shape:
            return 0
        
        size = 1
        for dim in shape:
            if isinstance(dim, int):
                size *= dim
        
        return size * 4  # Assume float32
