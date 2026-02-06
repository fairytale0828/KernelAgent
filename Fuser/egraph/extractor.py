"""
最优表达式提取器

从 E-Graph 中提取最优表达式，支持：
- 贪心提取
- 动态规划提取
- Top-K 提取
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .cost_model import Cost, CostModel, IOAwareCostModel

if TYPE_CHECKING:
    from .egraph import EGraph, ENode, EClass


@dataclass
class ExtractedExpr:
    """提取的表达式"""
    root_eclass: int
    enode_choices: Dict[int, "ENode"]  # eclass_id -> chosen ENode
    cost: Cost
    proof: List[str]  # 变换证明轨迹

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "root_eclass": self.root_eclass,
            "cost": self.cost.value,
            "cost_breakdown": self.cost.breakdown,
            "proof": self.proof,
        }


class Extractor:
    """
    最优表达式提取器

    使用动态规划从 E-Graph 中提取代价最小的表达式
    """

    def __init__(self, egraph: "EGraph", cost_model: Optional[CostModel] = None):
        self.egraph = egraph
        self.cost_model = cost_model or IOAwareCostModel()

        # 缓存
        self._best_cost: Dict[int, Cost] = {}
        self._best_node: Dict[int, "ENode"] = {}

    def extract(self, root_eclass: int) -> ExtractedExpr:
        """
        提取最优表达式

        Args:
            root_eclass: 根等价类 ID

        Returns:
            提取的最优表达式
        """
        # 清空缓存
        self._best_cost.clear()
        self._best_node.clear()

        # 动态规划计算最优代价
        self._compute_best(root_eclass)

        # 回溯构建表达式
        choices = {}
        proof = []
        self._backtrack(root_eclass, choices, proof)

        return ExtractedExpr(
            root_eclass=root_eclass,
            enode_choices=choices,
            cost=self._best_cost.get(root_eclass, Cost.infinity()),
            proof=proof,
        )

    def _compute_best(self, eclass_id: int) -> Cost:
        """动态规划计算最优代价"""
        eclass_id = self.egraph.find(eclass_id)

        # 检查缓存
        if eclass_id in self._best_cost:
            return self._best_cost[eclass_id]

        eclass = self.egraph.get_eclass(eclass_id)
        if eclass is None:
            return Cost.infinity()

        best_cost = Cost.infinity()
        best_node = None

        for enode in eclass.nodes:
            # 递归计算子节点代价
            child_costs = []
            valid = True
            for child_id in enode.children:
                child_cost = self._compute_best(child_id)
                if child_cost.value == float('inf'):
                    valid = False
                    break
                child_costs.append(child_cost)

            if not valid:
                continue

            # 计算当前节点代价
            node_cost = self.cost_model.enode_cost(enode, child_costs)

            if node_cost < best_cost:
                best_cost = node_cost
                best_node = enode

        self._best_cost[eclass_id] = best_cost
        if best_node is not None:
            self._best_node[eclass_id] = best_node

        return best_cost

    def _backtrack(
        self,
        eclass_id: int,
        choices: Dict[int, "ENode"],
        proof: List[str]
    ) -> None:
        """回溯构建表达式"""
        eclass_id = self.egraph.find(eclass_id)

        if eclass_id in choices:
            return

        enode = self._best_node.get(eclass_id)
        if enode is None:
            return

        choices[eclass_id] = enode

        # 记录选择
        proof.append(
            f"eclass_{eclass_id}: {enode.op}({', '.join(f'eclass_{c}' for c in enode.children)})")

        # 递归处理子节点
        for child_id in enode.children:
            self._backtrack(child_id, choices, proof)

    def extract_top_k(self, root_eclass: int, k: int = 5) -> List[ExtractedExpr]:
        """
        提取 Top-K 最优表达式

        使用 beam search 或 k-best 算法

        Args:
            root_eclass: 根等价类 ID
            k: 返回的表达式数量

        Returns:
            Top-K 表达式列表
        """
        # 简化实现：使用不同的随机选择生成多个候选
        results = []

        # 首先提取最优
        best = self.extract(root_eclass)
        results.append(best)

        # 尝试生成其他候选
        root_id = self.egraph.find(root_eclass)
        eclass = self.egraph.get_eclass(root_id)

        if eclass is None:
            return results

        # 对于每个可选的根节点，生成一个候选
        seen_signatures = {self._expr_signature(best.enode_choices)}

        for enode in eclass.nodes:
            if len(results) >= k:
                break

            # 尝试以这个节点为根
            candidate = self._extract_with_root(root_id, enode)
            if candidate is None:
                continue

            sig = self._expr_signature(candidate.enode_choices)
            if sig in seen_signatures:
                continue

            seen_signatures.add(sig)
            results.append(candidate)

        # 按代价排序
        results.sort(key=lambda x: x.cost.value)

        return results[:k]

    def _extract_with_root(
        self,
        root_id: int,
        root_node: "ENode"
    ) -> Optional[ExtractedExpr]:
        """以指定节点为根提取表达式"""
        choices = {root_id: root_node}
        proof = [f"eclass_{root_id}: {root_node.op}"]

        # 计算子节点代价并选择
        child_costs = []
        for child_id in root_node.children:
            child_id = self.egraph.find(child_id)

            # 使用已缓存的最优选择
            if child_id in self._best_node:
                self._backtrack(child_id, choices, proof)
                child_costs.append(self._best_cost.get(
                    child_id, Cost.infinity()))
            else:
                return None

        # 计算总代价
        total_cost = self.cost_model.enode_cost(root_node, child_costs)

        return ExtractedExpr(
            root_eclass=root_id,
            enode_choices=choices,
            cost=total_cost,
            proof=proof,
        )

    def _expr_signature(self, choices: Dict[int, "ENode"]) -> str:
        """计算表达式签名用于去重"""
        parts = []
        for eclass_id in sorted(choices.keys()):
            enode = choices[eclass_id]
            parts.append(f"{eclass_id}:{enode.op}")
        return "|".join(parts)


class KBestExtractor:
    """
    K-Best 提取器

    使用更精确的算法提取 Top-K 表达式
    """

    def __init__(self, egraph: "EGraph", cost_model: Optional[CostModel] = None):
        self.egraph = egraph
        self.cost_model = cost_model or IOAwareCostModel()

    def extract(self, root_eclass: int, k: int = 5) -> List[ExtractedExpr]:
        """
        提取 K-Best 表达式

        使用优先队列和懒惰展开
        """
        root_id = self.egraph.find(root_eclass)

        # 优先队列: (cost, expr_state)
        heap: List[Tuple[float, int, Dict[int, "ENode"], List[str]]] = []
        results: List[ExtractedExpr] = []
        seen: Set[str] = set()

        # 初始化：添加根节点的所有可能选择
        eclass = self.egraph.get_eclass(root_id)
        if eclass is None:
            return results

        counter = 0
        for enode in eclass.nodes:
            # 估计代价 (使用启发式)
            est_cost = self._estimate_cost(enode)
            heapq.heappush(heap, (est_cost, counter, {root_id: enode}, []))
            counter += 1

        while heap and len(results) < k:
            cost, _, choices, proof = heapq.heappop(heap)

            # 检查是否完整
            is_complete, missing = self._check_complete(choices)

            if is_complete:
                sig = self._signature(choices)
                if sig not in seen:
                    seen.add(sig)
                    # 计算精确代价
                    exact_cost = self._compute_exact_cost(choices)
                    results.append(ExtractedExpr(
                        root_eclass=root_id,
                        enode_choices=choices,
                        cost=exact_cost,
                        proof=proof,
                    ))
            else:
                # 展开缺失的等价类
                for eclass_id in missing:
                    ec = self.egraph.get_eclass(eclass_id)
                    if ec is None:
                        continue

                    for enode in ec.nodes:
                        new_choices = dict(choices)
                        new_choices[eclass_id] = enode
                        new_proof = proof + [f"eclass_{eclass_id}: {enode.op}"]
                        est = self._estimate_cost_partial(new_choices)
                        heapq.heappush(
                            heap, (est, counter, new_choices, new_proof))
                        counter += 1

                    break  # 一次只展开一个

        return results

    def _check_complete(
        self,
        choices: Dict[int, "ENode"]
    ) -> Tuple[bool, List[int]]:
        """检查表达式是否完整，自动添加叶子节点"""
        missing = []
        visited = set()

        def check_node(eclass_id: int):
            eclass_id = self.egraph.find(eclass_id)
            if eclass_id in visited:
                return
            visited.add(eclass_id)

            if eclass_id not in choices:
                # 检查是否是叶子节点（input, weight, const）或只有一个选择
                ec = self.egraph.get_eclass(eclass_id)
                if ec and ec.nodes:
                    # 优先选择叶子节点
                    for enode in ec.nodes:
                        if enode.op in ("input", "weight", "const", "constant"):
                            choices[eclass_id] = enode
                            return  # 叶子节点没有子节点

                    # 如果只有一个节点，自动选择
                    if len(ec.nodes) == 1:
                        enode = next(iter(ec.nodes))
                        choices[eclass_id] = enode
                        # 继续检查子节点
                        for child_id in enode.children:
                            check_node(child_id)
                        return

                missing.append(eclass_id)
                return

            # 已选择的节点，检查子节点
            enode = choices[eclass_id]
            for child_id in enode.children:
                check_node(child_id)

        # 从所有已选择的节点开始检查
        for eclass_id in list(choices.keys()):
            enode = choices[eclass_id]
            for child_id in enode.children:
                check_node(child_id)

        return len(missing) == 0, list(set(missing))

    def _estimate_cost(self, enode: "ENode") -> float:
        """估计单个节点的代价"""
        # 简单启发式
        base = IOAwareCostModel.IO_COSTS.get(enode.op, 1.0)
        return base + len(enode.children) * 0.5

    def _estimate_cost_partial(self, choices: Dict[int, "ENode"]) -> float:
        """估计部分表达式的代价"""
        total = 0.0
        for enode in choices.values():
            total += self._estimate_cost(enode)
        return total

    def _compute_exact_cost(self, choices: Dict[int, "ENode"]) -> Cost:
        """计算精确代价"""
        # 拓扑排序
        order = self._topological_sort(choices)

        costs: Dict[int, Cost] = {}
        for eclass_id in order:
            enode = choices[eclass_id]
            child_costs = [
                costs.get(self.egraph.find(c), Cost.zero())
                for c in enode.children
            ]
            costs[eclass_id] = self.cost_model.enode_cost(enode, child_costs)

        # 返回根节点代价
        if order:
            return costs.get(order[-1], Cost.zero())
        return Cost.zero()

    def _topological_sort(self, choices: Dict[int, "ENode"]) -> List[int]:
        """拓扑排序"""
        visited = set()
        order = []

        def visit(eclass_id: int):
            if eclass_id in visited:
                return
            visited.add(eclass_id)

            enode = choices.get(eclass_id)
            if enode:
                for child_id in enode.children:
                    child_id = self.egraph.find(child_id)
                    if child_id in choices:
                        visit(child_id)

            order.append(eclass_id)

        for eclass_id in choices:
            visit(eclass_id)

        return order

    def _signature(self, choices: Dict[int, "ENode"]) -> str:
        """计算签名"""
        parts = []
        for eclass_id in sorted(choices.keys()):
            enode = choices[eclass_id]
            parts.append(f"{eclass_id}:{enode.op}")
        return "|".join(parts)
