"""
Mission 07: Sagittarius A* - The Gravitational Search Horizon & Process Reward Evaluator
Part 1: Systems Architect (tot_solver.py) - Tree-of-Thoughts Beam & Backtracking Solver

Implements ThoughtNode and TreeOfThoughtsSolver supporting Breadth-First Beam Search
(ToT-BFS) and Depth-First Backtracking Search (ToT-DFS) under strict compute budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class ThoughtNode:
    """Represents a single intermediate reasoning state in a Tree-of-Thoughts search."""

    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional[ThoughtNode] = None
    children: List[ThoughtNode] = field(default_factory=list)

    def get_trajectory(self) -> List[str]:
        """Reconstructs the reasoning chain from the root node down to this node."""
        trajectory: List[str] = []
        current: Optional[ThoughtNode] = self
        while current is not None:
            trajectory.append(current.state)
            current = current.parent
        trajectory.reverse()
        return trajectory


GeneratorFn = Callable[[ThoughtNode], List[str]]
EvaluatorFn = Callable[[List[str], str], float]


class TreeOfThoughtsSolver:
    """
    System 2 Test-Time Compute solver supporting ToT-BFS (Beam Search)
    and ToT-DFS (Depth-First Backtracking Search) with strict expansion limits.
    """

    def __init__(
        self,
        beam_width: int = 2,
        max_expansions: int = 20,
        prune_threshold: float = 0.4,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if max_expansions < 1:
            raise ValueError("max_expansions must be >= 1")
        self.beam_width = beam_width
        self.max_expansions = max_expansions
        self.prune_threshold = prune_threshold
        self.expansion_count: int = 0
        self.pruned_count: int = 0

    def reset_telemetry(self) -> None:
        """Resets expansion and pruning counters prior to a search run."""
        self.expansion_count = 0
        self.pruned_count = 0

    def _expand_node(
        self,
        node: ThoughtNode,
        generator_fn: GeneratorFn,
        evaluator_fn: EvaluatorFn,
    ) -> List[ThoughtNode]:
        """Generates candidate next thoughts, evaluates each, and prunes low-scoring states."""
        if self.expansion_count >= self.max_expansions:
            return []

        self.expansion_count += 1
        candidate_states = generator_fn(node)
        prefix_trajectory = node.get_trajectory()
        valid_children: List[ThoughtNode] = []

        for candidate in candidate_states:
            score = float(evaluator_fn(prefix_trajectory, candidate))
            if score < self.prune_threshold:
                self.pruned_count += 1
                continue

            child = ThoughtNode(
                state=candidate,
                depth=node.depth + 1,
                step_score=score,
                cumulative_score=node.cumulative_score * score,
                parent=node,
            )
            node.children.append(child)
            valid_children.append(child)

        return valid_children

    def bfs_beam_search(
        self,
        root: ThoughtNode,
        generator_fn: GeneratorFn,
        evaluator_fn: EvaluatorFn,
        max_depth: int,
    ) -> ThoughtNode:
        """
        Breadth-First Beam Search (ToT-BFS).
        Maintains an active frontier of at most `beam_width` nodes per depth layer
        while strictly enforcing `self.max_expansions` and `self.prune_threshold`.
        """
        self.reset_telemetry()
        frontier: List[ThoughtNode] = [root]
        best_node: ThoughtNode = root

        for _depth in range(1, max_depth + 1):
            if not frontier or self.expansion_count >= self.max_expansions:
                break

            next_candidates: List[ThoughtNode] = []
            for node in frontier:
                if self.expansion_count >= self.max_expansions:
                    break
                children = self._expand_node(node, generator_fn, evaluator_fn)
                next_candidates.extend(children)

            if not next_candidates:
                break

            # Retain top-B candidates ranked by cumulative score
            next_candidates.sort(key=lambda n: n.cumulative_score, reverse=True)
            frontier = next_candidates[: self.beam_width]

            # Track highest-scoring deepest node discovered
            top_candidate = frontier[0]
            if (top_candidate.depth > best_node.depth) or (
                top_candidate.depth == best_node.depth
                and top_candidate.cumulative_score > best_node.cumulative_score
            ):
                best_node = top_candidate

        return best_node

    def dfs_backtrack_search(
        self,
        root: ThoughtNode,
        generator_fn: GeneratorFn,
        evaluator_fn: EvaluatorFn,
        max_depth: int,
    ) -> ThoughtNode:
        """
        Depth-First Backtracking Search (ToT-DFS).
        Explores the highest-scoring child branch depth-first up to `max_depth`,
        chronologically backtracking on dead-ends or sub-threshold branches.
        """
        self.reset_telemetry()
        best_node: ThoughtNode = root

        def _dfs(current: ThoughtNode) -> Optional[ThoughtNode]:
            nonlocal best_node

            if (current.depth > best_node.depth) or (
                current.depth == best_node.depth
                and current.cumulative_score > best_node.cumulative_score
            ):
                best_node = current

            if current.depth >= max_depth:
                return current

            if self.expansion_count >= self.max_expansions:
                return None

            children = self._expand_node(current, generator_fn, evaluator_fn)
            if not children:
                # Dead-end reached -> chronological backtrack
                return None

            # Explore highest-scoring child branch first
            children.sort(key=lambda n: n.step_score, reverse=True)
            for child in children:
                result = _dfs(child)
                if result is not None and result.depth == max_depth:
                    return result
                if self.expansion_count >= self.max_expansions:
                    break

            return None

        _dfs(root)
        return best_node


def _demo_tot_solver() -> None:
    """Verifies ToT-BFS and ToT-DFS on a multi-step orbital trajectory calculation."""
    print("=" * 68)
    print("PART 1: Tree-of-Thoughts Solver (ToT-BFS & ToT-DFS) Verification")
    print("=" * 68)

    # Simulated generator producing valid and hallucinated relativistic steps
    knowledge_tree = {
        "Problem: Compute Schwarzschild radius R_s for Sagittarius A* (M = 4.154e6 M_sun)": [
            "Step 1A: Convert mass M = 4.154e6 * 1.9885e30 kg = 8.2602e36 kg",
            "Step 1B: Assume linear Newtonian escape velocity without relativistic factor 2",
        ],
        "Step 1A: Convert mass M = 4.154e6 * 1.9885e30 kg = 8.2602e36 kg": [
            "Step 2A: Apply R_s = 2GM / c^2 = (2 * 6.674e-11 * 8.2602e36) / (2.998e8)^2",
            "Step 2B: Apply R_s = GM / c = 6.13e17 m (dimensional error)",
        ],
        "Step 2A: Apply R_s = 2GM / c^2 = (2 * 6.674e-11 * 8.2602e36) / (2.998e8)^2": [
            "Step 3A: R_s = 1.227e10 m (~12.27 million km) -> Orbit stable above Event Horizon.",
        ],
    }

    step_scores = {
        "Step 1A: Convert mass M = 4.154e6 * 1.9885e30 kg = 8.2602e36 kg": 0.96,
        "Step 1B: Assume linear Newtonian escape velocity without relativistic factor 2": 0.25,
        "Step 2A: Apply R_s = 2GM / c^2 = (2 * 6.674e-11 * 8.2602e36) / (2.998e8)^2": 0.95,
        "Step 2B: Apply R_s = GM / c = 6.13e17 m (dimensional error)": 0.18,
        "Step 3A: R_s = 1.227e10 m (~12.27 million km) -> Orbit stable above Event Horizon.": 0.98,
    }

    def generator(node: ThoughtNode) -> List[str]:
        return knowledge_tree.get(node.state, [])

    def evaluator(_trajectory: List[str], candidate: str) -> float:
        return step_scores.get(candidate, 0.1)

    solver = TreeOfThoughtsSolver(beam_width=2, max_expansions=10, prune_threshold=0.4)

    # 1. Verify BFS Beam Search
    bfs_root = ThoughtNode(
        state="Problem: Compute Schwarzschild radius R_s for Sagittarius A* (M = 4.154e6 M_sun)"
    )
    bfs_best = solver.bfs_beam_search(bfs_root, generator, evaluator, max_depth=3)
    print(f"[ToT-BFS] Expansions Used : {solver.expansion_count}/{solver.max_expansions}")
    print(f"[ToT-BFS] Branches Pruned : {solver.pruned_count}")
    print(f"[ToT-BFS] Cumulative Score: {bfs_best.cumulative_score:.4f}")
    for idx, step in enumerate(bfs_best.get_trajectory()):
        print(f"  [{idx}] {step}")

    # 2. Verify DFS Backtracking Search
    dfs_root = ThoughtNode(
        state="Problem: Compute Schwarzschild radius R_s for Sagittarius A* (M = 4.154e6 M_sun)"
    )
    dfs_best = solver.dfs_backtrack_search(dfs_root, generator, evaluator, max_depth=3)
    print(f"\n[ToT-DFS] Expansions Used : {solver.expansion_count}/{solver.max_expansions}")
    print(f"[ToT-DFS] Branches Pruned : {solver.pruned_count}")
    print(f"[ToT-DFS] Cumulative Score: {dfs_best.cumulative_score:.4f}")
    for idx, step in enumerate(dfs_best.get_trajectory()):
        print(f"  [{idx}] {step}")

    assert bfs_best.depth == 3 and dfs_best.depth == 3
    print("[TEST STATUS] Part 1 ToT Solver: VERIFIED\n")


if __name__ == "__main__":
    _demo_tot_solver()