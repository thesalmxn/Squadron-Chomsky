"""
Mission 07: Sagittarius A* - The Gravitational Search Horizon & Process Reward Evaluator
Part 2: Communications & Protocol Officer (mcts_protocol.py) - 4-Phase MCTS & PUCT Engine

Implements MCTSNode and MCTSEngine with exact PUCT selection, expansion,
rollout simulation, and upward backpropagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class MCTSNode:
    """Tree node storing visit counts, accumulated value, and policy prior P(a|s)."""

    state: str
    prior_prob: float = 1.0
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Mean action-value Q(s, a) = W(s, a) / N(s, a), or 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Returns True if child nodes have been instantiated."""
        return len(self.children) > 0

    def get_trajectory(self) -> List[str]:
        """Reconstructs the reasoning path from the root down to this node."""
        path: List[str] = []
        curr: Optional[MCTSNode] = self
        while curr is not None:
            path.append(curr.state)
            curr = curr.parent
        path.reverse()
        return path


PolicyGeneratorFn = Callable[[MCTSNode], List[Tuple[str, float]]]
RolloutEvaluatorFn = Callable[[MCTSNode], float]


class MCTSEngine:
    """
    4-Phase Monte Carlo Tree Search engine guided by LLM policy priors P_theta(a|s)
    and the AlphaZero/MuZero PUCT selection criterion.
    """

    def __init__(self, c_puct: float = 1.414) -> None:
        if c_puct < 0.0:
            raise ValueError("c_puct must be non-negative")
        self.c_puct = c_puct

    def puct_score(self, child: MCTSNode) -> float:
        """
        Computes exact PUCT(s, a):
            PUCT(s, a) = Q(s, a) + c_puct * P_theta(a|s) * sqrt(N(s)) / (1 + N(s, a))
        where Q(s, a) = W(s, a) / N(s, a) if N(s, a) > 0 else 0.0,
        and N(s) = max(1, parent.visit_count).
        """
        q_sa = child.q_value
        parent_visits = max(1, child.parent.visit_count) if child.parent is not None else 1
        exploration_bonus = (
            self.c_puct
            * child.prior_prob
            * (math.sqrt(parent_visits) / (1.0 + child.visit_count))
        )
        return q_sa + exploration_bonus

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Selects the child maximizing PUCT(s, a)."""
        if not node.children:
            raise ValueError("Cannot select best child from a node with no children.")
        return max(node.children, key=self.puct_score)

    def select_leaf(self, root: MCTSNode) -> MCTSNode:
        """
        Phase 1 (Selection):
        Traverses from root by greedily selecting select_best_child(node)
        until reaching an unexpanded or terminal leaf node.
        """
        current = root
        while current.is_expanded and not current.is_terminal:
            current = self.select_best_child(current)
        return current

    def expand(
        self,
        node: MCTSNode,
        candidate_priors: List[Tuple[str, float]],
        terminal_check: Optional[Callable[[str], bool]] = None,
    ) -> List[MCTSNode]:
        """
        Phase 2 (Expansion):
        Instantiates child MCTSNode objects for each (thought_state, prior_prob) pair.
        """
        if node.is_terminal or not candidate_priors:
            node.is_terminal = True
            return []

        created_children: List[MCTSNode] = []
        for thought_state, prior_prob in candidate_priors:
            is_term = terminal_check(thought_state) if terminal_check is not None else False
            child = MCTSNode(
                state=thought_state,
                prior_prob=float(prior_prob),
                parent=node,
                is_terminal=is_term,
            )
            node.children.append(child)
            created_children.append(child)

        return created_children

    def simulate_rollout(
        self,
        node: MCTSNode,
        rollout_evaluator: RolloutEvaluatorFn,
    ) -> float:
        """
        Phase 3 (Simulation / Rollout):
        Evaluates the newly expanded node via a simulated trajectory rollout or PRM
        value estimate clamped strictly to [0.0, 1.0].
        """
        raw_reward = float(rollout_evaluator(node))
        return max(0.0, min(1.0, raw_reward))

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Phase 4 (Backpropagation):
        Traverses upward from `node` to root, incrementing visit_count += 1
        and accumulating value_sum += reward at every ancestor.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent

    def run_search(
        self,
        root: MCTSNode,
        policy_generator: PolicyGeneratorFn,
        rollout_evaluator: RolloutEvaluatorFn,
        num_simulations: int = 25,
        terminal_check: Optional[Callable[[str], bool]] = None,
    ) -> MCTSNode:
        """Executes `num_simulations` full 4-phase MCTS iterations and returns best child."""
        for _ in range(num_simulations):
            # Phase 1: Selection
            leaf = self.select_leaf(root)

            # Phase 2: Expansion (if non-terminal)
            target_node = leaf
            if not leaf.is_terminal:
                priors = policy_generator(leaf)
                children = self.expand(leaf, priors, terminal_check=terminal_check)
                if children:
                    # Select highest-prior newly expanded child for rollout
                    target_node = max(children, key=lambda c: c.prior_prob)

            # Phase 3: Simulation / Evaluation
            reward = self.simulate_rollout(target_node, rollout_evaluator)

            # Phase 4: Backpropagation
            self.backpropagate(target_node, reward)

        if not root.children:
            return root
        return max(root.children, key=lambda c: c.visit_count)


def _demo_mcts_engine() -> None:
    """Verifies PUCT calculation and 4-phase MCTS search convergence."""
    print("=" * 68)
    print("PART 2: 4-Phase MCTS & PUCT Engine Verification")
    print("=" * 68)

    transitions = {
        "Query: Stabilize AURA trajectory at 3.0 R_s": [
            ("Action A: Engage tangential ion-thruster burn (v_orb = sqrt(GM/r))", 0.65),
            ("Action B: Radial dive toward photon sphere (r = 1.5 R_s)", 0.35),
        ],
        "Action A: Engage tangential ion-thruster burn (v_orb = sqrt(GM/r))": [
            ("Step A1: Apply relativistic frame-dragging correction +0.04c", 0.80),
            ("Step A2: Ignore Kerr spin parameter a*", 0.20),
        ],
        "Action B: Radial dive toward photon sphere (r = 1.5 R_s)": [
            ("Step B1: Unstable circular photon orbit -> tidal shear loss", 0.90),
        ],
    }

    reward_table = {
        "Action A: Engage tangential ion-thruster burn (v_orb = sqrt(GM/r))": 0.88,
        "Action B: Radial dive toward photon sphere (r = 1.5 R_s)": 0.15,
        "Step A1: Apply relativistic frame-dragging correction +0.04c": 0.96,
        "Step A2: Ignore Kerr spin parameter a*": 0.42,
        "Step B1: Unstable circular photon orbit -> tidal shear loss": 0.05,
    }

    engine = MCTSEngine(c_puct=1.414)
    root = MCTSNode(state="Query: Stabilize AURA trajectory at 3.0 R_s")

    best_action = engine.run_search(
        root=root,
        policy_generator=lambda n: transitions.get(n.state, []),
        rollout_evaluator=lambda n: reward_table.get(n.state, 0.5),
        num_simulations=20,
        terminal_check=lambda s: s.startswith("Step "),
    )

    print(f"Root Visit Count : {root.visit_count}")
    for child in root.children:
        print(
            f"  Child: '{child.state[:48]}...' | Visits={child.visit_count:2d} "
            f"| Q={child.q_value:.3f} | PUCT={engine.puct_score(child):.3f}"
        )
    print(f"Selected Optimal Action: {best_action.state}")
    assert best_action.state.startswith("Action A")
    print("[TEST STATUS] Part 2 MCTS & PUCT Engine: VERIFIED\n")


if __name__ == "__main__":
    _demo_mcts_engine()