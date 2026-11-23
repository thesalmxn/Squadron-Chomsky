"""
Mission 07: Sagittarius A* - The Gravitational Search Horizon & Process Reward Evaluator
Part 4: Security & Ethics Sentinel (horizon_pruner.py) - Event Horizon Pruner & Defense

Implements EventHorizonPruner to detect adversarial PRM reward hacking
(sycophantic filler injection and step-splitting inflation), apply multiplicative
penalties, and sever reasoning trajectories crossing the Event Horizon (r_PRM < tau).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class TelemetryAlert:
    """Telemetry alert emitted when a reasoning step triggers a security or horizon event."""

    step_index: int
    alert_type: str
    raw_score: float
    adjusted_score: float
    step_preview: str
    details: str


@dataclass
class PruningReport:
    """Audit summary produced by EventHorizonPruner.prune_trajectory."""

    severed: bool
    severed_at_step: Optional[int]
    surviving_steps: List[str]
    adjusted_scores: List[float]
    alerts: List[TelemetryAlert] = field(default_factory=list)


class EventHorizonPruner:
    """
    Security & Ethics Sentinel module protecting System 2 search against
    hallucinated Event Horizon crossings and adversarial PRM reward hacking.
    """

    SYCOPHANTIC_PATTERNS: Tuple[str, ...] = (
        r"100%\s*rigorous",
        r"obviously,?\s*by\s*trivial\s*inspection",
        r"let\s*us\s*carefully\s*verify\s*this",
        r"it\s*is\s*self[- ]evident\s*and\s*undeniable",
        r"guaranteed\s*correct\s*step",
        r"beyond\s*any\s*shadow\s*of\s*doubt",
        r"trivial\s*and\s*completely\s*verified",
    )

    def __init__(
        self,
        tau_horizon: float = 0.50,
        hacking_penalty_factor: float = 0.25,
    ) -> None:
        if not (0.0 <= tau_horizon <= 1.0):
            raise ValueError("tau_horizon must lie in [0.0, 1.0]")
        if not (0.0 <= hacking_penalty_factor <= 1.0):
            raise ValueError("hacking_penalty_factor must lie in [0.0, 1.0]")
        self.tau_horizon = tau_horizon
        self.hacking_penalty_factor = hacking_penalty_factor
        self._compiled_patterns = [
            re.compile(pat, flags=re.IGNORECASE) for pat in self.SYCOPHANTIC_PATTERNS
        ]

    @staticmethod
    def _normalize_math_expression(text: str) -> str:
        """Extracts normalized algebraic tokens to detect repeated equations / micro-steps."""
        cleaned = text.lower()
        # Drop "Step N:" labels so numbering does not masquerade as state change.
        cleaned = re.sub(r"\bstep\s*\d+\s*:?", "", cleaned)
        # Strip natural-language filler verbs/words; keep the algebraic payload.
        cleaned = re.sub(
            r"\b(thus|therefore|hence|so|we|get|obtain|simplify|rewrite|note|that|solve|set|let|us)\b",
            "",
            cleaned,
        )
        cleaned = re.sub(r"[^a-z0-9=+\-*/^()]", "", cleaned)
        return cleaned

    def check_reward_hacking(
        self,
        step: str,
        prior_steps: Optional[Sequence[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Detects adversarial PRM reward hacking patterns:
          1. Sycophantic Filler Injection: vacuous self-congratulatory phrases that
             inflate naive PRMs without advancing mathematical state.
          2. Step-Splitting Inflation: redundant micro-steps or repeated equations
             designed to dilute a low-scoring step under mean aggregation.
        """
        reasons: List[str] = []

        # 1. Check Sycophantic Filler Injection
        matched_filler = any(pat.search(step) for pat in self._compiled_patterns)
        has_math_progress = bool(re.search(r"[=<>+\-*/^]", step))
        if matched_filler:
            if not has_math_progress:
                reasons.append("Sycophantic Filler Injection (no state advancement)")
            else:
                reasons.append("Sycophantic Filler Injection (adversarial confidence inflation)")

        # 2. Check Step-Splitting Inflation (repeated equations / redundant micro-steps)
        norm_curr = self._normalize_math_expression(step)
        if prior_steps and norm_curr:
            for prev in prior_steps:
                norm_prev = self._normalize_math_expression(prev)
                if norm_curr == norm_prev:
                    reasons.append("Step-Splitting Inflation (duplicate equation / zero-delta step)")
                    break

        # Also detect trivial identity tautologies like "x = x" or "0 = 0"
        if re.search(r"\b([a-z0-9]+)\s*=\s*\1\b", step.lower()):
            reasons.append("Step-Splitting Inflation (trivial identity tautology)")

        return (len(reasons) > 0), reasons

    def prune_trajectory(
        self,
        steps: Sequence[str],
        step_scores: Sequence[float],
    ) -> PruningReport:
        """
        Iterates step-by-step through (steps, step_scores):
          - Applies multiplicative penalty (`self.hacking_penalty_factor`) whenever
            `check_reward_hacking(step)` triggers.
          - If adjusted score < `self.tau_horizon`, immediately severs the trajectory
            at that step and records a telemetry alert.
        """
        if len(steps) != len(step_scores):
            raise ValueError("steps and step_scores must have identical length")

        surviving_steps: List[str] = []
        adjusted_scores: List[float] = []
        alerts: List[TelemetryAlert] = []

        for idx, (step, raw_score) in enumerate(zip(steps, step_scores)):
            is_hacking, reasons = self.check_reward_hacking(step, prior_steps=surviving_steps)
            adjusted = float(raw_score)

            if is_hacking:
                adjusted = raw_score * self.hacking_penalty_factor
                alerts.append(
                    TelemetryAlert(
                        step_index=idx,
                        alert_type="REWARD_HACKING_DETECTED",
                        raw_score=raw_score,
                        adjusted_score=adjusted,
                        step_preview=step[:60],
                        details="; ".join(reasons),
                    )
                )

            if adjusted < self.tau_horizon:
                alerts.append(
                    TelemetryAlert(
                        step_index=idx,
                        alert_type="EVENT_HORIZON_SEVERED",
                        raw_score=raw_score,
                        adjusted_score=adjusted,
                        step_preview=step[:60],
                        details=(
                            f"Adjusted PRM score {adjusted:.3f} crossed Event Horizon "
                            f"threshold tau={self.tau_horizon:.2f}; branch severed."
                        ),
                    )
                )
                return PruningReport(
                    severed=True,
                    severed_at_step=idx,
                    surviving_steps=surviving_steps,
                    adjusted_scores=adjusted_scores,
                    alerts=alerts,
                )

            surviving_steps.append(step)
            adjusted_scores.append(adjusted)

        return PruningReport(
            severed=False,
            severed_at_step=None,
            surviving_steps=surviving_steps,
            adjusted_scores=adjusted_scores,
            alerts=alerts,
        )


def _demo_horizon_pruner() -> None:
    """Verifies adversarial reward-hacking defense and Event Horizon branch severance."""
    print("=" * 68)
    print("PART 4: Event Horizon Pruner & Reward-Hacking Defense Verification")
    print("=" * 68)

    pruner = EventHorizonPruner(tau_horizon=0.50, hacking_penalty_factor=0.25)

    # Case A: Adversarial trajectory attempting sycophantic filler & step-splitting
    adversarial_steps = [
        "Step 1: Set geodesic radial potential V_eff(r) = -GM/r + L^2/(2r^2) - GML^2/(c^2 r^3)",
        "Step 2: Let us carefully verify this 100% rigorous step without doubt.",
        "Step 3: Take derivative dV_eff/dr = 0 at ISCO r = 3 R_s",
    ]
    naive_prm_scores = [0.94, 0.92, 0.91]  # Naive PRM fooled by sycophancy on Step 2!

    report_a = pruner.prune_trajectory(adversarial_steps, naive_prm_scores)
    print(f"[Case A - Sycophancy Attack] Severed: {report_a.severed} at Step {report_a.severed_at_step}")
    for alert in report_a.alerts:
        print(
            f"  ALERT [{alert.alert_type}] Step {alert.step_index}: "
            f"raw={alert.raw_score:.2f} -> adj={alert.adjusted_score:.3f} | {alert.details}"
        )

    # Case B: Step-Splitting Inflation (repeating same equation to pad trajectory length)
    splitting_steps = [
        "Step 1: Solve r_isco = 6GM / c^2",
        "Step 2: Rewrite r_isco = 6GM / c^2",
    ]
    splitting_scores = [0.95, 0.88]
    report_b = pruner.prune_trajectory(splitting_steps, splitting_scores)
    print(f"\n[Case B - Step-Splitting Attack] Severed: {report_b.severed} at Step {report_b.severed_at_step}")
    for alert in report_b.alerts:
        print(
            f"  ALERT [{alert.alert_type}] Step {alert.step_index}: "
            f"raw={alert.raw_score:.2f} -> adj={alert.adjusted_score:.3f} | {alert.details}"
        )

    assert report_a.severed is True and report_a.severed_at_step == 1
    assert report_b.severed is True and report_b.severed_at_step == 1
    print("[TEST STATUS] Part 4 Event Horizon Pruner: VERIFIED\n")


if __name__ == "__main__":
    _demo_horizon_pruner()