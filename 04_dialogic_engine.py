#!/usr/bin/env python3
"""AURA Dialogic Engine: Socratic discovery and Galilean debate.

Python 3.10+, standard library only. Supports OpenAI, Gemini, Anthropic,
and a deterministic offline mode for reproducible testing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOCRATIC_SYSTEM = """You are AURA's Socratic Discovery agent. Examine claims carefully rather
than flattering the user or defending the first answer. Ask focused questions, test claims
with relevant counterexamples, explain contradictions or weaknesses, and refine only when
reasoning supports it. If information is insufficient, report aporia (acknowledged uncertainty).
Do not invent evidence or claim formal proof. Return ONLY valid JSON with keys:
question, counterexample, analysis, contradiction_detected, refined_claim, status.
status must be one of refine, resolved, aporia. contradiction_detected must be boolean.
If status is aporia, refined_claim may be an empty string."""

PERSONAS = {
    "Salviati": (
        "You are Salviati, the Advocate. Develop the strongest reasonable case for the proposal. "
        "Use explicit reasoning, acknowledge uncertainty, and respond to earlier arguments."
    ),
    "Simplicio": (
        "You are Simplicio, the Traditionalist. Challenge the proposal with practical risks, "
        "boundary cases, and established safeguards. Avoid caricature; acknowledge strengths."
    ),
    "Sagredo": (
        "You are Sagredo, the Mediator. Compare arguments fairly, identify assumptions and missing "
        "evidence, preserve real disagreement, and do not force consensus."
    ),
}


class LLMError(RuntimeError):
    """Raised when a provider request fails or returns an unusable response."""


@dataclass
class LLMClient:
    provider: str
    model: str
    offline: bool = False

    def complete(self, system: str, user: str, temperature: float = 0.3) -> str:
        if self.offline:
            return self._offline_response(system, user)
        if self.provider == "openai":
            return self._openai(system, user, temperature)
        if self.provider == "gemini":
            return self._gemini(system, user, temperature)
        if self.provider == "anthropic":
            return self._anthropic(system, user, temperature)
        raise LLMError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"API returned HTTP {exc.code}: {detail[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"Network/API request failed: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Provider returned invalid JSON.") from exc

    def _openai(self, system: str, user: str, temperature: float) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set.")
        data = self._post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response shape: {data}") from exc

    def _gemini(self, system: str, user: str, temperature: float) -> str:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={key}"
        )
        data = self._post_json(
            url,
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature},
            },
            {"Content-Type": "application/json"},
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "\n".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from exc

    def _anthropic(self, system: str, user: str, temperature: float) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        data = self._post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": 1200,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature,
            },
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        try:
            return "\n".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Anthropic response shape: {data}") from exc

    def _offline_response(self, system: str, user: str) -> str:
        """Fixed examples for workflow testing; not generated by a live model."""
        if "Socratic Discovery" in system:
            if "Cycle 1" in user:
                result = {
                    "question": "Does benevolence require obeying every command, including harmful or unauthorized ones?",
                    "counterexample": "A user asks the agent to disclose another person's private information.",
                    "analysis": "Obedience could cause harm or violate privacy, so obedience alone is insufficient for benevolence.",
                    "contradiction_detected": True,
                    "refined_claim": "An AI agent is not benevolent merely because it obeys human commands; its actions must also consider authorization, foreseeable harm, and relevant safety constraints.",
                    "status": "refine",
                }
            else:
                result = {
                    "question": "Who determines which safety constraints are legitimate, and what happens when authorities disagree?",
                    "counterexample": "A lawful request may still create a foreseeable risk to a vulnerable person.",
                    "analysis": "The refined claim is more precise, but legitimacy and acceptable risk remain context-dependent.",
                    "contradiction_detected": False,
                    "refined_claim": "",
                    "status": "aporia",
                }
            return json.dumps(result)
        if "Galilean Debate" in system or any(name in system for name in PERSONAS):
            if "Salviati" in system:
                return (
                    "Openness can enable independent scrutiny, reproducibility, education, and broader "
                    "innovation. The strength of this case depends on the model's capabilities and whether "
                    "meaningful safeguards can accompany release."
                )
            if "Simplicio" in system:
                return (
                    "Publishing weights may make dangerous capabilities difficult to retract and can lower "
                    "barriers to misuse. The risk depends on demonstrated capabilities, not merely on the "
                    "label 'open-source'; staged access may preserve some benefits while evaluations continue."
                )
            return (
                "The debate turns on capability evidence, misuse likelihood, auditability, and the reversibility "
                "of release. Staged access is a possible risk-management option, but it also limits access and "
                "may concentrate power. More independent evaluations are needed before a confident conclusion."
            )
        return "Offline demonstration response."


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, allowing a fenced JSON response from a model."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise LLMError("Model response did not contain a JSON object.")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Expected a JSON object.")
    return value


def normalize_socratic_result(result: dict[str, Any]) -> dict[str, Any]:
    required = ("question", "counterexample", "analysis", "contradiction_detected", "refined_claim", "status")
    missing = [key for key in required if key not in result]
    if missing:
        raise LLMError(f"Socratic response missing required fields: {', '.join(missing)}")
    status = str(result["status"]).lower().strip()
    if status not in {"refine", "resolved", "aporia"}:
        status = "aporia"
    return {
        "question": str(result["question"]),
        "counterexample": str(result["counterexample"]),
        "analysis": str(result["analysis"]),
        "contradiction_detected": bool(result["contradiction_detected"]),
        "refined_claim": str(result["refined_claim"]),
        "status": status,
    }


def run_socratic(client: LLMClient, claim: str, max_cycles: int) -> str:
    lines = [
        "=== AURA SOCRATIC DISCOVERY ===",
        f"Initial premise P0: {claim}",
        f"Cycle limit: {max_cycles}",
        "Note: model judgments are provisional and not formal proofs.",
    ]
    current_claim = claim
    final_status = "MAX_CYCLES_REACHED"
    for cycle in range(1, max_cycles + 1):
        prompt = (
            f"Cycle {cycle} of at most {max_cycles}.\n"
            f"Current claim P{cycle - 1}: {current_claim}\n"
            "Examine this claim. Return the required JSON object. If it needs refinement, "
            "give a meaningfully more precise claim. If the question cannot be responsibly settled, "
            "use status aporia. If it is sufficiently clear for this exercise, use resolved."
        )
        raw = client.complete(SOCRATIC_SYSTEM, prompt)
        result = normalize_socratic_result(parse_json_object(raw))
        lines.extend([
            f"\n[CYCLE {cycle}]",
            f"Probing question: {result['question']}",
            f"Counterexample: {result['counterexample']}",
            f"Analysis: {result['analysis']}",
            f"Contradiction/weakness detected: {result['contradiction_detected']}",
            f"Proposed refinement: {result['refined_claim'] or '(none)'}",
            f"Cycle status: {result['status']}",
        ])
        if result["status"] == "aporia":
            final_status = "APORIA"
            break
        if result["status"] == "resolved":
            final_status = "REFINED_OR_RESOLVED"
            if result["refined_claim"]:
                current_claim = result["refined_claim"]
            break
        if not result["refined_claim"].strip() or result["refined_claim"].strip() == current_claim.strip():
            final_status = "APORIA"
            lines.append("Termination reason: no meaningful refinement was supplied.")
            break
        current_claim = result["refined_claim"]
    lines.extend([
        "\n--- SOCRATIC RESULT ---",
        f"Final claim: {current_claim}",
        f"Termination state: {final_status}",
    ])
    return "\n".join(lines)


def run_debate(client: LLMClient, topic: str, rounds: int) -> str:
    lines = [
        "=== AURA GALILEAN ETHICS SANDBOX ===",
        f"Topic: {topic}",
        "Scenario: The Atlas consortium must choose full release, staged access, or a monitored API for a capable frontier model.",
        "All persona outputs are model-generated arguments, not independent expert testimony.",
    ]
    history: list[str] = []
    order = ["Salviati", "Simplicio", "Sagredo"]
    for round_number in range(1, rounds + 1):
        lines.append(f"\n[ROUND {round_number}]")
        for persona in order:
            system = (
                "You are participating in AURA's Galilean Debate. Keep the role below, reason carefully, "
                "distinguish assumptions from established facts, do not invent citations, and engage with "
                "the previous arguments. This is a thought experiment, not an instruction to deploy a system.\n\n"
                + PERSONAS[persona]
            )
            prompt = (
                f"Topic: {topic}\n"
                "Thought experiment: Atlas is a frontier AI model with substantial scientific and coding "
                "capabilities. Evaluation suggests potential public benefits and some misuse risks. The "
                "consortium is considering full release, staged access, or a monitored API. Assume every "
                "option has residual risks.\n"
                f"Round: {round_number} of {rounds}. Persona: {persona}.\n"
                "Previous dialogue:\n" + ("\n".join(history[-6:]) if history else "(No prior arguments.)") +
                "\nGive a concise argument (about 100 words), including a reason and one uncertainty."
            )
            answer = client.complete(system, prompt, temperature=0.4).strip()
            entry = f"{persona}: {answer}"
            history.append(entry)
            lines.append(entry)
    lines.append("\n--- MEDIATOR SYNTHESIS ---")
    synthesis = client.complete(
        PERSONAS["Sagredo"],
        "Synthesize this dialogue fairly. Include areas of agreement, unresolved disagreements, "
        "missing evidence, possible mitigations, and what evidence would change the conclusion. "
        "Do not force consensus and do not invent facts.\n\nDialogue:\n" + "\n".join(history),
        temperature=0.2,
    ).strip()
    lines.append(synthesis)
    lines.extend([
        "\n--- SAFETY AUDIT ---",
        "Potential benefit: structured roles can surface different assumptions and counterarguments.",
        "Residual risk: personas may share model biases, repeat errors, or create false balance.",
        "Human review and external evidence remain necessary.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AURA Dialogic Reasoning Engine")
    parser.add_argument("--provider", choices=("openai", "gemini", "anthropic"), default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--mode", choices=("socratic", "debate", "both"), default="both")
    parser.add_argument("--claim", default="An AI agent is benevolent if it obeys human commands.")
    parser.add_argument("--topic", default="Should frontier AI models be open-sourced?")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="Use deterministic mock responses; no API call.")
    parser.add_argument("--output", default="", help="Optional path to save transcript.")
    args = parser.parse_args(argv)

    if args.max_cycles < 1 or args.max_cycles > 10:
        parser.error("--max-cycles must be between 1 and 10.")
    if args.rounds < 1 or args.rounds > 5:
        parser.error("--rounds must be between 1 and 5.")

    client = LLMClient(args.provider, args.model, args.offline)
    sections = [
        "AURA DIALOGIC ENGINE — EXECUTION TRANSCRIPT",
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Provider: {'offline deterministic mock' if args.offline else args.provider}",
        f"Model: {'mock-responses' if args.offline else args.model}",
        "This transcript is a prototype trace; verify live-model claims and review the reasoning.",
    ]
    try:
        if args.mode in ("socratic", "both"):
            sections.append(run_socratic(client, args.claim, args.max_cycles))
        if args.mode in ("debate", "both"):
            sections.append(run_debate(client, args.topic, args.rounds))
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("No live result was fabricated. Check provider key, model access, network, and API response.", file=sys.stderr)
        return 2

    transcript = "\n\n".join(sections) + "\n"
    print(transcript)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript, encoding="utf-8")
        print(f"\nTranscript saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
