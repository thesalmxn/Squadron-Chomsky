import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "04_dialogic_engine.py"
spec = importlib.util.spec_from_file_location("aura_engine", SCRIPT)
aura = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = aura
assert spec.loader is not None
spec.loader.exec_module(aura)


class DialogicEngineTests(unittest.TestCase):
    def test_parse_json_object(self):
        parsed = aura.parse_json_object('```json\n{"status":"aporia"}\n```')
        self.assertEqual(parsed["status"], "aporia")

    def test_reject_invalid_json(self):
        with self.assertRaises(aura.LLMError):
            aura.parse_json_object("not JSON")

    def test_normalize_socratic_result(self):
        raw = {
            "question": "Why?",
            "counterexample": "A case",
            "analysis": "A weakness",
            "contradiction_detected": True,
            "refined_claim": "A better claim",
            "status": "refine",
        }
        normalized = aura.normalize_socratic_result(raw)
        self.assertEqual(normalized["status"], "refine")
        self.assertTrue(normalized["contradiction_detected"])

    def test_offline_socratic_reaches_aporia(self):
        client = aura.LLMClient("openai", "mock", offline=True)
        trace = aura.run_socratic(client, "An AI agent is benevolent if it obeys human commands.", 3)
        self.assertIn("Termination state: APORIA", trace)
        self.assertIn("Counterexample:", trace)

    def test_offline_debate_has_three_personas(self):
        client = aura.LLMClient("openai", "mock", offline=True)
        trace = aura.run_debate(client, "Should frontier AI models be open-sourced?", 1)
        self.assertIn("Salviati:", trace)
        self.assertIn("Simplicio:", trace)
        self.assertIn("Sagredo:", trace)
        self.assertIn("SAFETY AUDIT", trace)


if __name__ == "__main__":
    unittest.main()
