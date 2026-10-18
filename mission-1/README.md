# AURA Dialogic Reasoning Engine — Mission 1

**Course:** Conversational AI, Winter Semester 2026/27  
**Role focus:** Core Architect  
**Language:** Python 3.10+  
**Dependencies:** Python standard library only

This project implements both dialogue protocols required by Mission 1:

1. **Socratic discovery:** examine an initial claim, ask a probing question, generate a counterexample, identify a contradiction or weakness, and refine the claim. If the evidence is insufficient, the engine can stop in **aporia** (acknowledged uncertainty).
2. **Galilean multi-agent debate:** run three distinct personas — Salviati (Advocate), Simplicio (Traditionalist), and Sagredo (Mediator) — on a modern AI-risk thought experiment.

The project includes reports, a protocol map, an ethics sandbox, an API-enabled Python implementation, and a reproducible offline execution transcript.

## 1. Project structure

```text
mission-1/
├── .env.example
├── 04_dialogic_engine.py
├── CORE_ARCHITECT_WALKTHROUGH.md
├── README.md
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
├── reports/
│   ├── 01_Alignment_Brief.md
│   ├── 02_Socratic_Protocol_Map.md
│   └── 03_Galilean_Sandbox.md
├── outputs/
│   ├── 05_Execution_Transcript.txt
│   └── TEST_RESULTS.txt
└── tests/
    └── test_dialogic_engine.py
```

## 2. Requirements

- Python 3.10 or newer.
- For live mode, an API key for one supported provider: OpenAI, Google Gemini, or Anthropic.
- Internet access for live API requests.

No third-party Python packages are required.

## 3. Quick start: reproducible offline demo

From this folder, run:

```bash
python 04_dialogic_engine.py --offline --mode both
```

This uses deterministic mock responses so you can inspect the dialogue flow without an API key or network access. It is a software demonstration, **not** a live LLM evaluation.

## 4. Live LLM mode

Copy `.env.example` to `.env`, then set the environment variables in your terminal. The program does not automatically load `.env`; this avoids an extra dependency. Set the variables in your shell or configure them in VS Code.

### OpenAI (Chat Completions API)

PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
python 04_dialogic_engine.py --provider openai --model gpt-4.1-mini --mode both
```

macOS/Linux:
```bash
export OPENAI_API_KEY="your-key"
python 04_dialogic_engine.py --provider openai --model gpt-4.1-mini --mode both
```

### Google Gemini

PowerShell:
```powershell
$env:GEMINI_API_KEY="your-key"
python 04_dialogic_engine.py --provider gemini --model gemini-2.5-flash --mode both
```

### Anthropic Claude

PowerShell:
```powershell
$env:ANTHROPIC_API_KEY="your-key"
python 04_dialogic_engine.py --provider anthropic --model claude-3-5-haiku-latest --mode both
```

Model availability, API pricing, and model names can change. Use a model name enabled for your account and check the provider's current documentation. API usage may incur charges.

## 5. Run individual protocols

Socratic discovery:
```bash
python 04_dialogic_engine.py --offline --mode socratic --claim "An AI agent is benevolent if it obeys human commands."
```

Galilean debate:
```bash
python 04_dialogic_engine.py --offline --mode debate --topic "Should frontier AI models be open-sourced?"
```

Set the number of Socratic cycles:
```bash
python 04_dialogic_engine.py --offline --mode socratic --max-cycles 3
```

Save a fresh transcript:
```bash
python 04_dialogic_engine.py --offline --mode both --output outputs/my_transcript.txt
```

## 6. Tests

```bash
python -m unittest discover -s tests -v
```

## 7. Core architecture

- `LLMClient`: provider adapter; handles OpenAI, Gemini, and Anthropic HTTP APIs.
- `SocraticEngine`: state-machine loop for probing, counterexamples, contradiction analysis, and refinement/aporia.
- `GalileanDebate`: three-role debate followed by a mediator synthesis.
- `main()`: CLI argument parsing, protocol selection, transcript formatting, and output persistence.

## 8. Important limitations

- Persona names are role prompts, not independent human experts.
- Separate LLM calls can still share biases, factual errors, or blind spots.
- The contradiction/refinement status is model-generated and should be reviewed, not treated as a formal proof.
- Offline responses are fixed examples for testing the workflow only.
- The current implementation is a command-line prototype, not a production safety system.
