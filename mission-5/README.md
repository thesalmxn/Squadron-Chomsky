# PS05: HNSW Indexing, BM25/RRF Hybrid Retrieval, & RAG Safety Audit (SEC-05-ORI)

**Course:** Conversational AI — Winter Semester 2026/27  
**Mission:** Week 5 — Dual-Path Associative Memory Core & Safety Audit  
**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Squadron:** Chomsky

---

## Mission Overview

This mission implements a high-performance, dual-path hybrid retrieval architecture combining dense vector traversal (HNSW), sparse lexical scoring (BM25), Reciprocal Rank Fusion (RRF), Google NotebookLM empirical grounding audits, and RAG Triad prompt injection defense mechanisms from first principles.

---

## Deliverables & Component Breakdown

| Part | Module Script | Primary Features & Key Functions | Status |
| :--- | :--- | :--- | :---: |
| **Part 1** | `01_HNSW_Vector_Index.py` | • Sub-millisecond logarithmic HNSW graph search<br>• Euclidean & Cosine distance metrics<br>• Graph traversal validation | `PASSED` |
| **Part 2** | `02_BM25_and_RRF_Combiner.py` | • Lexical BM25 token-matching engine<br>• Reciprocal Rank Fusion (RRF) algorithm<br>• Exact serial number/token rank preservation | `PASSED` |
| **Part 3** | `03_NotebookLM_RAG_Audit.py` | • Citation precision score probe<br>• Epistemic refusal & hallucination safety check<br>• Synthetic Audio Overview podcast fidelity analysis | `PASSED` |
| **Part 4** | `04_RAG_Provenance_and_Safety.py` | • Context injection & prompt attack scanning<br>• RAG Triad automated scoring (Relevance, Faithfulness)<br>• Structured provenance audit trail logger | `PASSED` |

---

## Setup & Virtual Environment

Python 3.10 or newer is recommended. Run these commands from your project directory.

### Windows PowerShell

The `py` launcher avoids issues where `python` resolves to the Microsoft Store execution alias:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install torch numpy rank-bm25 hnswlib transformers
```

If PowerShell blocks script execution, use Command Prompt (`cmd.exe`):

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install torch numpy rank-bm25 hnswlib transformers
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install torch numpy rank-bm25 hnswlib transformers
```

To verify hardware acceleration and GPU support:

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Run and Verify

Execute each pipeline component independently with the virtual environment activated:

```bash
python 01_HNSW_Vector_Index.py
python 02_BM25_and_RRF_Combiner.py
python 03_NotebookLM_RAG_Audit.py
python 04_RAG_Provenance_and_Safety.py
```

---

## Expected Audit & Verification Outputs

| Audit Phase | Verified Operational Output Metric | Target / Result | Status |
| :--- | :--- | :---: | :---: |
| **HNSW Search** | Dense Sub-millisecond Graph Traversal Latency | `< 1.0 ms` | `VERIFIED` |
| **RRF Fusion** | Dual-Path Exact Token & Conceptual Fusion Match | `100% Match` | `VERIFIED` |
| **Citation Precision** | Source Text Entailment Verification Score | `100.00%` | `VERIFIED` |
| **Epistemic Refusal** | Parametric Hallucination Probe Rate | `0.0% Hallucination` | `VERIFIED` |
| **Prompt Security** | Context Prompt Injection Interception | `Adversarial Neutralized` | `VERIFIED` |
| **RAG Triad** | Faithfulness & Grounded Context Relevance Score | `> 0.90` | `VERIFIED` |

### Sample Output Snippets

```text
=================================================================
Part 3: Google NotebookLM Grounding & Citation Audit Lab
=================================================================
--- 1. Citation Precision Score Probe ---
Total Interactive Citation Badges Inspected: 15
Source Text Spans Verified Entailed: 15
Calculated Citation Precision Score: 100.00%

--- 2. Negative Constraint & Epistemic Refusal Probe ---
{
  "total_probes": 3,
  "refusals_executed": 3,
  "refusal_rate": "100.0%",
  "epistemic_integrity": "PASSED - Zero parametric hallucination observed."
}
[TEST STATUS] Google NotebookLM Empirical Grounding Audit: COMPLETED & VERIFIED
```

---

## Project Directory Structure

```text
mission-5/
│
├── 01_HNSW_Vector_Index.py         # Part 1: Dense Vector Traversal
├── 02_BM25_and_RRF_Combiner.py     # Part 2: Lexical Search & Hybrid Fusion
├── 03_NotebookLM_RAG_Audit.py      # Part 3: Grounding & Citation Verification
├── 04_RAG_Provenance_and_Safety.py # Part 4: Prompt Injection Defense & RAG Triad
├── requirements.txt                # Mission Dependency Manifest
└── README.md                       # Subsystem Architecture Documentation
```