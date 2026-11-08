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

If PowerShell blocks script execution, use Command Prompt (`cmd.exe`):

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install torch numpy rank-bm25 hnswlib transformers
