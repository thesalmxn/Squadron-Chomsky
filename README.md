# Squadron Chomsky - USSE Noosphere Mission 5

**Course:** Conversational AI | Winter Semester 2026/27  
**Waypoint:** SEC-05-M01 (Crab Pulsar Remnant / Messier 1)  
**Restoration Goal:** AURA Subsystem 05: Dual-Path Associative Memory Core (+7.1% -> 35.5% Integrity)

---

## Repository Layout
```
Squadron-Chomsky/
├── README.md
├── requirements.txt
└── weeks/
    └── week-05/
        ├── 01_HNSW_Vector_Index.py
        ├── 02_BM25_and_RRF_Combiner.py
        ├── 03_NotebookLM_RAG_Audit.py
        ├── 04_RAG_Provenance_and_Safety.py
        ├── ps05_solution.ipynb
        ├── notebooklm_audit_report.md
        └── Mission_5_Technical_Report.md
```

## Setup & Quick Start

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/thesalmxn/Squadron-Chomsky.git
   cd Squadron-Chomsky
   ```

2. Activate virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the complete Mission 5 validation suite:
   ```bash
   python weeks/week-05/01_HNSW_Vector_Index.py
   python weeks/week-05/02_BM25_and_RRF_Combiner.py
   python weeks/week-05/03_NotebookLM_RAG_Audit.py
   python weeks/week-05/04_RAG_Provenance_and_Safety.py
   ```

4. Alternatively, launch Jupyter Lab and run `weeks/week-05/ps05_solution.ipynb`.
