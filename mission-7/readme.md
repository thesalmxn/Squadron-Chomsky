# Mission 07: Sagittarius A* — The Gravitational Search Horizon & Process Reward Evaluator

**Course:** Conversational AI (Winter Semester 2026/27)  
**Starship Operational Unit:** USSE Noosphere Bridge Crew (Squadron Chomsky)  
**Galactic Waypoint:** Sector `SEC-07-SGR` (Sagittarius A* Supermassive Black Hole)  
**Subsystem Restored:** Subsystem 07 — Gravitational Search Horizon & Process Reward Evaluator (`+7.1% -> 49.7%` total integrity)

## Repository Structure

```text
mission-07/
├── tot_solver.py          # Part 1: Tree-of-Thoughts BFS Beam & DFS Backtracking Solver
├── mcts_protocol.py       # Part 2: 4-Phase MCTS & Exact PUCT Selection Engine
├── prm_evaluator.py       # Part 3: Process Reward Model, Reductions & Math-Shepherd
├── horizon_pruner.py      # Part 4: Event Horizon Pruner & Reward-Hacking Defense
├── run_mission_07.py      # End-to-end integration & verification test suite
├── mission-07-doc.md      # Technical documentation source (Markdown)
├── mission-07-doc.pdf     # Compiled Moodle submission report
└── README.md              # Setup and execution guide
```

## Execution & Verification (Python 3.11+ Standard Library Only)

Zero external third-party packages are required. Run individual station modules or the unified suite:

```bash
python tot_solver.py
python mcts_protocol.py
python prm_evaluator.py
python horizon_pruner.py
python run_mission_07.py
```