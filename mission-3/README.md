# Mission 3 — Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache Optimization

**Squadron Chomsky · USSE Noosphere Bridge Crew**
Conversational AI — Winter Semester 2026/27 · Week 3 · Waypoint `SEC-03-M45` (Pleiades Open Cluster)
AURA Subsystem 03: *Distributed Multi-Head Attention Mesh & KV-Cache Transceiver*

This repository contains our solution to Problem Set 03: a from-scratch PyTorch reconstruction of the modern Transformer attention pipeline, plus a memory-hierarchy and safety audit.

## Team

| Name | Role |
|---|---|
| Nikhil Sakharam Thakare | Systems Architect (Lead Engineer) |
| Mohammad Salman Sunakal | Communications & Protocol Officer |
| Vishal Sudarshan K | Security & Ethics Sentinel |
| Abdul Haleem Haseeb | Data & Payload Specialist |

<!-- Adjust roles above to match your actual split. -->

## What's inside

```
docs/weeks/week-03/problem_set/
├── 01_MultiHead_Attention_Core.py        # Part 1: multi-head self-attention from scratch
├── 02_RoPE_Positional_Engine.py          # Part 2: Rotary Positional Embeddings
├── 03_GQA_and_KVCache_Benchmark.py       # Part 3: GQA layer, KV-cache, MHA/GQA/MQA benchmark
├── 04_Memory_Hierarchy_and_Safety_Audit.md  # Part 4: FlashAttention IO + KV-cache DoS audit
├── ps03_solution.ipynb                   # Notebook version of the solution
└── Mission_3_Technical_Report.md         # Full technical report
```

| Part | Deliverable | Summary |
|---|---|---|
| 1 | `01_MultiHead_Attention_Core.py` | `MultiHeadSelfAttention(nn.Module)` with `W_Q, W_K, W_V, W_O`, head reshaping to `(batch, heads, seq, head_dim)`, causal masking, and a variance-preservation unit test for the `1/√d_k` scaling. No `nn.MultiheadAttention`. |
| 2 | `02_RoPE_Positional_Engine.py` | `apply_rope(x, freqs_cis)` with `θ_i = base^(−2(i−1)/d)`, relative-invariance validation, and a RoPE-vs-ALiBi attention decay plot. |
| 3 | `03_GQA_and_KVCache_Benchmark.py` | `GQAAttention` (e.g. `H_Q=32`, `H_KV=8`), `KVCache` class, and a benchmark of MHA vs GQA 4:1 vs MQA 32:1 at N ∈ {1024, 4096, 16384, 32768}. |
| 4 | `04_Memory_Hierarchy_and_Safety_Audit.md` | FlashAttention IO-complexity derivation, KV-cache denial-of-service scenario, and evaluation of PagedAttention and MLA. |

## Requirements

- Python 3.10+
- PyTorch 2.4+
- matplotlib (for plots)
- Jupyter (only for `ps03_solution.ipynb`)

```bash
pip install torch matplotlib jupyter
```

A GPU is optional but recommended for realistic latency numbers. No API keys are required.

## How to run

From `docs/weeks/week-03/problem_set/`:

```bash
python 01_MultiHead_Attention_Core.py
python 02_RoPE_Positional_Engine.py
python 03_GQA_and_KVCache_Benchmark.py
```

Each script prints formatted terminal output; Parts 2 and 3 also produce plots/tables. To open the notebook:

```bash
jupyter notebook ps03_solution.ipynb
```

## Key concepts

- **Scaled dot-product attention:** `softmax(QKᵀ/√d_k)·V`. For unit-variance inputs, `q·k` has variance `d_k`; dividing by `√d_k` restores unit variance.
- **RoPE:** rotates pairs of feature dimensions by position-dependent angles, so `⟨R_m q, R_n k⟩` depends only on `m − n`.
- **GQA / MQA:** `H_Q` query heads share `H_KV` key/value heads, shrinking the KV-cache by a factor of `H_Q / H_KV`.
- **KV-cache:** retains past keys/values during autoregressive decoding to avoid recomputing the prefix.
- **FlashAttention:** tiles attention in on-chip SRAM, reducing HBM accesses from `Θ(Nd + N²)` to `O(Nd + N²d²/M)` and activation memory from `Θ(N²)` to `Θ(N)`.

## Benchmark snapshot (analytical)

Assumptions: LLaMA-3-8B-like model — 32 layers, `d_head = 128`, fp16, batch 1; cache read once per token at 50 tokens/s.

| N | MHA cache (MB) | GQA 4:1 (MB) | MQA 32:1 (MB) | MHA traffic (GB/s) | GQA (GB/s) | MQA (GB/s) |
|---|---|---|---|---|---|---|
| 1,024 | 512 | 128 | 16 | 25.00 | 6.25 | 0.78 |
| 4,096 | 2,048 | 512 | 64 | 100.00 | 25.00 | 3.13 |
| 16,384 | 8,192 | 2,048 | 256 | 400.00 | 100.00 | 12.50 |
| 32,768 | 16,384 | 4,096 | 512 | 800.00 | 200.00 | 25.00 |

These figures follow from tensor shapes. **Decoding latency is hardware-dependent and measured by the script**; see the report for how to record it.

## Results status

- Memory footprint and bandwidth tables: derived analytically (above).
- Latency, variance-test values, RoPE invariance checks and plots: produced by running the scripts on your own machine — record device, PyTorch version and dtype alongside any numbers you report.
- The models are untrained; no claims are made about language quality.

## Limitations

Educational prototype only. Bandwidth figures are a first-order model that excludes weight reads; the denial-of-service scenario is a calculation, not an experiment on real infrastructure; and the code must not be used for any safety-critical purpose.

## References

- Vaswani et al. (2017), *Attention Is All You Need*
- Su et al. (2021), *RoFormer* (RoPE) — arXiv:2104.09864
- Press et al. (2022), *ALiBi*
- Shazeer (2019), *Multi-Query Attention* — arXiv:1911.02150
- Ainslie et al. (2023), *GQA*
- Dao et al. (2022), *FlashAttention*; Shah et al. (2024), *FlashAttention-3* — arXiv:2407.08608
- Kwon et al. (2023), *PagedAttention / vLLM*
- DeepSeek-AI (2024), *DeepSeek-V3 Technical Report* (MLA) — arXiv:2412.19437

Course repository: https://github.com/Unisvet/couse-c-ai

## Commit convention

```
feat(ps03): restore AURA Subsystem 03 Spatial Attention Mesh (SEC-03-M45)
```
