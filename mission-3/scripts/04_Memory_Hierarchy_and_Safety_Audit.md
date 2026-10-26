# 04 — Memory Hierarchy & Safety Audit
**Mission 3 / Part 4 — Security & Ethics Sentinel · USSE Noosphere · SEC-03-M45**

Scope: FlashAttention IO complexity, a KV-cache denial-of-service (DoS) scenario, and architectural mitigations.
Status: derivations and a *modeled* scenario. No crash was observed on real hardware; all numbers below are computed from the stated assumptions.

---

## 1. FlashAttention IO Complexity Derivation

**Notation.** `N` sequence length, `d` head dimension, `M` SRAM size in elements (`d ≤ M ≤ N·d`). HBM = off-chip GPU memory (slow, large); SRAM = on-chip memory (fast, small). Cost is counted in HBM element reads + writes.

### 1.1 Standard attention
| Step | HBM traffic |
|---|---|
| read Q, K; write S = QKᵀ (N×N) | 2Nd + N² |
| read S; write P = softmax(S) (N×N) | 2N² |
| read P, V; write O | N² + 2Nd |
| **Total** | **Θ(N·d + N²)** |

For N ≫ d the N×N matrices dominate: traffic and extra memory are both Θ(N²).

### 1.2 FlashAttention (tiling + online softmax)
Split K, V into `T_c = ⌈N / B_c⌉` column blocks and Q, O into `T_r = ⌈N / B_r⌉` row blocks, with
`B_c = ⌈M / 4d⌉` and `B_r = min(⌈M / 4d⌉, d)`, so the tiles of Q, K, V, O and the score tile `S_ij` (≤ B_r×B_c ≤ M/4) fit in SRAM.

For each column block `j` (outer loop):
1. load `K_j, V_j` into SRAM — `2·B_c·d` elements; over all `j` this totals `2Nd`;
2. for each row block `i` (inner loop): load `Q_i, O_i, ℓ_i, m_i`, compute `S_ij = Q_i K_jᵀ`, update the running row-max `m` and normalizer `ℓ` (online softmax), update `O_i`, write `O_i, ℓ_i, m_i` back — about `2Nd` per outer iteration for Q and O in total.

`S` and `P` are never written to HBM. Hence

HBM traffic = Θ(N·d + T_c · N·d) = Θ(N·d + (N / (M/4d)) · N·d) = **Θ(N·d + N²d² / M)**.

### 1.3 Interpretation
* The quadratic term shrinks by a factor ≈ `M / d²`. Example: `d = 128 → d² = 16 384`; if ~100 KB of SRAM per SM holds ≈ 50 000 fp16 elements, the factor is about 3 (more with larger effective `M`; Dao et al. report 9× fewer accesses at typical sizes). The gain is a constant factor in the N² term, **not** an asymptotic elimination.
* **Precision note on the mission summary** ("HBM traffic from O(N²) to O(N)"): what becomes O(N) is the **extra memory footprint**, because the N×N matrix is never stored. HBM traffic stays quadratic in N (with the smaller coefficient `d²/M`); FLOPs remain O(N²·d).
* FlashAttention is exact (not an approximation): the online softmax reproduces the standard result up to floating-point rounding.

---

## 2. KV-Cache Denial-of-Service Simulation (modeled scenario)

### 2.1 Assumptions
* One GPU with 80 GiB HBM; 8B-parameter model in fp16 → ≈ 16 GiB weights; ≈ **64 GiB** usable for KV caches (activations/overhead ignored).
* LLaMA-3-8B-style dimensions: L = 32 layers, head_dim = 128, fp16.
* Cache per token = `2 · L · H_KV · head_dim · 2 B`: **MHA** (H_KV = 32) = 512 KiB; **GQA 4:1** (H_KV = 8) = 128 KiB; **MQA** (H_KV = 1) = 16 KiB.
* Per-session cache at N = 32 768: MHA **16 GiB**, GQA **4 GiB**, MQA **0.5 GiB**.

### 2.2 Attack narrative
1. An unauthenticated or low-privilege client opens many concurrent multi-turn dialogs; each keeps pasting large payloads and never closes the session, so every dialog's cache grows toward the context limit.
2. The server admits sessions without a memory budget and reserves a contiguous max-length buffer per session.
3. **MHA:** 64 GiB / 16 GiB → the 5th full-length session exceeds the pool; the next allocation raises `CUDA out of memory`, which can abort the process and every co-located session (blast radius = whole mainframe).
4. **GQA 4:1:** the same pool admits ≈ 16 full-length sessions; **MQA:** ≈ 128. The attack costs the adversary more, but the vector remains open.
5. **Fragmentation variant:** with max-length reservation, 100 short sessions of 1 000 tokens each strand ≈ 97% of their reservations; free memory exists but is unusable as one contiguous block (external fragmentation + internal waste).
6. **Bandwidth variant:** at 50 tok/s, one full-length MHA session reads ≈ 800 GiB/s of cache (32k context); a few such sessions saturate HBM bandwidth, degrading latency for all users even without OOM.

### 2.3 Impact (CIA lens)
Availability is the primary loss (service crash, latency collapse). Integrity/confidentiality risks arise only if an OOM path leaves caches shared or un-zeroed across tenants — an implementation property to verify.

---

## 3. Mitigation Strategy

| Defense | Mechanism | Addresses | Limits |
|---|---|---|---|
| **GQA / MQA** | fewer KV heads → 4× / 32× smaller cache | total demand, bandwidth | quality trade-off (MQA larger); needs training or uptraining |
| **PagedAttention (vLLM)** | KV stored in fixed-size blocks (e.g. 16 tokens) via a per-sequence block table; allocated on demand; prefix blocks shareable (copy-on-write) | fragmentation, over-reservation | does not reduce total demand; indirection cost; still needs admission control |
| **MLA (DeepSeek-V2/V3)** | cache a compressed latent `c_KV` (d_c = 512) plus a decoupled RoPE key (64) per token per layer; K, V reconstructed by up-projection | total demand | architecture change (needs training); extra compute; RoPE needs the decoupled path |
| **Operational controls** | per-tenant token quotas, max context, admission control with a free-block watermark, preemption/swap/eviction, idle-session timeouts, rate limits | worst-case bound, abuse | tuning; may reject legitimate long-context users |

**MLA size comparison (per token per layer, DeepSeek-V3 style):** MHA caches `2 · n_h · d_h = 2 · 128 · 128 = 32 768` values; MLA caches `512 + 64 = 576` → ≈ **57× smaller**. (Verify exact figures against the DeepSeek papers before quoting.)

**PagedAttention effect:** waste is bounded by < 1 block per sequence (≤ 15 tokens for 16-token blocks) instead of up to the full reservation; vLLM reports that prior systems wasted roughly 60–80% of KV memory.

### Recommended layered defense
1. Architecture: GQA (or MLA) to cut per-token cost.
2. Allocator: PagedAttention-style block paging to remove fragmentation.
3. Policy: hard max context, per-tenant quotas, admission control that refuses or queues new sessions below a free-block watermark, and graceful preemption instead of OOM.
4. Monitoring: alerts on cache-pool utilization, per-session growth rate, and HBM bandwidth.

---

## 4. Limitations
* Back-of-envelope model: ignores activations, CUDA graphs, fragmentation of weights, quantized caches, and speculative decoding buffers.
* The scenario is simulated on paper; do not run a real exhaustion test on shared infrastructure.
* Numeric claims about FlashAttention and MLA come from the cited papers and should be re-checked against them.

## References
Dao et al. (2022) FlashAttention, NeurIPS · Shah et al. (2024) FlashAttention-3 · Kwon et al. (2023) PagedAttention, SOSP · DeepSeek-AI (2024) DeepSeek-V2 and DeepSeek-V3 Technical Reports · Ainslie et al. (2023) GQA, EMNLP · Shazeer (2019) Multi-Query Attention.
