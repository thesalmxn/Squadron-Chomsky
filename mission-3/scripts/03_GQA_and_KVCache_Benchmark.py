"""
03_GQA_and_KVCache_Benchmark.py
Mission 3 / Part 3 - Data & Payload Specialist
Grouped-Query Attention, a stateful KV-cache, and an MHA / GQA 4:1 / MQA 32:1 benchmark.

Run:  python 03_GQA_and_KVCache_Benchmark.py [--device cpu|cuda] [--iters 10] [--max-n 32768]
      python 03_GQA_and_KVCache_Benchmark.py --skip-latency      (analytic memory table only)
Writes benchmark_results.json next to the script.
"""
import argparse
import gc
import json
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LENS = (1024, 4096, 16384, 32768)
CONFIGS = {"MHA (32:32)": 32, "GQA 4:1 (32:8)": 8, "MQA 32:1 (32:1)": 1}   # name -> H_KV
H_Q, HEAD_DIM, D_MODEL, N_LAYERS, TOK_PER_S = 32, 128, 4096, 32, 50


# ----------------------------------------------------------------------------- RoPE (compact copy)
def precompute_freqs_cis(head_dim, max_pos, base=10000.0, device=None):
    theta = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), theta)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs_cis):
    orig = x.dtype
    xc = torch.view_as_complex(x.float().contiguous().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(orig)


# ----------------------------------------------------------------------------- KV cache
class KVCache:
    """Pre-allocated per-layer cache; append() writes new (K_t, V_t) slices at the current length."""

    def __init__(self, batch, n_kv_heads, max_len, head_dim, dtype=torch.float32, device=None):
        shape = (batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len, self.length = max_len, 0

    def update(self, k_new, v_new):
        t = k_new.size(2)
        if self.length + t > self.max_len:
            raise RuntimeError("KV cache is full")
        self.k[:, :, self.length:self.length + t] = k_new
        self.v[:, :, self.length:self.length + t] = v_new
        self.length += t
        return self.k[:, :, :self.length], self.v[:, :, :self.length]

    def reset(self):
        self.length = 0

    @property
    def allocated_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


# ----------------------------------------------------------------------------- GQA layer
class GQAAttention(nn.Module):
    """h_q query heads share h_kv key/value heads (group size g = h_q / h_kv)."""

    def __init__(self, d_model=D_MODEL, n_q_heads=H_Q, n_kv_heads=8, bias=False):
        super().__init__()
        if d_model % n_q_heads or n_q_heads % n_kv_heads:
            raise ValueError("need d_model % h_q == 0 and h_q % h_kv == 0")
        self.n_q, self.n_kv = n_q_heads, n_kv_heads
        self.group = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.w_q = nn.Linear(d_model, n_q_heads * self.head_dim, bias=bias)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.w_o = nn.Linear(n_q_heads * self.head_dim, d_model, bias=bias)

    def project(self, x, start_pos=0, freqs_cis=None):
        b, t, _ = x.shape
        q = self.w_q(x).view(b, t, self.n_q, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        if freqs_cis is not None:                       # RoPE on q and k only (never on v)
            fc = freqs_cis[start_pos:start_pos + t]
            q, k = apply_rope(q, fc), apply_rope(k, fc)
        return q, k, v

    def forward(self, x, cache=None, start_pos=0, freqs_cis=None):
        q, k, v = self.project(x, start_pos, freqs_cis)
        if cache is not None:                           # cache stores K AFTER rope, un-expanded
            k, v = cache.update(k, v)
        k = torch.repeat_interleave(k, self.group, dim=1)   # (B, h_kv, T, D) -> (B, h_q, T, D)
        v = torch.repeat_interleave(v, self.group, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        t_q, t_k = q.size(2), k.size(2)
        q_idx = torch.arange(t_q, device=x.device) + start_pos
        mask = torch.arange(t_k, device=x.device)[None, :] > q_idx[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        b = x.size(0)
        return self.w_o(ctx.transpose(1, 2).contiguous().view(b, t_q, self.n_q * self.head_dim))


# ----------------------------------------------------------------------------- correctness tests
def test_head_grouping():
    kv = torch.randn(1, 2, 5, 8)
    exp = torch.repeat_interleave(kv, 4, dim=1)
    assert exp.shape == (1, 8, 5, 8)
    for h in range(8):
        assert torch.equal(exp[:, h], kv[:, h // 4]), "query head h must use KV head h // g"
    print("[PASS] repeat_interleave maps query head h -> KV head h // group")


def test_matches_reference():
    torch.manual_seed(0)
    for n_kv in (8, 2, 1):                       # MHA, GQA 4:1, MQA
        layer = GQAAttention(64, 8, n_kv)
        freqs = precompute_freqs_cis(8, 32)
        x = torch.randn(2, 12, 64)
        q, k, v = layer.project(x, 0, freqs)
        k, v = (torch.repeat_interleave(t, layer.group, dim=1) for t in (k, v))
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = layer.w_o(ref.transpose(1, 2).contiguous().view(2, 12, 64))
        assert torch.allclose(layer(x, freqs_cis=freqs), ref, atol=1e-5)
    print("[PASS] GQA output matches scaled_dot_product_attention for h_kv = 8, 2, 1")


def test_cache_equivalence():
    """Incremental decoding with the cache must equal a full forward pass."""
    torch.manual_seed(1)
    layer = GQAAttention(64, 8, 2).eval()
    freqs = precompute_freqs_cis(8, 32)
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        full = layer(x, freqs_cis=freqs)
        cache = KVCache(1, 2, 32, 8)
        outs = [layer(x[:, i:i + 1], cache=cache, start_pos=i, freqs_cis=freqs) for i in range(12)]
        inc = torch.cat(outs, dim=1)
        assert cache.length == 12
        assert torch.allclose(full, inc, atol=1e-5), f"max diff {(full - inc).abs().max():.2e}"
        cache2 = KVCache(1, 2, 32, 8)                     # prefill 5 tokens, then decode the rest
        pre = layer(x[:, :5], cache=cache2, start_pos=0, freqs_cis=freqs)
        rest = [layer(x[:, i:i + 1], cache=cache2, start_pos=i, freqs_cis=freqs) for i in range(5, 12)]
        assert torch.allclose(full, torch.cat([pre] + rest, dim=1), atol=1e-5)
    print("[PASS] cached incremental decoding == full recomputation (token-by-token and prefill+decode)")


# ----------------------------------------------------------------------------- analytic memory model
def kv_bytes(n_kv, n, bytes_per=2, layers=N_LAYERS, head_dim=HEAD_DIM):
    """2 (K and V) x layers x h_kv x head_dim x N x bytes, for one sequence."""
    return 2 * layers * n_kv * head_dim * n * bytes_per


def analytic_table():
    rows = {}
    print(f"\nAnalytic KV-cache table  (batch=1, L={N_LAYERS}, head_dim={HEAD_DIM}, fp16, "
          f"{TOK_PER_S} tok/s; 1 MB = 2^20 B, 1 GB = 2^30 B)")
    print(f"{'N':>7} | " + " | ".join(f"{n + ' MB':>17}" for n in CONFIGS) + " | "
          + " | ".join(f"{n.split(' ')[0] + ' GB/s':>10}" for n in CONFIGS))
    print("-" * 105)
    for n in SEQ_LENS:
        mem = [kv_bytes(h, n) / 2 ** 20 for h in CONFIGS.values()]
        bw = [kv_bytes(h, n) * TOK_PER_S / 2 ** 30 for h in CONFIGS.values()]
        print(f"{n:>7} | " + " | ".join(f"{m:>17,.0f}" for m in mem) + " | "
              + " | ".join(f"{b:>10.2f}" for b in bw))
        rows[n] = {name: {"mem_MB": m, "bw_GBps": b} for name, m, b in zip(CONFIGS, mem, bw)}
    return rows


def check_cache_footprint(dtype):
    """Allocated cache bytes of one layer must equal the formula (per layer, actual dtype)."""
    es = torch.empty((), dtype=dtype).element_size()
    for name, h in CONFIGS.items():
        c = KVCache(1, h, 1024, HEAD_DIM, dtype)
        assert c.allocated_bytes == kv_bytes(h, 1024, es, layers=1), name
    print(f"[PASS] allocated cache bytes match the formula for all three configs ({dtype})")


# ----------------------------------------------------------------------------- latency benchmark
def time_decode_step(n_kv, n, device, dtype, warmup, iters):
    layer = GQAAttention(D_MODEL, H_Q, n_kv).to(device=device, dtype=dtype).eval()
    cache = KVCache(1, n_kv, n, HEAD_DIM, dtype, device)
    cache.k.normal_()
    cache.v.normal_()
    freqs = precompute_freqs_cis(HEAD_DIM, n, device=device)
    x = torch.randn(1, 1, D_MODEL, device=device, dtype=dtype)
    times = []
    with torch.no_grad():
        for i in range(warmup + iters):
            cache.length = n - 1                      # decode token number n against n-1 cached tokens
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            layer(x, cache=cache, start_pos=n - 1, freqs_cis=freqs)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    del layer, cache, freqs, x
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return statistics.median(times)


def latency_table(device, dtype, warmup, iters, max_n):
    results = {}
    print(f"\nMeasured decoding latency for ONE attention layer (median of {iters} runs, "
          f"device={device}, dtype={dtype}); 'x{N_LAYERS}' is a linear extrapolation, not a measurement")
    print(f"{'N':>7} | " + " | ".join(f"{n + ' ms':>17}" for n in CONFIGS) + " | "
          + " | ".join(f"{n.split(' ')[0] + ' x' + str(N_LAYERS):>10}" for n in CONFIGS))
    print("-" * 105)
    for n in SEQ_LENS:
        row = {}
        for name, h in CONFIGS.items():
            if n > max_n:
                row[name] = None
                continue
            try:
                row[name] = time_decode_step(h, n, device, dtype, warmup, iters)
            except (RuntimeError, MemoryError) as e:     # includes torch.cuda.OutOfMemoryError
                row[name] = None
                print(f"   ({name}, N={n}: skipped - {type(e).__name__})")
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        fmt = lambda v: f"{v:>17.3f}" if v is not None else f"{'n/a':>17}"
        fmt2 = lambda v: f"{v * N_LAYERS:>10.1f}" if v is not None else f"{'n/a':>10}"
        print(f"{n:>7} | " + " | ".join(fmt(row[k]) for k in CONFIGS) + " | "
              + " | ".join(fmt2(row[k]) for k in CONFIGS))
        results[n] = row
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=32768)
    ap.add_argument("--skip-latency", action="store_true")
    args, _ = ap.parse_known_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 62)
    print(" Mission 3 / Part 3 : GQA & KV-Cache Benchmark")
    print(f" torch {torch.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print("=" * 62)
    test_head_grouping()
    test_matches_reference()
    test_cache_equivalence()
    check_cache_footprint(dtype)

    out = {"device": device, "dtype": str(dtype), "torch": torch.__version__,
           "analytic": analytic_table()}
    if not args.skip_latency:
        out["latency_ms_per_layer"] = latency_table(device, dtype, args.warmup, args.iters, args.max_n)
    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[saved] benchmark_results.json")


if __name__ == "__main__":
    main()
