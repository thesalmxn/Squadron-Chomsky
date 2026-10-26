"""
01_MultiHead_Attention_Core.py
Mission 3 / Part 1 - Systems Architect
Scaled dot-product multi-head self-attention from first principles (no nn.MultiheadAttention).

Run:  python 01_MultiHead_Attention_Core.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V, computed for h heads in parallel."""

    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # W_Q, W_K, W_V in R^{d_model x d_model} and output projection W_O
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, H, Dh) -> (B, H, T, Dh)
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, d_model)
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    @staticmethod
    def causal_mask(t: int, device=None) -> torch.Tensor:
        """True above the diagonal = positions that must NOT be attended to."""
        return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)

    def project_qkv(self, x: torch.Tensor):
        return (self._split_heads(self.w_q(x)),
                self._split_heads(self.w_k(x)),
                self._split_heads(self.w_v(x)))

    def forward(self, x: torch.Tensor, causal: bool = True, return_weights: bool = False):
        q, k, v = self.project_qkv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            scores = scores.masked_fill(self.causal_mask(x.size(1), x.device), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = self.w_o(self._merge_heads(torch.matmul(weights, v)))
        return (out, weights) if return_weights else out


# ----------------------------------------------------------------------------- tests
def test_shapes_and_mask():
    torch.manual_seed(0)
    layer = MultiHeadSelfAttention(64, 8)
    x = torch.randn(2, 10, 64)
    out, w = layer(x, return_weights=True)
    assert out.shape == (2, 10, 64)
    assert w.shape == (2, 8, 10, 10)
    assert torch.allclose(w.sum(-1), torch.ones(2, 8, 10), atol=1e-5), "rows must sum to 1"
    upper = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert (w[..., upper] == 0).all(), "masked weights must be exactly zero"
    print("[PASS] shapes, row-sums = 1, masked weights exactly 0")


def test_causality():
    torch.manual_seed(1)
    layer = MultiHeadSelfAttention(64, 8)
    x = torch.randn(1, 12, 64)
    x2 = x.clone()
    x2[:, -1] += torch.randn(64)  # perturb the LAST token only
    o1, o2 = layer(x), layer(x2)
    assert torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-6), "future token leaked into the past"
    assert not torch.allclose(o1[:, -1], o2[:, -1], atol=1e-6)
    print("[PASS] causality: perturbing token t does not change outputs < t")


def test_matches_reference():
    torch.manual_seed(2)
    layer = MultiHeadSelfAttention(64, 8)
    x = torch.randn(2, 16, 64)
    q, k, v = layer.project_qkv(x)
    ref = layer.w_o(layer._merge_heads(F.scaled_dot_product_attention(q, k, v, is_causal=True)))
    assert torch.allclose(layer(x), ref, atol=1e-5)
    print("[PASS] matches torch.nn.functional.scaled_dot_product_attention (reference only)")


def test_variance_preservation(n_samples: int = 50_000, tol: float = 0.05):
    """For q,k ~ N(0,1) i.i.d.: Var(q.k) = d_k, so Var(q.k / sqrt(d_k)) = 1."""
    torch.manual_seed(3)
    print("\nVariance preservation (n = %d samples, tolerance +/-%.0f%%)" % (n_samples, tol * 100))
    print(f"{'d_k':>5} | {'Var(unscaled)':>14} | {'expected':>9} | {'Var(scaled)':>12} | status")
    print("-" * 62)
    for d_k in (16, 64, 256):
        q, k = torch.randn(n_samples, d_k), torch.randn(n_samples, d_k)
        s = (q * k).sum(-1)
        v_raw, v_scaled = s.var().item(), (s / math.sqrt(d_k)).var().item()
        ok = abs(v_raw / d_k - 1) < tol and abs(v_scaled - 1) < tol
        print(f"{d_k:>5} | {v_raw:>14.3f} | {d_k:>9d} | {v_scaled:>12.4f} | {'PASS' if ok else 'FAIL'}")
        assert ok, f"variance test failed for d_k={d_k}"

    # Why it matters: softmax saturation (mean of the largest attention weight over 64 keys)
    d_k, n_keys = 64, 64
    q, k = torch.randn(2000, d_k), torch.randn(2000, n_keys, d_k)
    raw = torch.einsum("bd,bnd->bn", q, k)
    m_raw = torch.softmax(raw, -1).max(-1).values.mean().item()
    m_scl = torch.softmax(raw / math.sqrt(d_k), -1).max(-1).values.mean().item()
    print(f"\nMean max softmax weight over {n_keys} keys (d_k={d_k}): "
          f"unscaled = {m_raw:.3f}, scaled = {m_scl:.3f}  (unscaled is closer to one-hot)")
    print("[PASS] variance preservation")


def main():
    print("=" * 62)
    print(" Mission 3 / Part 1 : Multi-Head Self-Attention Core")
    print(f" torch {torch.__version__}")
    print("=" * 62)
    test_shapes_and_mask()
    test_causality()
    test_matches_reference()
    test_variance_preservation()
    print("\nAll Part 1 checks passed.")


if __name__ == "__main__":
    main()
