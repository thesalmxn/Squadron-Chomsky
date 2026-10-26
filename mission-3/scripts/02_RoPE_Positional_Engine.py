"""
02_RoPE_Positional_Engine.py
Mission 3 / Part 2 - Communications & Protocol Officer
Rotary Positional Embeddings (Su et al., 2021): implementation, relative-invariance
validation, and a RoPE vs ALiBi decay comparison plot.

Run:  python 02_RoPE_Positional_Engine.py      (saves rope_vs_alibi_decay.png)
"""
import math

import matplotlib

if "get_ipython" not in globals():  # headless when run as a script, inline in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def precompute_freqs_cis(head_dim: int, max_pos: int, base: float = 10000.0,
                         dtype=torch.float32, device=None) -> torch.Tensor:
    """freqs_cis[m, i] = exp(j * m * theta_i), theta_i = base^(-2(i-1)/d), i = 1..d/2."""
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    two_i_minus_2 = torch.arange(0, head_dim, 2, dtype=dtype, device=device)  # = 2(i-1)
    theta = base ** (-two_i_minus_2 / head_dim)
    pos = torch.arange(max_pos, dtype=dtype, device=device)
    angles = torch.outer(pos, theta)                       # (max_pos, d/2)
    return torch.polar(torch.ones_like(angles), angles)    # unit-modulus complex numbers


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive coordinate pairs of x by the angles in freqs_cis.

    x:         (..., T, head_dim)   e.g. (B, H, T, Dh)
    freqs_cis: (T, head_dim // 2)   rows already sliced to the positions of x
    """
    orig = x.dtype
    work = x if x.dtype in (torch.float32, torch.float64) else x.float()
    xc = torch.view_as_complex(work.contiguous().reshape(*work.shape[:-1], -1, 2))
    out = xc * freqs_cis.to(xc.dtype)                      # complex multiply = 2-D rotation
    return torch.view_as_real(out).flatten(-2).to(orig)


# ----------------------------------------------------------------------------- validation
def rope_score(q, k, m, n, freqs):
    qr = apply_rope(q.view(1, 1, 1, -1), freqs[m:m + 1])
    kr = apply_rope(k.view(1, 1, 1, -1), freqs[n:n + 1])
    return (qr * kr).sum().item()


def test_basic_properties():
    torch.manual_seed(0)
    d = 64
    freqs = precompute_freqs_cis(d, 128, dtype=torch.float64)
    x = torch.randn(2, 4, 16, d, dtype=torch.float64)
    xr = apply_rope(x, freqs[:16])
    assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-10), "rotation must preserve norm"
    assert torch.allclose(apply_rope(x[..., :1, :], freqs[:1]), x[..., :1, :], atol=1e-12), \
        "position 0 must be the identity"
    print("[PASS] norm preserved; position 0 is the identity")


def test_relative_invariance(trials: int = 500, d: int = 64):
    """<RoPE(q,m), RoPE(k,n)> == <RoPE(q,m+delta), RoPE(k,n+delta)>."""
    torch.manual_seed(1)
    freqs64 = precompute_freqs_cis(d, 4096, dtype=torch.float64)
    freqs32 = precompute_freqs_cis(d, 4096, dtype=torch.float32)
    max64 = max32 = 0.0
    for _ in range(trials):
        q64, k64 = torch.randn(d, dtype=torch.float64), torch.randn(d, dtype=torch.float64)
        m, n, delta = (int(t) for t in torch.randint(0, 500, (3,)))
        max64 = max(max64, abs(rope_score(q64, k64, m, n, freqs64) -
                               rope_score(q64, k64, m + delta, n + delta, freqs64)))
        q32, k32 = q64.float(), k64.float()
        max32 = max(max32, abs(rope_score(q32, k32, m, n, freqs32) -
                               rope_score(q32, k32, m + delta, n + delta, freqs32)))
    print(f"[invariance] {trials} random (q,k,m,n,delta) trials, d={d}")
    print(f"             max |difference| float64 = {max64:.2e}, float32 = {max32:.2e}")
    assert max64 < 1e-8, "float64 invariance violated"
    assert max32 < 5e-2, "float32 invariance violated beyond rounding"

    # Control: changing the RELATIVE distance must change the score.
    q, k = torch.randn(d, dtype=torch.float64), torch.randn(d, dtype=torch.float64)
    changed = abs(rope_score(q, k, 10, 5, freqs64) - rope_score(q, k, 10, 7, freqs64))
    assert changed > 1e-3, "score should depend on m - n"
    print(f"[PASS] invariant under a common shift; control (m-n changed) differs by {changed:.3f}")


# ----------------------------------------------------------------------------- RoPE vs ALiBi
def rope_score_curve(head_dim: int, max_dist: int, base: float = 10000.0) -> torch.Tensor:
    """Score vs relative distance r for q = k = all-ones (the illustration used by Su et al.).
    Each plane contributes 2*cos(r*theta_i); normalised so that r = 0 gives 1."""
    theta = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    r = torch.arange(max_dist, dtype=torch.float64)
    return (2 * torch.cos(torch.outer(r, theta))).sum(-1) / head_dim


def causal_softmax(logits: torch.Tensor) -> torch.Tensor:
    t = logits.size(0)
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    return torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1)


def make_plot(path: str = "rope_vs_alibi_decay.png", head_dim: int = 64, T: int = 128):
    dist = torch.arange(T, dtype=torch.float64)
    rel = (dist[:, None] - dist[None, :]).abs().long()      # |i - j|

    curve = rope_score_curve(head_dim, 512)
    slopes = [2.0 ** (-(h + 1)) for h in range(0, 8, 2)]    # 1/2, 1/8, 1/32, 1/128

    # Content-neutral attention maps (q = k = ones for RoPE, zero content logits for ALiBi)
    rope_logits = rope_score_curve(head_dim, T)[rel] * head_dim / math.sqrt(head_dim)
    alibi_logits = -0.25 * rel.double()
    rope_w, alibi_w = causal_softmax(rope_logits), causal_softmax(alibi_logits)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(curve.numpy(), lw=1.2)
    ax[0, 0].set(title="RoPE: score vs relative distance (q = k = ones, d=%d)" % head_dim,
                 xlabel="relative distance |m - n|", ylabel="normalised score")
    ax[0, 0].axhline(0, color="gray", lw=0.5)
    for s in slopes:
        ax[0, 1].plot((-s * torch.arange(512)).numpy(), label=f"slope {s:g}")
    ax[0, 1].set(title="ALiBi: linear distance penalty -m*|i-j|",
                 xlabel="relative distance |i - j|", ylabel="bias added to logits")
    ax[0, 1].legend()
    im1 = ax[1, 0].imshow(rope_w.numpy(), cmap="magma", aspect="auto")
    ax[1, 0].set(title="RoPE attention weights (content-neutral, causal)", xlabel="key j", ylabel="query i")
    fig.colorbar(im1, ax=ax[1, 0])
    im2 = ax[1, 1].imshow(alibi_w.numpy(), cmap="magma", aspect="auto")
    ax[1, 1].set(title="ALiBi attention weights (slope 0.25, causal)", xlabel="key j", ylabel="query i")
    fig.colorbar(im2, ax=ax[1, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[saved] {path}")
    if "get_ipython" in globals():
        plt.show()
    plt.close(fig)

    # Numeric summary for the report
    print("\nRoPE normalised score at selected distances (q = k = ones):")
    for r in (0, 1, 8, 32, 128, 256, 511):
        print(f"   r = {r:>3d}: {curve[r].item():+.3f}")
    print("ALiBi bias at distance 128 for slopes", ", ".join(f"{s:g}: {-s * 128:.1f}" for s in slopes))
    print("Note: the RoPE curve oscillates inside a decaying envelope; ALiBi is a monotone linear ramp.")


def main():
    print("=" * 62)
    print(" Mission 3 / Part 2 : RoPE Positional Engine")
    print(f" torch {torch.__version__}")
    print("=" * 62)
    test_basic_properties()
    test_relative_invariance()
    make_plot()
    print("\nAll Part 2 checks passed.")


if __name__ == "__main__":
    main()
