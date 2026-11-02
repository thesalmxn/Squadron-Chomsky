"""
Part 3: Length Bias Diagnostic + SimPO Loss + VRAM Benchmark
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "dpo_core", pathlib.Path(__file__).parent / "01_DPO_Loss_Core.py"
)
dpo_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpo_core)
get_batch_logps = dpo_core.get_batch_logps


# ----------------------------------------------------------------------
# 3.1 Length correlation diagnostic
# ----------------------------------------------------------------------
def length_correlation(len_w, len_l, margin):
    """
    Pearson r between (|y_w| - |y_l|) and (Delta_w - Delta_l).
    """
    diff = np.asarray(len_w) - np.asarray(len_l)
    m = np.asarray(margin)
    if diff.std() == 0 or m.std() == 0:
        return 0.0
    return float(np.corrcoef(diff, m)[0, 1])


# ----------------------------------------------------------------------
# 3.2 SimPO loss
# ----------------------------------------------------------------------
class SimP0Loss(nn.Module):
    """
    Reference-free objective (Meng et al., 2024):
      L = -log sigmoid( beta * (avg_logp_w - avg_logp_l - gamma) )

    Per-token average log-probs; target reward margin gamma.
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, policy_chosen_logps, policy_rejected_logps):
        # Inputs must already be per-token averages.
        logratios = policy_chosen_logps - policy_rejected_logps
        loss = -F.logsigmoid(self.beta * (logratios - self.gamma))
        with torch.no_grad():
            margin = logratios.detach() * self.beta
        return loss.mean(), margin


# ----------------------------------------------------------------------
# 3.3 VRAM benchmark
# ----------------------------------------------------------------------
def _make_toy_model(vocab=512, hidden=256, n_layers=2, device="cpu"):
    return nn.Sequential(
        nn.Embedding(vocab, hidden),
        *[nn.TransformerEncoderLayer(hidden, 4, dim_feedforward=hidden * 2,
                                     batch_first=True, dropout=0.0)
          for _ in range(n_layers)],
    ).to(device)


def vram_benchmark(device="cuda"):
    """Compare peak memory: Policy+Ref (DPO) vs Policy-only (SimPO)."""
    if device == "cuda" and not torch.cuda.is_available():
        print("[vram] CUDA unavailable — skipping on-device benchmark.")
        return None

    def _peak_for(n_models):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        models = [_make_toy_model(device=device) for _ in range(n_models)]
        x = torch.randint(0, 512, (32, 64), device=device)
        for m in models:
            _ = m(x)
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak = sum(p.numel() * p.element_size()
                       for m in models for p in m.parameters()) / (1024 ** 2)
        del models, x
        gc.collect()
        return peak

    dpo_peak = _peak_for(2)
    simpo_peak = _peak_for(1)
    reduction = 100 * (1 - simpo_peak / dpo_peak) if dpo_peak else 0.0

    print(f"[vram] DPO (2 models) peak : {dpo_peak:.2f} MB")
    print(f"[vram] SimPO (1 model) peak: {simpo_peak:.2f} MB")
    print(f"[vram] Reduction           : {reduction:.1f}%")
    return {"dpo_mb": dpo_peak, "simpo_mb": simpo_peak, "reduction_pct": reduction}


# ----------------------------------------------------------------------
# Smoke tests
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Length correlation demo
    len_w = np.random.randint(50, 200, size=256)
    len_l = np.random.randint(50, 200, size=256)
    margin = 0.4 * (len_w - len_l) + np.random.randn(256) * 5
    r = length_correlation(len_w, len_l, margin)
    print(f"[length] Pearson r = {r:.3f} (positive => verbose bias)")

    # SimPO smoke test
    p_w = -torch.rand(64) * 2 - 3      # avg per-token logp
    p_l = -torch.rand(64) * 2 - 3
    loss_fn = SimP0Loss(beta=2.0, gamma=0.5)
    loss, margin = loss_fn(p_w, p_l)
    print(f"[simpo] loss={loss.item():.4f}  mean margin={margin.mean().item():.4f}")

    # VRAM benchmark (auto-fallback to CPU model-size proxy)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_benchmark(device=device)

    print("\n03_Length_Bias_and_SimPO.py finished successfully.")