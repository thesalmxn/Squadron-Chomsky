"""
Part 2: Communications & Protocol Officer
Implicit Reward Extraction & Bradley-Terry Evaluation
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from importlib import import_module
_dpo = import_module("01_DPO_Loss_Core".replace("-", "_")) if False else None
# Import safely regardless of filename hyphens:
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "dpo_core", pathlib.Path(__file__).parent / "01_DPO_Loss_Core.py"
)
dpo_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpo_core)

get_batch_logps = dpo_core.get_batch_logps


# ----------------------------------------------------------------------
# Core API
# ----------------------------------------------------------------------
def _seq_logps_from_batch(policy, ref, batch, average_log_prob=False):
    """
    batch: dict with keys
        policy_chosen_logits, policy_rejected_logits,
        ref_chosen_logits, ref_rejected_logits,
        chosen_labels, rejected_labels
    Returns dict of [B] tensors.
    """
    with torch.no_grad():
        p_w = get_batch_logps(batch["policy_chosen_logits"],
                              batch["chosen_labels"], average_log_prob)
        p_l = get_batch_logps(batch["policy_rejected_logits"],
                              batch["rejected_labels"], average_log_prob)
        r_w = get_batch_logps(batch["ref_chosen_logits"],
                              batch["chosen_labels"], average_log_prob)
        r_l = get_batch_logps(batch["ref_rejected_logits"],
                              batch["rejected_labels"], average_log_prob)
    return p_w, p_l, r_w, r_l


def evaluate_implicit_rewards(policy, ref, batch, beta=0.1):
    """
    Returns dict with implicit rewards and pairwise stats.
    (policy/ref args are kept for interface parity; logits are precomputed.)
    """
    p_w, p_l, r_w, r_l = _seq_logps_from_batch(policy, ref, batch)

    r_hat_w = beta * (p_w - r_w)
    r_hat_l = beta * (p_l - r_l)
    acc = (r_hat_w > r_hat_l).float().mean().item()

    bt_prob = torch.sigmoid(r_hat_w - r_hat_l)

    return {
        "reward_chosen": r_hat_w,
        "reward_rejected": r_hat_l,
        "pairwise_accuracy": acc,
        "bt_probabilities": bt_prob,
    }


def bradley_terry_calibration_curve(bt_probs, human_labels, n_bins=10, out_path="bt_calibration.png"):
    """
    bt_probs:     [N] predicted P(chosen > rejected)
    human_labels: [N] 1.0 if humans preferred chosen else 0.0
    Produces a reliability diagram.
    """
    bt = np.asarray(bt_probs, dtype=np.float64)
    y = np.asarray(human_labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(bt, bins) - 1

    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        xs.append(bt[mask].mean())
        ys.append(y[mask].mean())
        ns.append(int(mask.sum()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(xs, ys, "o-", label="Model")
    plt.xlabel("Predicted P(chosen ≻ rejected)")
    plt.ylabel("Empirical preference rate")
    plt.title("Bradley-Terry Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[calibration] saved reliability diagram -> {out_path}")
    return list(zip(xs, ys, ns))


# ----------------------------------------------------------------------
# Smoke test (synthetic data, no real model needed)
# ----------------------------------------------------------------------
def _synthetic_batch(n=64, L=8, V=17, device="cpu"):
    torch.manual_seed(0)
    return {
        "policy_chosen_logits":  torch.randn(n, L, V, device=device),
        "policy_rejected_logits":torch.randn(n, L, V, device=device),
        "ref_chosen_logits":     torch.randn(n, L, V, device=device),
        "ref_rejected_logits":   torch.randn(n, L, V, device=device),
        "chosen_labels":   torch.randint(0, V, (n, L), device=device),
        "rejected_labels": torch.randint(0, V, (n, L), device=device),
    }


if __name__ == "__main__":
    batch = _synthetic_batch()
    res = evaluate_implicit_rewards(None, None, batch)
    print(f"Pairwise accuracy: {res['pairwise_accuracy']:.4f}")

    # Fake human labels from the model itself for demo
    human = (res["bt_probabilities"] > 0.5).float().numpy()
    curve = bradley_terry_calibration_curve(res["bt_probabilities"], human)
    print("Calibration bins (pred, empirical, n):")
    for row in curve:
        print(f"  {row[0]:.2f}  {row[1]:.2f}  ({row[2]})")

    print("\n02_Implicit_Reward_Evaluator.py finished successfully.")