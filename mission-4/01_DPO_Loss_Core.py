"""
Part 1: Systems Architect - PyTorch DPO Loss Core & Log-Likelihood Extraction
Pure PyTorch implementation, no TRL / high-level wrappers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_batch_logps(logits: torch.Tensor,
                    labels: torch.Tensor,
                    average_log_prob: bool = False) -> torch.Tensor:
    """
    Compute per-sequence log-probabilities of `labels` under `logits`.

    Args:
        logits: [batch, seq_len, vocab]  (raw model outputs)
        labels: [batch, seq_len]         (token ids; padding = -100)
        average_log_prob: if True, divide by number of valid tokens

    Returns:
        [batch] tensor of log-probabilities
    """
    assert logits.shape[:-1] == labels.shape, "logits/labels shape mismatch"

    # Shift so that token t is predicted from position t-1
    logits = logits[:, :-1, :]          # [B, L-1, V]
    labels = labels[:, 1:]              # [B, L-1]

    # Mask for valid (non-padding) labels
    loss_mask = labels != -100

    # Gather log-probs of the target tokens
    log_probs = F.log_softmax(logits, dim=-1)                 # [B, L-1, V]
    target_logps = torch.gather(
        log_probs, dim=2, index=labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                                             # [B, L-1]

    # Zero out padding positions
    target_logps = target_logps * loss_mask

    if average_log_prob:
        return target_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return target_logps.sum(-1)


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss (Rafailov et al., 2023).

    L = -E[ log sigmoid( beta * ( log pi_theta(y_w|x)/pi_ref(y_w|x)
                                - log pi_theta(y_l|x)/pi_ref(y_l|x) ) ) ]
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0,
                 average_log_prob: bool = False):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.average_log_prob = average_log_prob

    def forward(self,
                policy_chosen_logps: torch.Tensor,
                policy_rejected_logps: torch.Tensor,
                reference_chosen_logps: torch.Tensor,
                reference_rejected_logps: torch.Tensor):
        """
        All inputs are [batch] tensors of sequence log-probs.
        Returns (loss, chosen_rewards, rejected_rewards, reward_margin).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios                    # Delta_w - Delta_l

        # Numerically stable -log sigmoid(beta * logits)
        losses = -F.logsigmoid(self.beta * logits)

        # Optional label smoothing (cDPO)
        if self.label_smoothing > 0:
            losses = (
                (1 - self.label_smoothing) * losses
                - self.label_smoothing * F.logsigmoid(-self.beta * logits)
            )

        # Implicit rewards for diagnostics
        chosen_rewards = self.beta * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()
        rejected_rewards = self.beta * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()

        return (
            losses.mean(),
            chosen_rewards,
            rejected_rewards,
            (chosen_rewards - rejected_rewards),
        )


# ----------------------------------------------------------------------
# Smoke tests
# ----------------------------------------------------------------------
def _fake_batch(batch=2, seq=6, vocab=11, device="cpu"):
    torch.manual_seed(0)
    logits = torch.randn(batch, seq, vocab, device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    labels[:, -1] = -100  # simulate padding
    return logits, labels


def test_get_batch_logps():
    logits, labels = _fake_batch()
    out = get_batch_logps(logits, labels)
    assert out.shape == (2,), out.shape
    avg = get_batch_logps(logits, labels, average_log_prob=True)

    # correct invariant
    mask = (labels[:, 1:] != -100)
    counts = mask.sum(-1).clamp(min=1)
    assert torch.allclose(avg * counts, out, atol=1e-6), (avg, out)
    print("[test_get_batch_logps] OK  sums:", out.tolist(), "avgs:", avg.tolist())


def test_dpo_gradient_isolation():
    """Verify reference weights receive NO gradient."""
    torch.manual_seed(0)
    vocab, hidden = 32, 16

    policy = nn.Embedding(vocab, hidden)
    ref = nn.Embedding(vocab, hidden)
    for p in ref.parameters():
        p.requires_grad_(False)          # freeze reference

    loss_fn = DPOLoss(beta=0.1)
    ids = torch.randint(0, vocab, (4, 5))

    def seq_logp(model, ids):
        emb = model(ids)                              # [B, L, H]
        logits = emb @ model.weight.t()               # [B, L, V]
        labels = ids.clone()
        return get_batch_logps(logits, labels)

    pol_w = seq_logp(policy, ids)
    pol_l = seq_logp(policy, ids.flip(0))
    with torch.no_grad():
        ref_w = seq_logp(ref, ids)
        ref_l = seq_logp(ref, ids.flip(0))

    loss, *_ = loss_fn(pol_w, pol_l, ref_w, ref_l)
    loss.backward()

    assert policy.weight.grad is not None and policy.weight.grad.abs().sum() > 0
    assert ref.weight.grad is None, "Reference model must stay frozen!"
    print(f"[test_dpo_gradient_isolation] OK  loss={loss.item():.4f}")


if __name__ == "__main__":
    test_get_batch_logps()
    test_dpo_gradient_isolation()
    print("\n01_DPO_Loss_Core.py finished successfully.")