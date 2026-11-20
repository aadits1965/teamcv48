"""Losses. Batch layout from PairedStratifiedBatchSampler: rows [0, P) are reals, rows [P, 2P) are their
paired fakes (pair i = rows i and P+i); pair i belongs to manipulation method / environment i // (P / n_env).

L_GenD  = CE + 0.1 * L_align + 0.5 * L_unif         (official src.loss.Loss, unchanged)
M1 PDC  : L = L_GenD + lam * mean_i (1 - cos(d_i, u)),  d_i = z_fake_i - z_real_i (L2-normalised features),
          u = stop-grad EMA of the mean normalised difference direction (unit vector).
M2 VREx : L = L_GenD + lam(t) * Var_e[R_e],  R_e = mean CE over environment e (its fakes + their paired reals);
          lam(t) = 0 during warm-up, then lam (Krueger et al., ICML 2021, Risk Extrapolation).
"""
import torch
import torch.nn.functional as F

from src.loss import Loss, LossInputs  # official


class GenDLoss:
    def __init__(self, gend_criterion: Loss):
        self.crit = gend_criterion

    def __call__(self, logits, z, y):
        out = self.crit(LossInputs(logits_labels=logits, labels=y, l2_embeddings=z))
        return out.total, dict(ce=out.ce_labels, align=out.alignment_labels, unif=out.uniformity)


class PDC:
    """Pair-Difference Consistency (M1, novelty)."""

    def __init__(self, momentum: float = 0.9):
        self.m, self.u = momentum, None

    def __call__(self, z: torch.Tensor, P: int):
        d = z[P:2 * P] - z[:P]
        dn = F.normalize(d, dim=1, eps=1e-8)
        with torch.no_grad():
            mb = F.normalize(dn.mean(0), dim=0)
            self.u = mb if self.u is None else F.normalize(self.m * self.u + (1 - self.m) * mb, dim=0)
        cos = dn @ self.u
        return (1 - cos).mean(), dict(pdc_cos=cos.mean().item(), d_norm=d.norm(dim=1).mean().item())

    def state(self):
        return None if self.u is None else self.u.detach().cpu()


def vrex_penalty(logits: torch.Tensor, y: torch.Tensor, P: int, n_env: int):
    """M2: variance of per-environment risks (population variance, as in V-REx)."""
    ce = F.cross_entropy(logits, y, reduction="none")
    k = P // n_env
    risks = []
    for e in range(n_env):
        rows = torch.cat([torch.arange(e * k, (e + 1) * k), P + torch.arange(e * k, (e + 1) * k)]).to(logits.device)
        risks.append(ce[rows].mean())
    R = torch.stack(risks)
    return R.var(unbiased=False), {f"risk_e{e}": r.item() for e, r in enumerate(R.detach())}
