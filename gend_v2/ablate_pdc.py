"""Ablation for M1 (PDC): identical training (same batches, loss, weight lambda, schedule), but each
fake is paired with a NON-matching real from the same batch: reals are cyclically shifted by a random
offset k in [1, P-1] at every step (a derangement), which destroys the source-video correspondence.
If PDC's effect comes from the pairing, it should vanish here.

  python -m gend_v2.ablate_pdc --seed 0 --out .../runs_ablation/pdc_shuffled_s0
"""
import argparse
import json
from pathlib import Path

import torch

from . import train as T
from .losses import PDC


class ShuffledPDC(PDC):
    def __init__(self, momentum=0.9, seed=0):
        super().__init__(momentum)
        self.g = torch.Generator().manual_seed(10_000 + seed)

    def __call__(self, z, P):
        k = int(torch.randint(1, P, (1,), generator=self.g))
        idx = ((torch.arange(P) + k) % P).to(z.device)
        return super().__call__(torch.cat([z[:P][idx], z[P:]]), P)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--lam", type=float, default=0.5)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "ablation.json").write_text(json.dumps(
        dict(ablation="pdc_shuffled_pairs", base_arm="pdc", seed=args.seed, lam=args.lam,
             note="reals cyclically shifted within batch by random k in [1,P-1]; generator not resumed"), indent=2))
    T.PDC = lambda: ShuffledPDC(seed=args.seed)
    T.main(["--arm", "pdc", "--seed", str(args.seed), "--pairs", "24", "--epochs", "6",
            "--lam", str(args.lam), "--out", args.out, "--resume"])


if __name__ == "__main__":
    main()
