"""Detector = the OFFICIAL GenD implementation (commit 387a422) + optional M4 patch-pooling branch.

Baseline config is the paper's 'wacv-LN+L2+UnAl' experiment (src/exp/wacv_rebuttal.py):
  CLIP ViT-L/14, trainable LayerNorms {pre_layrnorm, layer_norm1, layer_norm2, post_layernorm},
  L2-normalised linear head (Head.NLinear), loss = CE + 0.1 * alignment + 0.5 * uniformity.
The features of the baseline are exactly CLIPEncoder.forward(x) = vision_model(x).pooler_output.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import GEND_ROOT  # noqa: E402
if GEND_ROOT not in sys.path:
    sys.path.insert(0, GEND_ROOT)

from src import config as C  # noqa: E402
from src.model.GenD import GenD  # noqa: E402

ARMS = ("baseline", "pdc", "vrex", "patch", "pdcvrex")


def gend_config(seed: int = 0):
    return C.Config(
        backbone=C.Backbone.CLIP_L_14,
        head=C.Head.NLinear,
        unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
        loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
        seed=seed,
        wandb=False,
    )


class AttnPatchPool(nn.Module):
    """M4 (novelty): h = CLS_pooled + gamma * sum_j softmax_j(q . p_j / sqrt(D)) p_j,
    where p_j = post_layernorm(patch token j) (the backbone's own, trainable, final LayerNorm).
    gamma is initialised to 0, so at step 0 the model is exactly the baseline."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 17)
        self.q = nn.Parameter(torch.randn(dim, generator=g) * dim ** -0.5)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = dim ** -0.5

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, D)
        attn = torch.softmax((patches @ self.q.to(patches.dtype)) * self.scale, dim=1)  # (B, T)
        pooled = (attn.unsqueeze(-1) * patches).sum(1)
        return self.gamma.to(patches.dtype) * pooled


class Detector(nn.Module):
    def __init__(self, arm: str = "baseline", seed: int = 0):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        self.gend = GenD(gend_config(seed), verbose=False)  # seeds everything + builds official modules
        dim = self.gend.feature_extractor.get_features_dim()
        self.pool = AttnPatchPool(dim, seed) if arm == "patch" else None

    @property
    def clip_processor(self):
        return self.gend.feature_extractor._preprocess

    def features(self, x: torch.Tensor) -> torch.Tensor:
        vm = self.gend.feature_extractor.vision_model
        if self.pool is None:
            return vm(x).pooler_output                      # == official CLIPEncoder.forward
        out = vm(x)
        patches = vm.post_layernorm(out.last_hidden_state[:, 1:])
        return out.pooler_output + self.pool(patches)

    def head(self, feats: torch.Tensor):
        """Official LinearProbe(normalize_inputs=True) in fp32 -> HeadOutput(logits_labels, l2_embeddings)."""
        return self.gend.model(feats.float())

    def trainable_named(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def trainable_state(self):
        return {n: p.detach().float().cpu().clone() for n, p in self.trainable_named()}

    def load_trainable_state(self, state: dict):
        missing = [n for n, _ in self.trainable_named() if n not in state]
        assert not missing, f"missing trainable tensors: {missing[:5]}"
        with torch.no_grad():
            for n, p in self.trainable_named():
                p.copy_(state[n].to(p.device, p.dtype))


def count_trainable(model: nn.Module):
    ps = [p for p in model.parameters() if p.requires_grad]
    return len(ps), sum(p.numel() for p in ps)
