"""Frame -> video scoring identical to the official protocol: softmax P(fake) per frame,
video score = mean over its frames (Config.reduce_video_predictions='mean')."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

from .data import CropDataset


@torch.no_grad()
def predict(det, table: pd.DataFrame, preprocess, batch_size: int = 64, workers: int = 2, keep_emb: bool = False):
    was_training = det.training
    det.eval()
    ds = CropDataset(table, preprocess, augment=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    probs, embs = [], []
    for x, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            f = det.features(x.cuda(non_blocking=True))
        out = det.head(f)
        probs.append(out.logits_labels.softmax(1)[:, 1].float().cpu())
        if keep_emb:
            embs.append(out.l2_embeddings.half().cpu())
    det.train(was_training)
    t = table.copy()
    t["p_fake"] = torch.cat(probs).numpy()
    return t, (torch.cat(embs) if keep_emb else None)


def video_scores(t: pd.DataFrame) -> pd.DataFrame:
    return (t.groupby("video", sort=False)
             .agg(p=("p_fake", "mean"), y=("label", "first"), method=("method", "first"), n=("p_fake", "size"))
             .reset_index())


def binary_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                acc=float((pred == y).mean()), bacc=float(balanced_accuracy_score(y, pred)),
                f1=float(f1_score(y, pred, zero_division=0)), n=int(len(y)))


def video_metrics(t: pd.DataFrame) -> dict:
    v = video_scores(t)
    return binary_metrics(v.y, v.p)
