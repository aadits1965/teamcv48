"""Stage-2 feasibility checks for all arms (run once before any training).

  python -m gend_v2.sanity --out /content/drive/MyDrive/GenD_PCR_Project/v2/step6

Checks: data/pairing integrity, identical batches across arms, trainable parameter sets, finite losses,
gradients only on trainable tensors, each novelty term actually active, M4 == baseline at init (gamma=0),
peak VRAM + GPU throughput for several batch sizes, data-loader throughput, validation-pass time,
and a tiny overfitting test for the baseline.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as D
from .evalutil import predict, video_metrics
from .losses import PDC, GenDLoss, vrex_penalty
from .model import Detector, count_trainable
from .train import batch_fingerprint, enable_grad_ckpt


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def first_batch(ds, sampler, n=1):
    it = iter(sampler)
    out = []
    for _ in range(n):
        idxs = next(it)
        items = [ds[i] for i in idxs]
        x = torch.stack([t[0] for t in items])
        y = torch.tensor([t[1] for t in items])
        out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {}

    # ---------------- [A] data integrity
    idx = D.load_index()
    D.ensure_crops([f"train__{m}" for m in ["real"] + D.FF_METHODS] + ["val__real", "val__FaceShifter"], idx=idx)
    trn, val = D.frame_table(idx, "train"), D.frame_table(idx, "val")
    missing = [p for p in trn.path.tolist() + val.path.tolist() if not (D.LOCAL_CROPS / p).exists()]
    man = pd.read_csv(D.DRIVE_V2 / "manifest" / "manifest_v2.csv", dtype={"video_id": str, "target_id": str})
    nf = dict(zip(man.rel, man.n_frames))
    fk = man[man.label == 1].copy()
    fk["real_rel"] = "original_sequences/youtube/c23/videos/" + fk.video_id.str.split("_").str[0] + ".mp4"
    fk["same_len"] = [nf[a] == nf[b] for a, b in zip(fk.rel, fk.real_rel)]
    fk["len_diff"] = [nf[a] - nf[b] for a, b in zip(fk.rel, fk.real_rel)]
    R["A_data"] = dict(train_crops=len(trn), val_crops=len(val), missing_files=len(missing),
                       train_by_method=trn.method.value_counts().to_dict(),
                       val_by_method=val.method.value_counts().to_dict(),
                       fake_same_frame_count_as_real_by_method=fk.groupby("method").same_len.mean().round(3).to_dict(),
                       fake_minus_real_frames_by_method=fk.groupby("method").len_diff.describe()[["min", "50%", "max"]]
                       .round(0).to_dict("index"))
    print("[A] data:", json.dumps(R["A_data"], indent=1, default=str), flush=True)

    sampler = D.PairedStratifiedBatchSampler(trn, pairs_per_batch=args.pairs, seed=0)
    R["A_pairs"] = dict(exact_pair_fraction=sampler.exact, steps_per_epoch=len(sampler),
                        fingerprint=batch_fingerprint(sampler))
    b0 = next(iter(sampler))
    P = args.pairs
    lab = [int(trn.label[i % len(trn)]) for i in b0]
    tg = [trn.target[i % len(trn)] for i in b0]
    R["A_pairs"]["layout_ok"] = (lab[:P] == [0] * P and lab[P:] == [1] * P and tg[:P] == tg[P:])
    print("[A] pairing:", R["A_pairs"], flush=True)

    # ---------------- [B] per-arm checks
    ref_feats = None
    for arm in ["baseline", "pdc", "vrex", "patch"]:
        free()
        t0 = time.time()
        det = Detector(arm, seed=0).cuda().train()
        load_s = time.time() - t0
        pre = D.make_preprocess(det.clip_processor)
        ds = D.CropDataset(trn, pre, augment=D.official_augmentations(), seed=0)
        smp = D.PairedStratifiedBatchSampler(trn, pairs_per_batch=P, seed=0)
        (x, y), = first_batch(ds, smp)
        x, y = x.cuda(), y.cuda()
        crit = GenDLoss(det.gend.criterion)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = det.features(x)
        ho = det.head(feats)
        loss, parts = crit(ho.logits_labels, ho.l2_embeddings, y)
        extra, more = torch.zeros((), device="cuda"), {}
        if arm == "pdc":
            extra, more = PDC()(ho.l2_embeddings, P)
        if arm == "vrex":
            extra, more = vrex_penalty(ho.logits_labels, y, P, 4)
        lam = {"baseline": 0, "pdc": 0.5, "vrex": 10.0, "patch": 0}[arm]
        (loss + lam * extra).backward()
        tr = det.trainable_named()
        no_grad = [n for n, p in tr if p.grad is None]
        zero_grad = [n for n, p in tr if p.grad is not None and p.grad.abs().sum().item() == 0]
        nonfinite = [n for n, p in tr if p.grad is not None and not torch.isfinite(p.grad).all()]
        frozen_with_grad = [n for n, p in det.named_parameters() if not p.requires_grad and p.grad is not None]
        n_t, n_p = count_trainable(det)
        r = dict(load_s=round(load_s, 1), trainable_tensors=n_t, trainable_params=n_p,
                 loss=round(loss.item(), 4), extra=round(extra.item(), 6), parts=parts, mechanism=more,
                 grads_missing=no_grad[:5], grads_all_zero=zero_grad[:5], grads_nonfinite=nonfinite[:5],
                 frozen_with_grad=len(frozen_with_grad), finite=bool(torch.isfinite(loss + lam * extra)))
        if arm == "baseline":
            ref_feats = feats.detach().float()
        if arm == "patch":
            r["gamma_grad"] = det.pool.gamma.grad.item()
            r["q_grad_norm_at_gamma0"] = det.pool.q.grad.norm().item()
            r["equals_baseline_at_init"] = bool(torch.allclose(feats.detach().float(), ref_feats, atol=1e-3))
        if arm == "vrex":
            r["per_sample_ce_by_env_is_variance_source"] = more
        R[f"B_{arm}"] = r
        print(f"[B] {arm}:", json.dumps(r, default=str), flush=True)
        del det, x, y, feats, ho
    free()

    # ---------------- [C] memory / GPU throughput (baseline, synthetic inputs, optimizer step included)
    det = Detector("baseline", seed=0).cuda().train()
    params = [p for _, p in det.trainable_named()]
    opt = torch.optim.AdamW(params, lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    crit = GenDLoss(det.gend.criterion)

    def bench(pairs, ckpt):
        if ckpt:
            enable_grad_ckpt(det)
        free()
        B = 2 * pairs
        x = torch.randn(B, 3, 224, 224, device="cuda")
        y = torch.tensor([0] * pairs + [1] * pairs, device="cuda")
        try:
            times = []
            for i in range(6):
                torch.cuda.synchronize()
                t = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    f = det.features(x)
                ho = det.head(f)
                loss, _ = crit(ho.logits_labels, ho.l2_embeddings, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.time() - t)
            s = sum(times) / len(times)
            return dict(batch=B, grad_ckpt=ckpt, s_per_step=round(s, 3), img_per_s=round(B / s, 1),
                        peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        except torch.cuda.OutOfMemoryError:
            return dict(batch=B, grad_ckpt=ckpt, oom=True)

    R["C_gpu"] = [bench(16, False), bench(24, False), bench(32, False), bench(32, True)]
    for r in R["C_gpu"]:
        print("[C] gpu:", r, flush=True)

    # ---------------- [D] data loader throughput (real augmentations + CLIP preprocessing, 2 workers)
    pre = D.make_preprocess(det.clip_processor)
    ds = D.CropDataset(trn, pre, augment=D.official_augmentations(), seed=0)
    smp = D.PairedStratifiedBatchSampler(trn, pairs_per_batch=P, seed=0)
    loader = DataLoader(ds, batch_sampler=list(smp)[:12], num_workers=2)
    t = time.time()
    n = 0
    for i, (xb, yb, _) in enumerate(loader):
        if i == 1:
            t, n = time.time(), 0
        n += xb.shape[0]
    R["D_loader_img_per_s"] = round(n / (time.time() - t), 1)
    print("[D] loader img/s:", R["D_loader_img_per_s"], flush=True)

    # ---------------- [E] validation pass time + untrained (= frozen CLIP + random head) val metrics
    free()
    t = time.time()
    tv, _ = predict(det, val, pre, workers=2)
    R["E_val"] = dict(seconds=round(time.time() - t, 1), frames=len(val), init_metrics=video_metrics(tv))
    print("[E] val:", R["E_val"], flush=True)

    # ---------------- [F] tiny overfit (baseline): 2 fixed batches, 30 steps, lr 3e-4
    free()
    det2 = Detector("baseline", seed=0).cuda().train()
    params = [p for _, p in det2.trainable_named()]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    crit = GenDLoss(det2.gend.criterion)
    pre2 = D.make_preprocess(det2.clip_processor)
    ds2 = D.CropDataset(trn, pre2, augment=None, seed=0)
    fixed = first_batch(ds2, D.PairedStratifiedBatchSampler(trn, pairs_per_batch=P, seed=0), n=2)
    ces = []
    for s in range(30):
        xb, yb = fixed[s % 2]
        xb, yb = xb.cuda(), yb.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            f = det2.features(xb)
        ho = det2.head(f)
        loss, parts = crit(ho.logits_labels, ho.l2_embeddings, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ces.append(parts["ce"])
    R["F_overfit_ce"] = dict(first=round(ces[0], 4), last=round(ces[-1], 4), min=round(min(ces), 4))
    print("[F] overfit CE:", R["F_overfit_ce"], flush=True)

    (out / "sanity.json").write_text(json.dumps(R, indent=2, default=str))
    print("saved", out / "sanity.json")


if __name__ == "__main__":
    main()
