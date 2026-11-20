"""Train one arm of the v2 comparison on a single T4.

  python -m gend_v2.train --arm baseline --seed 0 --epochs 2 --out /content/drive/MyDrive/GenD_PCR_Project/v2/runs/baseline_s0

Shared by all arms: official GenD model/loss/augmentations/preprocessing, identical paired batches,
AdamW(lr 3e-4, wd 0, betas .9/.999) = the paper's optimiser and peak lr; schedule differs from the paper: one cosine
decay to 1e-6 over the run, no warm-up (paper: cyclic cosine with 1-epoch warm-up). fp16 autocast.
Checkpoint selection: best video-level AP on the validation set (FaceShifter val split), mirroring the
official monitor metric 'val/mAP_video'. The test sets are never touched here.
Outputs (in --out): config.json, log.csv, val.csv, best.pt, last.pt (resumable), snapshots/step_*.pt.
"""
import argparse
import hashlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as D
from .evalutil import predict, video_metrics
from .losses import PDC, GenDLoss, vrex_penalty
from .model import Detector, count_trainable

DEFAULT_LAM = {"baseline": 0.0, "pdc": 0.5, "vrex": 10.0, "patch": 0.0, "pdcvrex": 0.5}


def enable_grad_ckpt(det):
    import torch.utils.checkpoint as cp
    fn = partial(cp.checkpoint, use_reentrant=False)
    enc = det.gend.feature_extractor.vision_model.encoder
    enc.gradient_checkpointing = True
    enc._gradient_checkpointing_func = fn
    for layer in enc.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = fn


def get_args(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--arm", required=True, choices=["baseline", "pdc", "vrex", "patch", "pdcvrex"])
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--epochs", type=int, default=2)
    a.add_argument("--max_steps", type=int, default=0, help="override total steps (sanity runs)")
    a.add_argument("--pairs", type=int, default=16, help="pairs per batch; batch size = 2 * pairs")
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--min_lr", type=float, default=1e-6)
    a.add_argument("--lam", type=float, default=None)
    a.add_argument("--vrex_warmup", type=float, default=0.2, help="fraction of steps with penalty off")
    a.add_argument("--lam_vrex", type=float, default=10.0, help="V-REx weight for arm pdcvrex (lam = PDC weight)")
    a.add_argument("--select_after", type=int, default=0,
                   help="only checkpoints after this step can become best.pt (pre-registered for pdcvrex)")
    a.add_argument("--data", choices=["v2", "full"], default="v2",
                   help="v2 = 120 train pairs (main study); full = all 360 official train pairs (scale-up check)")
    a.add_argument("--eval_every", type=int, default=160)
    a.add_argument("--snap_every", type=int, default=80)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--grad_ckpt", action="store_true")
    a.add_argument("--no_val", action="store_true")
    a.add_argument("--out", required=True)
    a.add_argument("--resume", action="store_true")
    args = a.parse_args(argv)
    if args.lam is None:
        args.lam = DEFAULT_LAM[args.arm]
    return args


def batch_fingerprint(sampler, n=5):
    h = hashlib.sha1()
    for b, batch in enumerate(sampler):
        if b >= n:
            break
        h.update(str(batch).encode())
    return h.hexdigest()[:12]


def main(argv=None):
    args = get_args(argv)
    out = Path(args.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---------------- data
    full = args.data == "full"
    idx = D.load_index(name="crops_index_full.csv" if full else "crops_index_v2.csv")
    groups = [f"train__{m}" for m in ["real"] + D.FF_METHODS] + ["val__real", "val__FaceShifter"]
    if full:
        groups += [f"train2__{m}" for m in ["real"] + D.FF_METHODS]
    D.ensure_crops(groups, idx=idx)
    trn = D.frame_table(idx, ["train", "train2"] if full else "train")
    val = D.frame_table(idx, "val")

    # ---------------- model (seeded by the official GenD constructor)
    det = Detector(args.arm, seed=args.seed).cuda()
    det.train()
    if args.grad_ckpt:
        enable_grad_ckpt(det)
    preprocess = D.make_preprocess(det.clip_processor)
    n_tensors, n_params = count_trainable(det)

    sampler = D.PairedStratifiedBatchSampler(trn, pairs_per_batch=args.pairs, seed=args.seed)
    ds = D.CropDataset(trn, preprocess, augment=D.official_augmentations(), seed=args.seed)
    steps_per_epoch = len(sampler)
    total = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch

    params = [p for _, p in det.trainable_named()]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda")
    base_loss = GenDLoss(det.gend.criterion)
    pdc = PDC() if args.arm in ("pdc", "pdcvrex") else None
    n_env = len(D.FF_METHODS)

    step, best = 0, {"ap": -1.0}
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        det.load_trainable_state(ck["trainable"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        if pdc is not None and ck.get("pdc_u") is not None:
            pdc.u = ck["pdc_u"].cuda()
        print(f"[resume] from step {step}")

    cfg = dict(vars(args), total_steps=total, steps_per_epoch=steps_per_epoch,
               trainable_tensors=n_tensors, trainable_params=n_params,
               exact_pair_fraction=sampler.exact, batch_fingerprint=batch_fingerprint(sampler),
               n_train_crops=len(trn), n_val_crops=len(val), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(json.dumps({k: cfg[k] for k in ["arm", "seed", "lam", "total_steps", "steps_per_epoch",
                                          "trainable_params", "batch_fingerprint", "exact_pair_fraction"]},
                     default=str))

    def save_last():
        torch.save(dict(trainable=det.trainable_state(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), step=step, best=best,
                        pdc_u=None if pdc is None else pdc.state()), str(out / "last.pt") + ".tmp")
        os.replace(str(out / "last.pt") + ".tmp", out / "last.pt")

    def evaluate():
        t, _ = predict(det, val, preprocess, workers=args.workers)
        m = video_metrics(t)
        m.update(step=step)
        pd.DataFrame([m]).to_csv(out / "val.csv", mode="a", header=not (out / "val.csv").exists(), index=False)
        return m

    log_rows, t_last, imgs_since = [], time.time(), 0
    while step < total:
        epoch = step // steps_per_epoch
        sampler.set_epoch(epoch)
        ds.set_epoch(epoch)
        batches = list(sampler)[step - epoch * steps_per_epoch:]
        loader = DataLoader(ds, batch_sampler=batches, num_workers=args.workers, pin_memory=True,
                            persistent_workers=False)
        for x, y, _ in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                feats = det.features(x)
            ho = det.head(feats)
            logits, z = ho.logits_labels, ho.l2_embeddings
            loss, parts = base_loss(logits, z, y)
            extra = torch.zeros((), device=x.device)
            if args.arm == "pdc":
                extra, more = pdc(z, args.pairs)
                parts.update(more)
            elif args.arm == "vrex":
                extra, more = vrex_penalty(logits, y, args.pairs, n_env)
                parts.update(more)
            elif args.arm == "pdcvrex":
                e1, m1 = pdc(z, args.pairs)
                e2, m2 = vrex_penalty(logits, y, args.pairs, n_env)
                w2 = 0.0 if step < args.vrex_warmup * total else args.lam_vrex
                extra = args.lam * e1 + w2 * e2          # already weighted -> lam_t = 1
                parts.update(m1)
                parts.update(m2)
                parts.update(pdc_term=e1.item(), vrex_term=e2.item(), vrex_w=w2)
            lam_t = args.lam
            if args.arm == "vrex" and step < args.vrex_warmup * total:
                lam_t = 0.0
            if args.arm == "pdcvrex":
                lam_t = 1.0
            total_loss = loss + lam_t * extra
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {parts}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))  # measure only, no clipping
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            imgs_since += x.shape[0]

            if step % 20 == 0 or step == total:
                dt = time.time() - t_last
                row = dict(step=step, epoch=epoch, lr=sched.get_last_lr()[0], loss=total_loss.item(),
                           extra=extra.item(), lam_t=lam_t, gnorm=gnorm.item(), imgs_per_s=imgs_since / dt,
                           mem_gb=torch.cuda.max_memory_allocated() / 1e9, **parts)
                if args.arm == "patch":
                    row["gamma"] = det.pool.gamma.item()
                log_rows.append(row)
                pd.DataFrame([row]).to_csv(out / "log.csv", mode="a", header=not (out / "log.csv").exists(),
                                           index=False)
                t_last, imgs_since = time.time(), 0
                if step % 40 == 0 or step == total:
                    print(f"step {step}/{total} ep {epoch} loss {row['loss']:.4f} ce {parts['ce']:.4f} "
                          f"extra {row['extra']:.4f} {row['imgs_per_s']:.1f} img/s mem {row['mem_gb']:.1f}GB",
                          flush=True)

            if step % args.snap_every == 0 or step == total:
                torch.save(det.trainable_state(), out / "snapshots" / f"step_{step:05d}.pt")

            if not args.no_val and (step % args.eval_every == 0 or step == total):
                m = evaluate()
                print(f"  [val step {step}] video AUC {m['auc']:.4f} AP {m['ap']:.4f} acc {m['acc']:.3f}", flush=True)
                if m["ap"] > best["ap"] and step > args.select_after:
                    best = m
                    torch.save(dict(trainable=det.trainable_state(), step=step, val=m,
                                    pdc_u=None if pdc is None else pdc.state()), out / "best.pt")
                save_last()
            if step >= total:
                break

    save_last()
    (out / "done.json").write_text(json.dumps(dict(best=best, steps=step), indent=2))
    print("DONE. best:", best)
    return best


if __name__ == "__main__":
    main()
