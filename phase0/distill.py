"""Knowledge distillation training loop.

The objective is a three-term compression loss:

  L = alpha * CE(z_s, y)                                  # hard-label term
    + (1 - alpha) * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))   # logit matching
    + beta * (1 - cos(P(h_s), h_t))                       # feature alignment

- **Hard-label term**: ordinary cross-entropy against the ground truth.
- **Logit matching via temperature-scaled KL divergence** (Hinton et al.
  2015): both logit sets are divided by temperature T before softmax.
  T > 1 flattens the posteriors, exposing the teacher's "dark knowledge"
  — the small probabilities on wrong-but-similar classes ("this 'go'
  also sounds like 'no'") that carry the inter-class similarity
  structure. The T^2 factor keeps soft-loss gradient magnitudes constant
  as T changes. alpha=1.0 disables distillation entirely (the baseline);
  alpha=0.0 trains purely on the teacher's posterior.
- **Feature alignment (FitNets-style hint training)**: a learned linear
  **projection head** P maps the student's penultimate embedding h_s
  (dim 128) into the teacher's representation space (dim 768), and a
  cosine-distance loss pulls it toward the teacher's pooled hidden state
  h_t. The projection head exists only during training. beta=0 (default)
  disables the term.

Teacher signals come in two regimes (see cache_teacher.py):

- **offline distillation** (default when packs exist): z_t and h_t are
  read from the fp16 on-disk cache; the teacher never runs during
  training. With input augmentation this makes the soft labels *stale*
  (computed on clean audio).
- **online distillation** (--augment online): the teacher recomputes
  z_t on each augmented batch — exact but pays an ~840x-larger forward
  pass every step.

Runs write to runs/<name>/: manifest.json (git SHA, args, final metrics),
epochs.csv (per-epoch curve), student_best.pt (best-val-accuracy weights).

Examples:
  python -m phase0.distill --alpha 1.0                    # no-teacher baseline
  python -m phase0.distill --alpha 0.3 --temperature 4.0  # canonical KD
  python -m phase0.distill --alpha 0.3 --feature-beta 1.0 # logits + features
  python -m phase0.distill --alpha 0.3 --augment offline  # aug, cached logits
  python -m phase0.distill --epochs 1 --max-batches 30    # plumbing check
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase0 import metrics
from phase0.data import (NUM_CLASSES, PackedSpeechCommands, WaveformAugment,
                         noise_dir_for, stratified_subset)
from phase0.student import build_student
from phase0.utils import (count_macs, pick_device, seed_worker, set_seed,
                          write_manifest)

# re-exported for backwards compatibility (tests import it from here)
pick_device = pick_device


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor | None,
                      labels: torch.Tensor,
                      alpha: float, temperature: float) -> torch.Tensor:
    hard = F.cross_entropy(student_logits, labels)
    if alpha >= 1.0 or teacher_logits is None:
        return hard  # baseline mode: no teacher signal
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * hard + (1.0 - alpha) * soft


def feature_loss(student_emb: torch.Tensor, projector: torch.nn.Linear,
                 teacher_emb: torch.Tensor) -> torch.Tensor:
    """Representation alignment: cosine distance after projection."""
    return (1.0 - F.cosine_similarity(projector(student_emb),
                                      teacher_emb, dim=-1)).mean()


@torch.no_grad()
def val_accuracy(model, loader, device, max_batches=None) -> float:
    model.eval()
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        waveforms, labels = batch[0].to(device), batch[1].to(device)
        correct += (model(waveforms).argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def test_metrics(model, loader, device, max_batches=None) -> dict[str, float]:
    """Final held-out metrics: accuracy, ECE, and (when cached teacher
    logits exist) teacher agreement and mean KL."""
    model.eval()
    all_logits, all_labels, all_teacher = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        waveforms, labels = batch[0], batch[1]
        all_logits.append(model(waveforms.to(device)).cpu())
        all_labels.append(labels)
        if len(batch) > 2 and batch[2].numel():
            all_teacher.append(batch[2])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = logits.softmax(-1)
    out = {
        "test_accuracy": (logits.argmax(-1) == labels).float().mean().item(),
        "test_ece": metrics.ece(probs, labels),
    }
    if all_teacher:
        teacher = torch.cat(all_teacher)
        out["teacher_agreement"] = metrics.agreement(logits, teacher)
        out["mean_kl_to_teacher"] = metrics.mean_kl(logits, teacher)
    return out


def run_name(args) -> str:
    parts = [f"{args.arch}{args.width:g}", f"a{args.alpha:g}", f"T{args.temperature:g}"]
    if args.feature_beta > 0:
        parts.append(f"b{args.feature_beta:g}")
    if args.augment != "none":
        parts.append(f"aug-{args.augment}")
    if args.fraction < 1.0:
        parts.append(f"f{args.fraction:g}")
    parts.append(f"s{args.seed}")
    return "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs-dir", default="data/packs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--arch", default="cnn", choices=["cnn", "dscnn"])
    parser.add_argument("--width", type=float, default=1.0,
                        help="channel width multiplier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="hard-label loss weight; 1.0 = no-distillation baseline")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--feature-beta", type=float, default=0.0,
                        help="weight of the FitNets-style feature-alignment loss")
    parser.add_argument("--augment", default="none",
                        choices=["none", "offline", "online"],
                        help="offline: augment inputs, reuse clean cached logits; "
                             "online: teacher recomputes logits on augmented audio")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="stratified fraction of the training set (low-resource ablation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default=None,
                        help="output directory; defaults to runs/<auto-name>")
    args = parser.parse_args()

    device = pick_device(args.device)
    generator = set_seed(args.seed)
    run_dir = Path(args.run_dir or f"runs/{run_name(args)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir}  device: {device}")

    augment = None
    if args.augment != "none":
        augment = WaveformAugment(noise_dir_for(args.data_dir))
    train_ds = PackedSpeechCommands(args.packs_dir, "training", augment=augment)
    if args.fraction < 1.0:
        train_ds.indices = stratified_subset(train_ds.labels, args.fraction, args.seed)
    val_ds = PackedSpeechCommands(args.packs_dir, "validation")
    test_ds = PackedSpeechCommands(args.packs_dir, "testing")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True, generator=generator,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)
    print(f"train clips: {len(train_ds)} (fraction {args.fraction}), "
          f"val: {len(val_ds)}, test: {len(test_ds)}")

    student = build_student(args.arch, args.width,
                            spec_augment=args.augment != "none").to(device)
    macs = count_macs(student, torch.randn(1, 16000))
    print(f"student: {args.arch} width={args.width} "
          f"({student.num_parameters():,} params, {macs / 1e6:.1f}M MACs)")

    teacher = None
    if args.augment == "online" and args.alpha < 1.0:
        from phase0.teacher import Teacher  # lazy: heavy import + download
        teacher = Teacher(device)

    trainable = list(student.parameters())
    projector = None
    if args.feature_beta > 0:
        projector = torch.nn.Linear(student.embedding_dim, 768).to(device)
        trainable += list(projector.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    steps_per_epoch = args.max_batches or len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch)

    best_val, epoch_rows = 0.0, []
    ckpt_path = run_dir / "student_best.pt"
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_start = time.time()
        running, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}",
                        total=steps_per_epoch, leave=False)
        for i, (waveforms, labels, t_logits, t_emb) in enumerate(progress):
            if args.max_batches is not None and i >= args.max_batches:
                break
            if teacher is not None:  # online: recompute on augmented audio
                t_logits = teacher.logits(waveforms)
            waveforms, labels = waveforms.to(device), labels.to(device)
            t_logits = t_logits.to(device) if t_logits.numel() else None

            s_logits, s_emb = student(waveforms, return_embedding=True)
            loss = distillation_loss(s_logits, t_logits, labels,
                                     args.alpha, args.temperature)
            if projector is not None and t_emb.numel():
                loss = loss + args.feature_beta * feature_loss(
                    s_emb, projector, t_emb.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * labels.numel()
            seen += labels.numel()
            progress.set_postfix(loss=f"{running / seen:.4f}")

        val_acc = val_accuracy(student, val_loader, device, args.max_batches)
        secs = time.time() - epoch_start
        epoch_rows.append({"epoch": epoch, "train_loss": round(running / seen, 5),
                           "val_accuracy": round(val_acc, 5),
                           "seconds": round(secs, 1)})
        print(f"epoch {epoch}: loss {running / seen:.4f}, "
              f"val acc {val_acc:.2%}, {secs:.0f}s")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": student.state_dict(),
                        "arch": args.arch, "width": args.width,
                        "val_acc": val_acc, "args": vars(args)}, ckpt_path)

    with open(run_dir / "epochs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    # evaluate the best checkpoint (not the last epoch) on the test split
    student.load_state_dict(torch.load(ckpt_path, map_location=device,
                                       weights_only=True)["model_state"])
    final = test_metrics(student, test_loader, device, args.max_batches)
    write_manifest(run_dir, {
        "args": vars(args),
        "params": student.num_parameters(),
        "macs": macs,
        "best_val_accuracy": round(best_val, 5),
        "mean_epoch_seconds": round(
            sum(r["seconds"] for r in epoch_rows) / len(epoch_rows), 1),
        "total_seconds": round(time.time() - train_start, 1),
        **{k: round(v, 5) for k, v in final.items()},
    })
    print(f"done. best val {best_val:.2%}, test {final['test_accuracy']:.2%} "
          f"-> {run_dir}/manifest.json")


if __name__ == "__main__":
    main()
