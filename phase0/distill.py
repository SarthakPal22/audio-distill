"""Knowledge distillation training loop.

The loss is a blend of two objectives:

  loss = alpha * CrossEntropy(student_logits, true_label)          # "hard" loss
       + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)      # "soft" loss

- CrossEntropy: the normal classification loss against the true label.
- KL divergence: "how different is the student's probability distribution
  from the teacher's?" Minimizing it pulls the student's *entire* output
  distribution toward the teacher's, not just the top answer.
- Temperature T: both logit sets are divided by T before softmax. T > 1
  flattens the distributions so small-but-informative probabilities
  ("this 'go' also sounds a bit like 'no'") aren't rounded to ~0.
  The T^2 factor keeps gradient magnitudes comparable when you change T
  (from Hinton et al. 2015, the original distillation paper).

Run `python -m phase0.distill --alpha 1.0` to train WITHOUT distillation
(pure hard labels, teacher never loaded) — that's your baseline to
compare against.

Quick sanity run (a few minutes, low accuracy, just proves the plumbing):
  python -m phase0.distill --epochs 1 --max-batches 30

Real run:
  python -m phase0.distill --epochs 8
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase0.data import KeywordDataset
from phase0.student import StudentCNN


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")     # NVIDIA GPU (Colab/Kaggle)
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon GPU
    return torch.device("cpu")


def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    hard = F.cross_entropy(student_logits, labels)
    if alpha >= 1.0:
        return hard  # baseline mode: no teacher signal
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * hard + (1.0 - alpha) * soft


@torch.no_grad()
def accuracy(model, loader, device, max_batches=None) -> float:
    model.eval()
    correct, total = 0, 0
    for i, (waveforms, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        preds = model(waveforms.to(device)).argmax(dim=-1)
        correct += (preds == labels.to(device)).sum().item()
        total += labels.numel()
    model.train()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="where the dataset is downloaded")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="weight of the hard-label loss; 1.0 = no distillation baseline")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="softening temperature for the soft loss")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit batches per epoch, for quick sanity runs")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--out", default="checkpoints/student_best.pt")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    train_ds = KeywordDataset(args.data_dir, "training")
    val_ds = KeywordDataset(args.data_dir, "validation")
    # num_workers=2 loads audio in background processes so the GPU isn't idle
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"train clips: {len(train_ds)}, validation clips: {len(val_ds)}")

    student = StudentCNN().to(device)
    print(f"student parameters: {student.num_parameters():,}")

    teacher = None
    if args.alpha < 1.0:
        from phase0.teacher import Teacher  # imported lazily: baseline runs skip the download
        teacher = Teacher(device)
        print(f"teacher parameters: {teacher.num_parameters():,} "
              f"({teacher.num_parameters() / student.num_parameters():.0f}x larger)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * (args.max_batches or len(train_loader))
    )

    best_val_acc = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}",
                        total=args.max_batches or len(train_loader))
        for i, (waveforms, labels) in enumerate(progress):
            if args.max_batches is not None and i >= args.max_batches:
                break
            # teacher reads waveforms from CPU (its feature extractor needs
            # numpy); the student gets them on the training device
            teacher_logits = teacher.logits(waveforms) if teacher else None
            waveforms, labels = waveforms.to(device), labels.to(device)

            student_logits = student(waveforms)
            loss = distillation_loss(student_logits, teacher_logits, labels,
                                     args.alpha, args.temperature)

            optimizer.zero_grad()
            loss.backward()      # backpropagation: compute gradients
            optimizer.step()     # update the student's weights
            scheduler.step()

            running_loss += loss.item() * labels.numel()
            seen += labels.numel()
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        val_acc = accuracy(student, val_loader, device,
                           max_batches=args.max_batches)
        print(f"epoch {epoch}: train loss {running_loss / seen:.4f}, "
              f"validation accuracy {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": student.state_dict(),
                        "val_acc": val_acc,
                        "args": vars(args)}, args.out)
            print(f"  new best -> saved to {args.out}")

    minutes = (time.time() - start) / 60
    print(f"\ndone in {minutes:.1f} min. best validation accuracy: {best_val_acc:.2%}")
    print(f"next: python -m phase0.evaluate --checkpoint {args.out} --with-teacher")


if __name__ == "__main__":
    main()
