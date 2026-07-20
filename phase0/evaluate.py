"""Compare the distilled student against the teacher.

Reports the three numbers that define a distillation result:
  1. accuracy on the held-out test set (quality)
  2. parameter count (size)
  3. single-clip CPU latency (speed)

Usage:
  python -m phase0.evaluate --checkpoint checkpoints/student_best.pt --with-teacher
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase0.data import KeywordDataset, CLIP_SAMPLES
from phase0.distill import pick_device
from phase0.student import StudentCNN


@torch.no_grad()
def test_accuracy(predict_fn, loader, device, desc) -> float:
    correct, total = 0, 0
    for waveforms, labels in tqdm(loader, desc=desc):
        preds = predict_fn(waveforms).argmax(dim=-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


@torch.no_grad()
def cpu_latency_ms(predict_fn, runs: int = 50) -> float:
    """Median time to classify a single 1-second clip on CPU — the number
    that matters for on-device deployment."""
    clip = torch.randn(1, CLIP_SAMPLES)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_fn(clip)
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/student_best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--with-teacher", action="store_true",
                        help="also evaluate the teacher (slower, downloads it)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    device = pick_device(args.device)
    test_loader = DataLoader(KeywordDataset(args.data_dir, "testing"),
                             batch_size=args.batch_size, num_workers=2)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    student = StudentCNN()
    student.load_state_dict(ckpt["model_state"])
    student.to(device).eval()

    student_cpu = StudentCNN()
    student_cpu.load_state_dict(ckpt["model_state"])
    student_cpu.eval()

    student_acc = test_accuracy(lambda w: student(w.to(device)),
                                test_loader, device, "student on test set")
    student_ms = cpu_latency_ms(student_cpu)

    print("\n=== student ===")
    print(f"parameters:     {student.num_parameters():,}")
    print(f"test accuracy:  {student_acc:.2%}")
    print(f"CPU latency:    {student_ms:.1f} ms per clip")
    print(f"(trained with:  {ckpt.get('args', {})})")

    if args.with_teacher:
        from phase0.teacher import Teacher
        teacher = Teacher(device)
        teacher_acc = test_accuracy(teacher.logits, test_loader, device,
                                    "teacher on test set")
        teacher_cpu = Teacher(torch.device("cpu"))
        teacher_ms = cpu_latency_ms(teacher_cpu.logits, runs=10)

        print("\n=== teacher ===")
        print(f"parameters:     {teacher.num_parameters():,}")
        print(f"test accuracy:  {teacher_acc:.2%}")
        print(f"CPU latency:    {teacher_ms:.1f} ms per clip")

        print("\n=== headline ===")
        print(f"student is {teacher.num_parameters() / student.num_parameters():.0f}x "
              f"smaller, {teacher_ms / student_ms:.1f}x faster on CPU, and reaches "
              f"{student_acc / teacher_acc:.1%} of the teacher's accuracy.")


if __name__ == "__main__":
    main()
