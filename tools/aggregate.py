"""Aggregate runs/*/manifest.json into mean ± std tables (grouped over
seeds) and a flat CSV. Prints README-ready markdown.

Usage:  python -m tools.aggregate
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

GROUP_KEYS = ["arch", "width", "alpha", "temperature", "feature_beta",
              "augment", "fraction"]
METRICS = ["test_accuracy", "test_ece", "teacher_agreement", "best_val_accuracy"]


def load_rows(runs_dir: Path = Path("runs")) -> list[dict]:
    rows = []
    for manifest in sorted(runs_dir.glob("*/manifest.json")):
        m = json.loads(manifest.read_text())
        row = {k: m["args"][k] for k in GROUP_KEYS + ["seed", "epochs"]}
        row.update({k: m.get(k) for k in METRICS})
        row.update({"params": m["params"], "macs": m["macs"],
                    "mean_epoch_seconds": m.get("mean_epoch_seconds"),
                    "run": manifest.parent.name})
        rows.append(row)
    return rows


def fmt(mean: float, std: float | None) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}" if std is not None else f"{mean * 100:.2f}"


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no manifests found under runs/")
        return

    Path("docs").mkdir(exist_ok=True)
    with open("docs/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)

    print(f"{len(rows)} runs, {len(groups)} configurations\n")
    header = ("| arch | width | alpha | T | beta | aug | frac | seeds "
              "| test acc (%) | ECE | agreement (%) |")
    print(header)
    print("|" + "---|" * 11)
    for key in sorted(groups):
        runs = groups[key]
        accs = [r["test_accuracy"] for r in runs if r["test_accuracy"] is not None]
        eces = [r["test_ece"] for r in runs if r["test_ece"] is not None]
        agrs = [r["teacher_agreement"] for r in runs
                if r.get("teacher_agreement") is not None]
        acc = fmt(statistics.mean(accs),
                  statistics.stdev(accs) if len(accs) > 1 else None)
        ece = (f"{statistics.mean(eces):.4f}" if eces else "—")
        agr = (fmt(statistics.mean(agrs),
                   statistics.stdev(agrs) if len(agrs) > 1 else None)
               if agrs else "—")
        cells = [str(v) for v in key] + [str(len(runs)), acc, ece, agr]
        print("| " + " | ".join(cells) + " |")
    print("\nwrote docs/results.csv")


if __name__ == "__main__":
    main()
