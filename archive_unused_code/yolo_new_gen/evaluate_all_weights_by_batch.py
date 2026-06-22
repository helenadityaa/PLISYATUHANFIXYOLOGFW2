"""Discover and evaluate YOLO runs stored under batch-size subfolders."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_new_gen import evaluate_all_weights as base


RUN_PATTERN = re.compile(
    r"^YOLOV12([NSMXL])_192_E(\d+)_B(\d+)_3class_"
    r"VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nested YOLOv12 B8/B16 runs.")
    parser.add_argument("--runs-dir", type=Path, default=base.DEFAULT_RUNS_DIR)
    parser.add_argument("--data", type=Path, default=base.DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=192)
    parser.add_argument("--batch", type=int, default=None, help="Optional evaluation-batch override.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=base.PROJECT_ROOT / "runs" / "detect_test_all")
    parser.add_argument("--output-csv", type=Path, default=base.DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-excel", type=Path, default=base.DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--weight-types", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--dry-run", action="store_true", help="List discovered jobs without evaluating.")
    return parser.parse_args()


def discover_jobs(runs_dir: Path, weight_types: list[str]) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_dir.rglob("*") if path.is_dir()):
        match = RUN_PATTERN.match(run_dir.name)
        if not match:
            continue
        for weight_type in weight_types:
            jobs.append(
                {
                    "run_name": run_dir.name,
                    "variant": match.group(1).lower(),
                    "epochs": int(match.group(2)),
                    "batch": int(match.group(3)),
                    "weight_type": weight_type,
                    "model_path": run_dir / "weights" / f"{weight_type}.pt",
                    "eval_name": f"{run_dir.name}_{weight_type}_test",
                }
            )
    return jobs


def main() -> None:
    args = parse_args()
    jobs = discover_jobs(args.runs_dir, args.weight_types)
    if not jobs:
        raise SystemExit(f"No matching YOLOv12 runs found in: {args.runs_dir}")

    if args.dry_run:
        for job in jobs:
            evaluation_batch = args.batch if args.batch is not None else job["batch"]
            print(f"{job['run_name']} {job['weight_type']}.pt evaluation_batch={evaluation_batch}")
        print(f"Discovered jobs: {len(jobs)}")
        return

    rows = []
    for job in jobs:
        requested_override = args.batch
        args.batch = requested_override if requested_override is not None else int(job["batch"])
        rows.append(base.run_evaluate(job, args))
        args.batch = requested_override

    summary = pd.DataFrame(rows, columns=base.SUMMARY_COLUMNS)
    base.export_summary(summary, args.output_csv, args.output_excel)


if __name__ == "__main__":
    main()
