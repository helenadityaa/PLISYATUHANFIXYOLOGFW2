"""Batch-aware wrapper around the established comparison workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_new_gen import compare_experiments as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLOv12 variants for one batch size.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=base.VARIANTS, choices=base.VARIANTS)
    parser.add_argument("--epochs", nargs="+", type=int, default=base.EPOCHS_LIST, choices=base.EPOCHS_LIST)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: runs/detect/batch_size_<batch>.",
    )
    return parser.parse_args()


def experiment_name(variant: str, epochs: int, batch: int) -> str:
    return (
        f"YOLOV12{variant.upper()}_192_E{epochs}_B{batch}_3class_"
        "VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop"
    )


def main() -> None:
    args = parse_args()
    if not base.DATASET_YAML.exists():
        raise SystemExit(f"{base.DATASET_YAML} tidak ditemukan.")

    runs_dir = (
        args.runs_dir.resolve()
        if args.runs_dir
        else base.PROJECT_ROOT / "runs" / "detect" / f"batch_size_{args.batch}"
    )
    summary_stem = (
        f"summary_comparison_yolov12_192_B{args.batch}_"
        "3class_vv_vh_rgb_scene_80_10_10_seed7"
    )
    summary_csv = base.PROJECT_ROOT / f"{summary_stem}.csv"
    summary_xlsx = base.PROJECT_ROOT / f"{summary_stem}.xlsx"

    # Reuse all established settings and functions; only select the batch/path.
    base.BATCH = args.batch
    rows = []
    for variant in args.variants:
        for epochs in args.epochs:
            run_name = experiment_name(variant, epochs, args.batch)
            run_dir = runs_dir / run_name
            print(f"\n>>> YOLOv12{variant} epoch={epochs}, batch={args.batch}")
            print(f"Run dir: {run_dir}")
            try:
                base.run_training(variant, epochs, run_dir)
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Training gagal untuk {run_name}: {exc}")

            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                rows.append(base.best_metrics_from_results(results_csv, variant, epochs, run_dir))
            else:
                rows.append(base.empty_summary_row(variant, epochs, run_dir))

    summary = pd.DataFrame(rows, columns=base.SUMMARY_COLUMNS)
    summary.to_csv(summary_csv, index=False)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary_All_Experiments", index=False)
        base.select_best_model(summary).to_excel(writer, sheet_name="Best_Model", index=False)

    print(f"CSV summary written to: {summary_csv}")
    print(f"Excel summary written to: {summary_xlsx}")


if __name__ == "__main__":
    main()
