"""Evaluate best.pt and last.pt for all YOLOv12 experiment runs on the test split."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
DEFAULT_DATA = SCRIPT_DIR / "dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "detect_test_all"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "summary_test_all_best_last_pt_80_10_10_seed7.csv"
DEFAULT_OUTPUT_XLSX = PROJECT_ROOT / "summary_test_all_best_last_pt_80_10_10_seed7.xlsx"
RUN_PATTERN = re.compile(r"^YOLOV12([NSMXL])_192_E(\d+)_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop$", re.IGNORECASE)
SUMMARY_COLUMNS = [
    "run_name",
    "variant",
    "epochs",
    "weight_type",
    "model_path",
    "data_path",
    "split",
    "imgsz",
    "batch",
    "device",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "status",
    "error",
    "metrics_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all YOLOv12 best.pt and last.pt checkpoints.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=192)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-excel", type=Path, default=DEFAULT_OUTPUT_XLSX)
    return parser.parse_args()


def parse_run_name(run_name: str) -> Optional[Dict[str, object]]:
    match = RUN_PATTERN.match(run_name)
    if not match:
        return None
    return {
        "variant": match.group(1).lower(),
        "epochs": int(match.group(2)),
    }


def iter_weight_jobs(runs_dir: Path) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        for weight_type in ("best", "last"):
            model_path = run_dir / "weights" / f"{weight_type}.pt"
            jobs.append(
                {
                    "run_name": run_dir.name,
                    "variant": parsed["variant"],
                    "epochs": parsed["epochs"],
                    "weight_type": weight_type,
                    "model_path": model_path,
                    "eval_name": f"{run_dir.name}_{weight_type}_test",
                }
            )
    return jobs


def read_metrics_csv(metrics_csv: Path, base_row: Dict[str, object]) -> Dict[str, object]:
    if not metrics_csv.exists():
        row = dict(base_row)
        row.update({"status": "missing_metrics_csv", "error": f"Not found: {metrics_csv}", "metrics_csv": str(metrics_csv)})
        return row
    with metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        metrics_row = next(reader, {})
    row = dict(base_row)
    row.update(metrics_row)
    row.update({"status": "ok", "error": "", "metrics_csv": str(metrics_csv)})
    return row


def run_evaluate(job: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    model_path = Path(job["model_path"])
    eval_name = str(job["eval_name"])
    metrics_csv = args.project / eval_name / "test_metrics.csv"
    base_row = {
        "run_name": job["run_name"],
        "variant": job["variant"],
        "epochs": job["epochs"],
        "weight_type": job["weight_type"],
        "model_path": str(model_path),
        "data_path": str(args.data),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "precision": None,
        "recall": None,
        "mAP50": None,
        "mAP50-95": None,
    }

    if not model_path.exists():
        row = dict(base_row)
        row.update({"status": "missing_model", "error": f"Not found: {model_path}", "metrics_csv": str(metrics_csv)})
        return row

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_test.py"),
        "--model",
        str(model_path),
        "--data",
        str(args.data),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--project",
        str(args.project),
        "--name",
        eval_name,
    ]
    print(f"\n>>> Evaluating {job['run_name']} {job['weight_type']}.pt")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        row = dict(base_row)
        row.update({"status": "failed", "error": str(exc), "metrics_csv": str(metrics_csv)})
        return row

    return read_metrics_csv(metrics_csv, base_row)


def export_summary(summary_df: pd.DataFrame, output_csv: Path, output_excel: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    best_rows = summary_df[summary_df["status"].eq("ok")].copy()
    if not best_rows.empty:
        best_rows["mAP50-95"] = pd.to_numeric(best_rows["mAP50-95"], errors="coerce")
        best_rows["mAP50"] = pd.to_numeric(best_rows["mAP50"], errors="coerce")
        best_rows = best_rows.sort_values(
            by=["mAP50-95", "mAP50"],
            ascending=[False, False],
            kind="mergesort",
        )

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="All_Test_Weights", index=False)
        best_rows.head(10).to_excel(writer, sheet_name="Top_10", index=False)

    print(f"\nSaved test summary CSV to {output_csv}")
    print(f"Saved test summary workbook to {output_excel}")


def main() -> None:
    args = parse_args()
    jobs = iter_weight_jobs(args.runs_dir)
    if not jobs:
        raise SystemExit(f"No YOLOv12 192 experiment runs found in: {args.runs_dir}")

    rows = [run_evaluate(job, args) for job in jobs]
    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    export_summary(summary_df, args.output_csv, args.output_excel)

    ok_rows = summary_df[summary_df["status"].eq("ok")].copy()
    if not ok_rows.empty:
        ok_rows["mAP50-95"] = pd.to_numeric(ok_rows["mAP50-95"], errors="coerce")
        ok_rows["mAP50"] = pd.to_numeric(ok_rows["mAP50"], errors="coerce")
        best = ok_rows.sort_values(["mAP50-95", "mAP50"], ascending=[False, False]).iloc[0]
        print(
            "Best test checkpoint: "
            f"{best['run_name']} {best['weight_type']}.pt "
            f"mAP50={best['mAP50']:.5f}, mAP50-95={best['mAP50-95']:.5f}"
        )


if __name__ == "__main__":
    main()
