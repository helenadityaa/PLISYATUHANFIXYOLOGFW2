"""Run prediction-characteristic analysis with class+IoU matching."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_new_gen import analyze_yolo_prediction_characteristics as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO characteristics with strict IoU matching.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument(
        "--dataset-dir",
        default="yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7",
    )
    parser.add_argument("--image-split", choices=["train", "val", "test", "auto"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.match_iou_threshold <= 1.0:
        raise ValueError("--match-iou-threshold must be between 0 and 1.")

    original_match = base.match_ground_truth

    def strict_match_ground_truth(
        dataset_dir: Path,
        image_split: str,
        image_name: str,
        image_width: int,
        image_height: int,
        pred_box: dict[str, float],
    ) -> dict[str, object]:
        result = original_match(
            dataset_dir,
            image_split,
            image_name,
            image_width,
            image_height,
            pred_box,
        )
        class_id_match = bool(result.get("is_class_match", False))
        best_iou = result.get("best_gt_iou", math.nan)
        iou_match = pd.notna(best_iou) and float(best_iou) >= args.match_iou_threshold
        result["is_class_id_match"] = class_id_match
        result["is_class_match"] = class_id_match and iou_match
        return result

    base.match_ground_truth = strict_match_ground_truth
    base.FEATURE_DESCRIPTIONS["is_class_id_match"] = (
        "Whether predicted class id matches the best-overlap ground-truth class, regardless of IoU."
    )
    base.FEATURE_DESCRIPTIONS["is_class_match"] = (
        f"Whether class id matches and best ground-truth IoU is at least {args.match_iou_threshold:.2f}."
    )

    output_dir = Path(args.output_dir)
    features = base.analyze_predictions(
        predictions_csv=Path(args.predictions_csv),
        dataset_dir=Path(args.dataset_dir),
        image_split=args.image_split,
        confidence_threshold=args.confidence_threshold,
    )
    base.write_outputs(features, output_dir)
    base.make_boxplots(features, output_dir)
    base.make_representative_montage(features, output_dir, args.samples_per_class)
    base.save_samples(features, output_dir, args.samples_per_class)
    print(f"Strict class+IoU analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
