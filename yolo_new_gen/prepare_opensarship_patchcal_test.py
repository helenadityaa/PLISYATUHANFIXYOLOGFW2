"""Prepare OpenSARShip PATCH_CAL TIFFs as a YOLO external test set.

This script is intended for cross-dataset evaluation. It does not split data
into train/val/test; every valid target-class sample is written to test only.

Expected output:

    <output-dir>/
      data.yaml
      external_test_manifest.csv
      skipped_rows.csv
      test/
        images/
        labels/
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import CLASS_NAMES, map_ship_class
from yolo_exp.bbox_utils import axis_aligned_bbox_local, bbox_to_yolo, dual_pol_to_rgb_uint8
from yolo_new_gen.prepare_data import format_yolo_bbox, resize_rgb_image


REQUIRED_BBOX_COLUMNS = [
    "category",
    "Center_x",
    "Center_y",
    "Head_x",
    "Head_y",
    "Tail_x",
    "Tail_y",
    "UpperLeft_x",
    "UpperLeft_y",
    "LowerRight_x",
    "LowerRight_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenSARShip two-channel PATCH_CAL TIFFs into a YOLO test-only dataset."
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Path to OpenSARShip metadata.csv.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder containing PATCH_CAL TIFFs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output YOLO external-test folder.")
    parser.add_argument(
        "--patch-column",
        default="patch_cal",
        help="Metadata column containing PATCH_CAL filename. Default: patch_cal.",
    )
    parser.add_argument("--source-name", default="opensarship", help="Dataset name written to the manifest.")
    parser.add_argument("--imgsz", type=int, default=192, help="Output image size. Default: 192.")
    parser.add_argument("--vv-channel", type=int, default=0, help="VV channel index in the TIFF. Default: 0.")
    parser.add_argument("--vh-channel", type=int, default=1, help="VH channel index in the TIFF. Default: 1.")
    parser.add_argument(
        "--channel-axis",
        choices=["auto", "first", "last"],
        default="auto",
        help="TIFF channel axis layout. Use auto for HWC/CHW detection.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    return parser.parse_args()


def is_blank(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "<na>"}


def read_polarizations(
    path: Path,
    vv_channel: int,
    vh_channel: int,
    channel_axis: str,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], str]:
    """Read VV/VH arrays from a channel-first or channel-last TIFF."""
    patch = np.asarray(tifffile.imread(path))
    original_shape = tuple(int(value) for value in patch.shape)
    if patch.ndim != 3:
        raise ValueError(f"expected a 3D TIFF, got shape={original_shape}")

    required_channels = max(vv_channel, vh_channel) + 1
    if channel_axis == "last":
        if patch.shape[-1] < required_channels:
            raise ValueError(f"last axis has too few channels: shape={original_shape}")
        channel_last = patch
        detected_axis = "last"
    elif channel_axis == "first":
        if patch.shape[0] < required_channels:
            raise ValueError(f"first axis has too few channels: shape={original_shape}")
        channel_last = np.moveaxis(patch, 0, -1)
        detected_axis = "first"
    else:
        last_possible = required_channels <= patch.shape[-1] <= 4
        first_possible = required_channels <= patch.shape[0] <= 4
        if last_possible and not first_possible:
            channel_last = patch
            detected_axis = "last"
        elif first_possible and not last_possible:
            channel_last = np.moveaxis(patch, 0, -1)
            detected_axis = "first"
        elif last_possible and first_possible:
            # Rare ambiguous case. OpenSARShip PATCH_CAL is normally H,W,2,
            # so prefer channel-last when both layouts look plausible.
            channel_last = patch
            detected_axis = "last_ambiguous_preferred"
        else:
            raise ValueError(f"no plausible channel axis in TIFF shape={original_shape}")

    vv = np.asarray(channel_last[..., vv_channel])
    vh = np.asarray(channel_last[..., vh_channel])
    if vv.shape != vh.shape or vv.ndim != 2:
        raise ValueError(f"invalid VV/VH shapes: {vv.shape}, {vh.shape}")
    if not np.isfinite(vv).all() or not np.isfinite(vh).all():
        raise ValueError("TIFF contains NaN or infinite values")
    return vv, vh, original_shape, detected_axis


def write_data_yaml(output_dir: Path) -> None:
    root = output_dir.resolve().as_posix()
    lines = [
        f'path: "{root}"',
        "train: test/images",
        "val: test/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES))
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_columns(metadata: pd.DataFrame, patch_column: str) -> None:
    required = [*REQUIRED_BBOX_COLUMNS, patch_column]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise KeyError("Missing metadata columns: " + ", ".join(missing))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Tuple[Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to regenerate.")
        shutil.rmtree(output_dir)

    images_dir = output_dir / "test" / "images"
    labels_dir = output_dir / "test" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Optional[Iterable[str]] = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = rows[0].keys() if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.vv_channel < 0 or args.vh_channel < 0:
        raise ValueError("--vv-channel and --vh-channel must be non-negative")
    if args.vv_channel == args.vh_channel:
        raise ValueError("--vv-channel and --vh-channel must be different")

    metadata_path = args.metadata.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"PATCH_CAL folder not found: {image_dir}")

    metadata = pd.read_csv(metadata_path)
    ensure_columns(metadata, args.patch_column)
    images_dir, labels_dir = prepare_output_dir(output_dir, args.overwrite)

    manifest = []
    skipped_rows = []
    skipped = Counter()
    class_counts = Counter()
    axis_counts = Counter()
    seen_patches = set()

    for row_number, row in metadata.iterrows():
        csv_row_number = int(row_number) + 2
        label = map_ship_class(row)
        if label is None:
            skipped["non_target_class"] += 1
            skipped_rows.append({"row": csv_row_number, "reason": "non_target_class"})
            continue

        patch_name = str(row.get(args.patch_column, "")).strip()
        if is_blank(patch_name):
            skipped["blank_patch_name"] += 1
            skipped_rows.append({"row": csv_row_number, "reason": "blank_patch_name"})
            continue

        patch_path = image_dir / Path(patch_name).name
        if not patch_path.is_file():
            skipped["missing_patch"] += 1
            skipped_rows.append({"row": csv_row_number, "reason": "missing_patch", "patch": patch_name})
            continue

        patch_key = str(patch_path.resolve()).lower()
        if patch_key in seen_patches:
            skipped["duplicate_patch"] += 1
            skipped_rows.append({"row": csv_row_number, "reason": "duplicate_patch", "patch": patch_name})
            continue
        seen_patches.add(patch_key)

        try:
            vv, vh, original_shape, detected_axis = read_polarizations(
                patch_path,
                vv_channel=args.vv_channel,
                vh_channel=args.vh_channel,
                channel_axis=args.channel_axis,
            )
            yolo_bbox = format_yolo_bbox(
                label,
                bbox_to_yolo(axis_aligned_bbox_local(row, image_shape=vv.shape), vv.shape),
            )
            rgb = dual_pol_to_rgb_uint8(vv, vh)
            image_name = f"{patch_path.stem}_vv_vh_rgb.png"
            resize_rgb_image(rgb, args.imgsz).save(images_dir / image_name)
            (labels_dir / f"{Path(image_name).stem}.txt").write_text(yolo_bbox + "\n", encoding="utf-8")
        except Exception as exc:  # keep preparing other rows and report all failures
            reason = f"error_{type(exc).__name__}"
            skipped[reason] += 1
            skipped_rows.append(
                {
                    "row": csv_row_number,
                    "reason": reason,
                    "patch": patch_name,
                    "error": str(exc),
                }
            )
            continue

        class_counts[int(label)] += 1
        axis_counts[detected_axis] += 1
        manifest.append(
            {
                "split": "test",
                "source_dataset": args.source_name,
                "metadata_row": csv_row_number,
                "scene": str(row.get("scene", "")),
                "category": str(row.get("category", "")),
                "image_name": image_name,
                "source_patch_cal": Path(patch_name).name,
                "original_tiff_shape": "x".join(str(value) for value in original_shape),
                "detected_channel_axis": detected_axis,
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "yolo_bbox": yolo_bbox,
            }
        )

    if not manifest:
        raise ValueError("No valid OpenSARShip samples were produced.")

    write_csv(output_dir / "external_test_manifest.csv", manifest)
    write_csv(
        output_dir / "skipped_rows.csv",
        skipped_rows,
        fieldnames=["row", "reason", "patch", "error"],
    )
    write_data_yaml(output_dir)

    image_count = len(list(images_dir.glob("*.png")))
    label_count = len(list(labels_dir.glob("*.txt")))
    if image_count != len(manifest) or label_count != len(manifest):
        raise RuntimeError(
            f"Output count mismatch: images={image_count}, labels={label_count}, manifest={len(manifest)}"
        )

    print("PASS: OpenSARShip external YOLO test set prepared.")
    print(f"Source metadata: {metadata_path}")
    print(f"Source PATCH_CAL: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Test images: {image_count}")
    print(f"Test labels: {label_count}")
    print("Class counts:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id} {class_name}: {class_counts[class_id]}")
    print(f"Channel axis counts: {dict(axis_counts)}")
    print(f"Skipped rows: {sum(skipped.values())} {dict(skipped)}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
