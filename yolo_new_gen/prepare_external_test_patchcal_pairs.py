"""Prepare all available paired VV/VH PATCH_CAL files as one YOLO test split."""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import CLASS_NAMES, map_ship_class
from yolo_exp.bbox_utils import axis_aligned_bbox_local, bbox_to_yolo, dual_pol_to_rgb_uint8
from yolo_new_gen.prepare_data import format_yolo_bbox, resize_rgb_image


def write_data_yaml(output_dir):
    root = Path(output_dir).resolve().as_posix()
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


def main():
    parser = argparse.ArgumentParser(description="Prepare paired PATCH_CAL files as external test only.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vv-column", default="patch_cal_vv_actual_file")
    parser.add_argument("--vh-column", default="patch_cal_vh_actual_file")
    parser.add_argument("--imgsz", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    test_dir = output_dir / "test"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"PATCH_CAL folder not found: {image_dir}")
    if test_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {test_dir}. Use --overwrite to regenerate.")
        shutil.rmtree(test_dir)

    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    metadata = pd.read_csv(metadata_path)
    required = [
        "category", "Center_x", "Center_y", "Head_x", "Head_y",
        "Tail_x", "Tail_y", "UpperLeft_x", "UpperLeft_y",
        "LowerRight_x", "LowerRight_y", args.vv_column, args.vh_column,
    ]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise KeyError("Missing metadata columns: " + ", ".join(missing))

    manifest = []
    class_counts = Counter()
    skipped = Counter()
    seen = set()

    for _, row in metadata.iterrows():
        label = map_ship_class(row)
        if label is None:
            skipped["non_target"] += 1
            continue

        vv_name = str(row.get(args.vv_column, "")).strip()
        vh_name = str(row.get(args.vh_column, "")).strip()
        if not vv_name or not vh_name or vv_name.lower() == "nan" or vh_name.lower() == "nan":
            skipped["blank_name"] += 1
            continue

        vv_path = image_dir / vv_name
        vh_path = image_dir / vh_name
        if not vv_path.is_file() or not vh_path.is_file():
            skipped["missing_pair"] += 1
            continue

        pair_key = (str(vv_path.resolve()).lower(), str(vh_path.resolve()).lower())
        if pair_key in seen:
            skipped["duplicate_pair"] += 1
            continue
        seen.add(pair_key)

        vv = tifffile.imread(vv_path)
        vh = tifffile.imread(vh_path)
        if vv.shape != vh.shape:
            skipped["shape_mismatch"] += 1
            continue

        bbox = format_yolo_bbox(
            label,
            bbox_to_yolo(axis_aligned_bbox_local(row, image_shape=vv.shape), vv.shape),
        )
        image_name = f"{Path(vv_name).stem}_vv_vh_rgb.png"
        resize_rgb_image(dual_pol_to_rgb_uint8(vv, vh), args.imgsz).save(images_dir / image_name)
        (labels_dir / f"{Path(image_name).stem}.txt").write_text(bbox + "\n", encoding="utf-8")

        class_counts[int(label)] += 1
        manifest.append(
            {
                "split": "test",
                "scene": str(row.get("scene", "")),
                "category": str(row.get("category", "")),
                "image_name": image_name,
                "source_vv_patch_name": vv_name,
                "source_vh_patch_name": vh_name,
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "yolo_bbox": bbox,
            }
        )

    if not manifest:
        raise ValueError("No complete target-class VV/VH pairs were found.")

    with (output_dir / "external_test_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_data_yaml(output_dir)

    image_count = len(list(images_dir.glob("*.png")))
    label_count = len(list(labels_dir.glob("*.txt")))
    if image_count != len(manifest) or label_count != len(manifest):
        raise RuntimeError(
            f"Output count mismatch: images={image_count}, labels={label_count}, manifest={len(manifest)}"
        )

    print("PASS: external-test-only dataset prepared.")
    print(f"Test images: {image_count}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id} {class_name}: {class_counts[class_id]}")
    print(f"Missing/incomplete pairs skipped: {skipped['missing_pair']}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
