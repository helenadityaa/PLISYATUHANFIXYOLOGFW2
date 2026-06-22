"""Analyze SAR ship characteristics from the final YOLO detection dataset."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is normally installed with Ultralytics.
    yaml = None


CLASS_NAMES = {
    0: "Fishing",
    1: "Cargo",
    2: "Passenger",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

FEATURE_DESCRIPTIONS = {
    "split": "Dataset split: train, val, or test.",
    "image_name": "Image file name.",
    "class_id": "YOLO class id from the ground truth label.",
    "class_name": "Ship class name.",
    "mean_intensity_vv": "Mean intensity of channel R, interpreted as VV.",
    "mean_intensity_vh": "Mean intensity of channel G, interpreted as VH.",
    "mean_intensity_rgb_mean": "Mean intensity of channel B, interpreted as mean(VV, VH).",
    "max_intensity_vv": "Maximum intensity of channel R, interpreted as VV.",
    "max_intensity_vh": "Maximum intensity of channel G, interpreted as VH.",
    "std_intensity_vv": "Standard deviation of channel R, interpreted as VV.",
    "std_intensity_vh": "Standard deviation of channel G, interpreted as VH.",
    "vv_vh_difference": "Difference between mean VV and mean VH intensity.",
    "bright_area_ratio": "Ratio of pixels above grayscale mean plus one standard deviation.",
    "entropy": "Shannon entropy of grayscale intensity histogram.",
    "bbox_x_center": "Normalized YOLO bounding box x center.",
    "bbox_y_center": "Normalized YOLO bounding box y center.",
    "bbox_width": "Normalized YOLO bounding box width.",
    "bbox_height": "Normalized YOLO bounding box height.",
    "bbox_area_ratio": "Normalized bounding box area, width multiplied by height.",
    "bbox_aspect_ratio": "Bounding box width divided by height.",
    "object_mean_intensity": "Mean grayscale intensity inside the bounding box.",
    "background_mean_intensity": "Mean grayscale intensity outside the bounding box.",
    "object_background_contrast": "Object mean intensity minus background mean intensity.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze visual characteristics of SAR ship objects in the final YOLO dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7",
        help="Path to the final YOLO detection dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory for CSV, Excel, plots, and annotated samples.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Maximum number of annotated sample images to save per class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    return parser.parse_args()


def read_data_yaml(data_yaml: Path) -> Dict[str, object]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if yaml is not None:
        with data_yaml.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            return loaded
    return read_simple_yaml(data_yaml)


def read_simple_yaml(data_yaml: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    names: Dict[int, str] = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            data["names"] = names
            continue
        if in_names and raw_line.startswith((" ", "\t")) and ":" in stripped:
            key, value = stripped.split(":", 1)
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                pass
            continue
        in_names = False
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            data[key.strip()] = value.strip().strip("'\"")
    return data


def read_split_manifest(dataset_dir: Path) -> Optional[pd.DataFrame]:
    manifest_path = dataset_dir / "split_manifest.csv"
    if not manifest_path.exists():
        return None
    return pd.read_csv(manifest_path)


def iter_image_paths(images_dir: Path) -> Iterable[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            print(f"Warning: skipping malformed label line {label_path}:{line_number}")
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            print(f"Warning: skipping non-numeric label line {label_path}:{line_number}")
            continue
        if class_id in CLASS_NAMES:
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def load_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_entropy(grayscale: np.ndarray) -> float:
    if grayscale.size == 0:
        return math.nan
    hist, _ = np.histogram(grayscale, bins=256, range=(0, 255))
    total = hist.sum()
    if total == 0:
        return math.nan
    probabilities = hist.astype(np.float64) / float(total)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalized_bbox_to_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x_min = int(math.floor((x_center - width / 2.0) * image_width))
    y_min = int(math.floor((y_center - height / 2.0) * image_height))
    x_max = int(math.ceil((x_center + width / 2.0) * image_width))
    y_max = int(math.ceil((y_center + height / 2.0) * image_height))

    x_min = max(0, min(image_width, x_min))
    y_min = max(0, min(image_height, y_min))
    x_max = max(0, min(image_width, x_max))
    y_max = max(0, min(image_height, y_max))

    if x_max <= x_min:
        x_max = min(image_width, x_min + 1)
    if y_max <= y_min:
        y_max = min(image_height, y_min + 1)
    return x_min, y_min, x_max, y_max


def image_level_features(rgb: np.ndarray) -> Dict[str, float]:
    vv = rgb[:, :, 0]
    vh = rgb[:, :, 1]
    rgb_mean_channel = rgb[:, :, 2]
    grayscale = rgb.mean(axis=2)
    bright_threshold = float(grayscale.mean() + grayscale.std())
    return {
        "mean_intensity_vv": float(vv.mean()),
        "mean_intensity_vh": float(vh.mean()),
        "mean_intensity_rgb_mean": float(rgb_mean_channel.mean()),
        "max_intensity_vv": float(vv.max()),
        "max_intensity_vh": float(vh.max()),
        "std_intensity_vv": float(vv.std()),
        "std_intensity_vh": float(vh.std()),
        "vv_vh_difference": float(vv.mean() - vh.mean()),
        "bright_area_ratio": float((grayscale > bright_threshold).mean()),
        "entropy": compute_entropy(grayscale),
    }


def object_features(
    grayscale: np.ndarray,
    x_center: float,
    y_center: float,
    bbox_width: float,
    bbox_height: float,
) -> Dict[str, float]:
    image_height, image_width = grayscale.shape
    x_min, y_min, x_max, y_max = normalized_bbox_to_pixels(
        x_center, y_center, bbox_width, bbox_height, image_width, image_height
    )
    object_pixels = grayscale[y_min:y_max, x_min:x_max]

    background_mask = np.ones(grayscale.shape, dtype=bool)
    background_mask[y_min:y_max, x_min:x_max] = False
    background_pixels = grayscale[background_mask]

    object_mean = float(object_pixels.mean()) if object_pixels.size else math.nan
    background_mean = float(background_pixels.mean()) if background_pixels.size else math.nan
    contrast = object_mean - background_mean if not math.isnan(background_mean) else math.nan
    aspect_ratio = bbox_width / bbox_height if bbox_height else math.nan

    return {
        "bbox_x_center": x_center,
        "bbox_y_center": y_center,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_ratio": bbox_width * bbox_height,
        "bbox_aspect_ratio": aspect_ratio,
        "object_mean_intensity": object_mean,
        "background_mean_intensity": background_mean,
        "object_background_contrast": contrast,
    }


def collect_features(dataset_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        for image_path in iter_image_paths(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt"
            labels = read_yolo_labels(label_path)
            if not labels:
                continue

            rgb = load_rgb_array(image_path)
            grayscale = rgb.mean(axis=2)
            base_features = image_level_features(rgb)

            for class_id, x_center, y_center, bbox_width, bbox_height in labels:
                row: Dict[str, object] = {
                    "split": split,
                    "image_name": image_path.name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                }
                row.update(base_features)
                row.update(object_features(grayscale, x_center, y_center, bbox_width, bbox_height))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_class(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["class_name", "feature", "mean", "median", "std", "min", "max"])
    numeric_columns = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column != "class_id"
    ]
    rows: List[Dict[str, object]] = []
    for class_name, group in features.groupby("class_name", sort=True):
        for column in numeric_columns:
            values = group[column].dropna()
            rows.append(
                {
                    "class_name": class_name,
                    "feature": column,
                    "mean": float(values.mean()) if not values.empty else math.nan,
                    "median": float(values.median()) if not values.empty else math.nan,
                    "std": float(values.std()) if not values.empty else math.nan,
                    "min": float(values.min()) if not values.empty else math.nan,
                    "max": float(values.max()) if not values.empty else math.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_by_split(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["split", "class_name", "object_count"])
    return (
        features.groupby(["split", "class_name"], sort=True)
        .size()
        .reset_index(name="object_count")
    )


def feature_description_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature": feature, "description": description} for feature, description in FEATURE_DESCRIPTIONS.items()]
    )


def write_summary_excel(features: pd.DataFrame, output_path: Path) -> None:
    summary_by_class = summarize_by_class(features)
    summary_by_split = summarize_by_split(features)
    descriptions = feature_description_frame()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        features.to_excel(writer, index=False, sheet_name="Raw_Features")
        summary_by_class.to_excel(writer, index=False, sheet_name="Summary_By_Class")
        summary_by_split.to_excel(writer, index=False, sheet_name="Summary_By_Split")
        descriptions.to_excel(writer, index=False, sheet_name="Feature_Description")


def make_boxplot(features: pd.DataFrame, feature_name: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    if features.empty or feature_name not in features.columns:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.axis("off")
    else:
        features.boxplot(column=feature_name, by="class_name", grid=False)
        plt.title(feature_name)
        plt.suptitle("")
        plt.xlabel("Class")
        plt.ylabel(feature_name)
        plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def create_plots(features: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_features = [
        "mean_intensity_vv",
        "mean_intensity_vh",
        "bright_area_ratio",
        "entropy",
        "bbox_area_ratio",
        "object_background_contrast",
    ]
    for feature_name in plot_features:
        make_boxplot(features, feature_name, plots_dir / f"boxplot_{feature_name}.png")


def draw_ground_truth_boxes(image_path: Path, label_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    colors = {
        0: (255, 60, 60),
        1: (60, 180, 255),
        2: (80, 220, 120),
    }
    for class_id, x_center, y_center, bbox_width, bbox_height in read_yolo_labels(label_path):
        x_min, y_min, x_max, y_max = normalized_bbox_to_pixels(
            x_center, y_center, bbox_width, bbox_height, image_width, image_height
        )
        color = colors.get(class_id, (255, 255, 0))
        label = CLASS_NAMES.get(class_id, str(class_id))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        text_position = (x_min, max(0, y_min - 12))
        draw.text(text_position, label, fill=color)
    annotated.save(output_path)


def save_samples_per_class(
    features: pd.DataFrame,
    dataset_dir: Path,
    sample_root: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    sample_root.mkdir(parents=True, exist_ok=True)
    if features.empty or samples_per_class <= 0:
        return

    rng = random.Random(seed)
    for class_id, class_name in CLASS_NAMES.items():
        class_dir = sample_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_rows = features[features["class_id"] == class_id][["split", "image_name"]].drop_duplicates()
        candidates = list(class_rows.itertuples(index=False, name=None))
        rng.shuffle(candidates)
        for split, image_name in candidates[:samples_per_class]:
            image_path = dataset_dir / split / "images" / image_name
            label_path = dataset_dir / split / "labels" / f"{Path(image_name).stem}.txt"
            if not image_path.exists():
                continue
            output_name = f"{split}_{image_path.stem}_gt{image_path.suffix}"
            draw_ground_truth_boxes(image_path, label_path, class_dir / output_name)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    read_data_yaml(dataset_dir / "data.yaml")
    manifest = read_split_manifest(dataset_dir)
    if manifest is not None:
        print(f"Loaded split_manifest.csv with {len(manifest)} rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    features = collect_features(dataset_dir)

    raw_csv = output_dir / "ship_characteristics_raw.csv"
    summary_excel = output_dir / "ship_characteristics_summary.xlsx"
    features.to_csv(raw_csv, index=False)
    write_summary_excel(features, summary_excel)
    create_plots(features, output_dir / "plots")
    save_samples_per_class(
        features,
        dataset_dir,
        output_dir / "sample_per_class",
        args.samples_per_class,
        args.seed,
    )

    print(f"Saved raw features to {raw_csv}")
    print(f"Saved summary workbook to {summary_excel}")
    print(f"Saved plots to {output_dir / 'plots'}")
    print(f"Saved annotated samples to {output_dir / 'sample_per_class'}")


if __name__ == "__main__":
    main()
