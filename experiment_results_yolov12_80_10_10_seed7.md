# YOLOv12 Test Results - 80/10/10 Seed 7

Ringkasan ini menyimpan setting dan hasil test untuk eksperimen yang menghasilkan:

- YOLOv12N test mAP50 0.7867 dan mAP50-95 0.4401
- YOLOv12S test mAP50 0.7258 dan mAP50-95 0.4221

## Dataset

- Metadata: `gfw/metadata/metadata_with_vv_vh_gfw_ais_identity.csv`
- Image dir: `gfw/Patch`
- Dataset YOLO: `yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7`
- Image size: `192`
- Image mode: VV/VH RGB (`--combine-vv-vh`)
- Split group: `scene`
- Requested split: train/val/test = `80/10/10`
- Seed split: `7`
- Actual split count: train `431`, val `56`, test `47`

Class count:

| Split | Fishing | Cargo | Passenger |
|---|---:|---:|---:|
| train | 188 | 144 | 99 |
| val | 22 | 23 | 11 |
| test | 22 | 11 | 14 |

Prepare command:

```powershell
python -c "from pathlib import Path; import yolo_new_gen.prepare_data as p; p.TRAIN_RATIO=0.80; p.VAL_RATIO=0.10; p.TEST_RATIO=0.10; p.prepare_data_clean(Path('gfw/metadata/metadata_with_vv_vh_gfw_ais_identity.csv'), Path('gfw/Patch'), Path('yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7'), 192, p.DEFAULT_IMAGE_COLUMN, 'scene', True, p.DEFAULT_VV_COLUMN, p.DEFAULT_VH_COLUMN, 7)"
```

## Shared Training Setting

- YOLO version: `12`
- Task: `detect`
- Epochs: `100`
- Batch: `8`
- Image size: `192`
- Device: `0`
- Seed training: `7`
- Deterministic: enabled
- Optimizer: `AdamW`
- LR0: `0.0005`
- LRF: `0.01`
- Cosine LR: enabled
- Weight decay: `0.0005`
- Mosaic: `0.0`
- Mixup: `0.0`
- Close mosaic: `10`
- Workers: `8`
- Patience: `100` / no practical early stop for 100 epochs

## Test Results

| Model | Test images | Precision | Recall | mAP50 | mAP50-95 | best.pt |
|---|---:|---:|---:|---:|---:|---|
| YOLOv12N | 47 | 0.6793 | 0.6992 | 0.7867 | 0.4401 | `runs/detect/YOLOV12N_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop/weights/best.pt` |
| YOLOv12S | 47 | 0.6027 | 0.7966 | 0.7258 | 0.4221 | `runs/detect/YOLOV12S_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop/weights/best.pt` |

Per-class test results from validation output:

| Model | Class | mAP50 | mAP50-95 |
|---|---|---:|---:|
| YOLOv12N | Fishing | 0.931 | 0.653 |
| YOLOv12N | Cargo | 0.726 | 0.398 |
| YOLOv12N | Passenger | 0.703 | 0.269 |
| YOLOv12S | Fishing | 0.909 | 0.640 |
| YOLOv12S | Cargo | 0.697 | 0.356 |
| YOLOv12S | Passenger | 0.572 | 0.270 |

## Training Commands

YOLOv12N:

```powershell
python yolo_new_gen/train_yolo.py --version 12 --variant n --task detect --data yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/data.yaml --epochs 100 --patience 100 --imgsz 192 --batch 8 --device 0 --seed 7 --deterministic --optimizer AdamW --lr0 0.0005 --lrf 0.01 --cos-lr --weight-decay 0.0005 --mosaic 0.0 --mixup 0.0 --close-mosaic 10 --workers 8 --output runs/detect/YOLOV12N_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop
```

YOLOv12S:

```powershell
python yolo_new_gen/train_yolo.py --version 12 --variant s --task detect --data yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/data.yaml --epochs 100 --patience 100 --imgsz 192 --batch 8 --device 0 --seed 7 --deterministic --optimizer AdamW --lr0 0.0005 --lrf 0.01 --cos-lr --weight-decay 0.0005 --mosaic 0.0 --mixup 0.0 --close-mosaic 10 --workers 8 --output runs/detect/YOLOV12S_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop
```

## Test Commands

YOLOv12N:

```powershell
python yolo_new_gen/evaluate_test.py --model runs/detect/YOLOV12N_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop/weights/best.pt --data yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/data.yaml --imgsz 192 --batch 8 --device 0 --name YOLOV12N_192_E100_B8_80_10_10_seed7_test
```

YOLOv12S:

```powershell
python yolo_new_gen/evaluate_test.py --model runs/detect/YOLOV12S_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop/weights/best.pt --data yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/data.yaml --imgsz 192 --batch 8 --device 0 --name YOLOV12S_192_E100_B8_80_10_10_seed7_test
```

## Saved Metric Files

- `runs/detect_test/YOLOV12N_192_E100_B8_80_10_10_seed7_test/test_metrics.csv`
- `runs/detect_test/YOLOV12S_192_E100_B8_80_10_10_seed7_test/test_metrics.csv`

Note: test split contains only 47 images, so the test metrics can be sensitive to split composition.
