# OpenSARShip YOLOv12 SAR Patch Detection

Repository ini berisi pipeline final untuk membuat dataset YOLO dan training YOLOv12 object detection pada patch SAR GFW/OpenSARShip. Eksperimen final memakai deteksi objek pada patch SAR 3 kelas: Fishing, Cargo, Passenger.

## Struktur Utama

```text
.
|-- labels.py
|-- requirements.txt
|-- gfw/
|   |-- Patch/
|   `-- metadata/
|       `-- metadata_with_vv_vh_gfw_ais_identity.csv
|-- yolo_exp/
|   `-- bbox_utils.py
|-- yolo_new_gen/
|   |-- prepare_data.py
|   |-- validate_dataset.py
|   |-- train_yolo.py
|   |-- evaluate_test.py
|   |-- evaluate_all_weights.py
|   `-- dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/
`-- runs/
    |-- detect/
    `-- detect_test/
```

Raw metadata dan raw TIFF tidak diubah oleh script. Filtering kelas dilakukan saat membuat dataset YOLO.

## Kelas Final

| ID | Kelas |
|---:|---|
| 0 | Fishing |
| 1 | Cargo |
| 2 | Passenger |

Kategori di luar tiga kelas target tidak dimasukkan ke dataset YOLO.

## Input Citra

Input YOLO berupa PNG RGB gabungan dual-pol SAR:

```text
R = VV
G = VH
B = mean(VV, VH)
```

Kolom metadata yang dipakai:

- `patch_vv_actual_file`
- `patch_vh_actual_file`

## Dataset Final

Dataset final yang sedang dipakai:

```text
yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7
```

Setting dataset:

```text
metadata     : gfw/metadata/metadata_with_vv_vh_gfw_ais_identity.csv
image dir    : gfw/Patch
imgsz        : 192
split        : train/val/test = 80/10/10
split group  : scene
seed         : 7
image mode   : vv_vh_rgb
```

Actual split count:

| Split | Total | Fishing | Cargo | Passenger |
|---|---:|---:|---:|---:|
| train | 431 | 188 | 144 | 99 |
| val | 56 | 22 | 23 | 11 |
| test | 47 | 22 | 11 | 14 |

Prepare dataset:

```powershell
python yolo_new_gen/prepare_data.py `
  --metadata gfw/metadata/metadata_with_vv_vh_gfw_ais_identity.csv `
  --image-dir gfw/Patch `
  --output-dir yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7 `
  --imgsz 192 `
  --split-group scene `
  --combine-vv-vh `
  --seed 7 `
  --train-ratio 0.80 `
  --val-ratio 0.10 `
  --test-ratio 0.10
```

Validasi dataset:

```powershell
python yolo_new_gen/validate_dataset.py --dataset-dir yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7
```

## Training Final

Setting training yang sedang dipakai:

```text
model         : yolo12n/s/m/x/l.pt
epochs        : 50 / 100 / 150
imgsz         : 192
batch         : 8
device        : 0
seed          : 7
optimizer     : AdamW
lr0           : 0.0005
lrf           : 0.01
cos_lr        : true
weight_decay  : 0.0005
mosaic        : 0.0
mixup         : 0.0
close_mosaic  : 10
deterministic : true
patience      : same as epochs, practical no early stop
```

Training satu model, contoh YOLOv12N epoch 100:

```powershell
python yolo_new_gen/train_yolo.py `
  --version 12 `
  --variant n `
  --task detect `
  --data yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/data.yaml `
  --epochs 100 `
  --patience 100 `
  --imgsz 192 `
  --batch 8 `
  --device 0 `
  --seed 7 `
  --deterministic `
  --optimizer AdamW `
  --lr0 0.0005 `
  --lrf 0.01 `
  --cos-lr `
  --weight-decay 0.0005 `
  --mosaic 0.0 `
  --mixup 0.0 `
  --close-mosaic 10 `
  --workers 8 `
  --output runs/detect/YOLOV12N_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop
```

Training semua varian epoch 50, 100, 150:

```powershell
python yolo_new_gen/compare_experiments.py
```

## Evaluasi Test

Evaluasi `best.pt` pada split test:

```powershell
python yolo_new_gen/evaluate_test.py `
  --model runs/detect/YOLOV12N_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop/weights/best.pt `
  --data yolo_new_gen/dataset_yolo_det_192_3class_vv_vh_rgb_scene_80_10_10_seed7/data.yaml `
  --imgsz 192 `
  --batch 8 `
  --device 0 `
  --name YOLOV12N_192_E100_B8_80_10_10_seed7_test
```

Output test disimpan ke:

```text
runs/detect_test/<name>/test_metrics.csv
runs/detect_test/<name>/args.yaml
```

## Hasil Test Tersimpan

| Model | Test images | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv12N E100 B8 | 47 | 0.6793 | 0.6992 | 0.7867 | 0.4401 |
| YOLOv12S E100 B8 | 47 | 0.6027 | 0.7966 | 0.7258 | 0.4221 |

Run final yang dipertahankan:

```text
runs/detect/YOLOV12N_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop
runs/detect/YOLOV12S_192_E100_B8_3class_VV_VH_RGB_scene_80_10_10_seed7_adamw_lr0005_nomosaic_noearlystop
runs/detect_test/YOLOV12N_192_E100_B8_80_10_10_seed7_test
runs/detect_test/YOLOV12S_192_E100_B8_80_10_10_seed7_test
```

Run non-final diarsipkan di:

```text
runs/archive_nonfinal_runs_20260620_122110
```

## Catatan

Test split hanya 47 gambar, sehingga metrik test sensitif terhadap komposisi split. Untuk klaim final, gunakan hasil test dan catat konfigurasi split/datasetnya dengan jelas.