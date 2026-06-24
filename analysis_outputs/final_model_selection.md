# Final model selection

Model terbaik final yang dipakai sebagai acuan utama laporan:

- Model: **YOLOv12N**
- Split: **80:10:10**
- Epoch: **50**
- Batch: **16**
- Image size: **192**
- Optimizer: **AdamW**
- LR0: **0.0005**
- Mosaic/Mixup: **0.0 / 0.0**
- Seed: **7**

Dasar pemilihan: nilai **validation mAP50 tertinggi** pada rekap final batch 16.

Ringkasan performa:

- Validation mAP50: **79.12%**
- Validation F1: **71.56%**
- Internal test mAP50: **75.51%**
- Internal test F1: **61.83%**

Catatan pembanding: split 70:15:15 terbaik adalah YOLOv12L epoch 100 batch 16, tetapi mAP50 validasinya lebih rendah daripada YOLOv12N epoch 50 pada split 80:10:10.
