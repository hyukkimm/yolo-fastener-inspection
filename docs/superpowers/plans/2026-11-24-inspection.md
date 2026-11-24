# Fastener inspection implementation plan

Goal: Deliver the Python/OpenCV/YOLO workflow already discussed with the user in their existing desktop repository.

Design: One defect class for screw and metal_nut. Convert each connected mask region to a box. Keep originals untouched; copy images into the project. Repartition each source category/defect stratum 60/20/20 with seed 42. This is a supervised adaptation, not the standard MVTec anomaly benchmark. No automatic OK decision without product presence verification.

Stack: Python 3.10 virtual environment, Ultralytics, OpenCV, PyTorch. CPU default on this Intel-only machine. YOLO11n baseline; 640px full training, 320px smoke check.

- [x] Test mask boundaries, disconnected components, empty masks and deterministic disjoint splits before implementation.
- [x] Implement prepare_dataset.py with Unicode-safe reads, mask validation, SHA256 duplicate leakage guard and manifest. Refuse existing output folders.
- [x] Add train.py, evaluate.py, predict.py, camera.py and shared path/device helpers. Evaluation reports localization and per-product image-level confusion counts.
- [x] Prepare actual data, audit labels and run the complete test suite.
- [x] Run a bounded CPU smoke training and inference; keep its weights separate from full training.
- [x] Document exact PowerShell commands, camera limitations and noncommercial dataset terms; review final changes.

Review focus: Korean paths; Windows multiprocessing; negative images with empty labels; no missing/empty defect masks; no test data in training; missing trained weights; unavailable camera; no false OK on empty frame.

Ruling: The earlier concrete workflow and request to write it authorize implementation; no additional design approval requested. Reuse the user-selected checkout, no commits/pushes.

Verification: 6 tests passed; 815 unique source images, 267 valid boxes; train/val/test = 484/158/173. CPU one-epoch smoke training, image prediction and validation-report generation succeeded. Live camera and full training were not run. Reviewer found no blocking issue; custom smoke run names now supported.
