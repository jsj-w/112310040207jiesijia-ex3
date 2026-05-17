# Traffic Sign Detection Challenge

## Task
Train an object detection model with the provided YOLO dataset and predict objects on the hidden-label test set.

## Classes
Green Light, Red Light, Speed Limit 10, Speed Limit 100, Speed Limit 110, Speed Limit 120, Speed Limit 20, Speed Limit 30, Speed Limit 40, Speed Limit 50, Speed Limit 60, Speed Limit 70, Speed Limit 80, Speed Limit 90, Stop

## Directory
- `train/`: training images and labels
- `val/`: validation images and labels
- `test/images/`: test images only
- `data.yaml`: original Ultralytics training config
- `sample_submission.csv`: submission schema
- `baseline_infer.py`: original baseline inference script

## Submission
Submit one `submission.csv` file with these columns:
- `image_id`
- `class_id`
- `x_center`
- `y_center`
- `width`
- `height`
- `confidence`

All coordinates must be YOLO-style normalized values in `[0, 1]`.

## Metric
Ranking metric: `mAP@0.5`

## Added Files
- `requirements.txt`: recommended dependencies
- `traffic_sign_utils.py`: shared utilities for CSV IO, label parsing, mAP@0.5 scoring and box fusion
- `build_balanced_dataset.py`: oversamples rare classes by repeating image paths in a generated train list
- `train_detector.py`: stronger Ultralytics training entrypoint
- `search_best_inference.py`: searches validation-time inference settings for better `mAP@0.5`
- `generate_submission.py`: generates the final `submission.csv`
- `score_submission.py`: evaluates any prediction CSV on the validation set

## PyCharm Setup
1. Open this folder as a project.
2. Create or select a Python interpreter.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

If you have a CUDA GPU, install the matching GPU version of PyTorch first, then run the command above.

## Recommended High-Score Pipeline

### Step 1: Build a balanced training set
This project is class-imbalanced, especially for rare speed-limit categories. The script below keeps the original dataset untouched and creates a repeated train list for rare classes.

```bash
python build_balanced_dataset.py
```

Generated files:
- `generated/balanced_train.txt`
- `generated/balanced_data.yaml`
- `generated/balanced_summary.json`

### Step 2: Train the detector
Recommended first run on GPU:

```bash
python train_detector.py --device 0 --model yolov8n.pt --epochs 30 --imgsz 640 --batch 8 --cache --amp
```

If your GPU memory is larger, try a stronger model:

```bash
python train_detector.py --device 0 --model yolov8s.pt --epochs 40 --imgsz 736 --batch 6 --cache --amp --name traffic_sign_yolov8s_gpu
```

If you only have CPU, reduce the model and image size so the run stays practical:

```bash
python train_detector.py --device cpu --model yolov8n.pt --epochs 30 --imgsz 640 --batch 2 --name traffic_sign_cpu
```

Weights are usually saved under:
- `runs/train/<run_name>/weights/best.pt`

### Step 3: Tune validation-time inference parameters
This step usually gives a better final `mAP@0.5` than using default thresholds directly.

Single-model tuning:

```bash
python search_best_inference.py --models runs/train/traffic_sign_strong/weights/best.pt --save-best-csv
```

Two-model ensemble tuning:

```bash
python search_best_inference.py --models runs/train/traffic_sign_strong/weights/best.pt runs/train/traffic_sign_yolov8m/weights/best.pt --save-best-csv
```

Main outputs:
- `runs/tuning/tuning_summary.json`
- `runs/tuning/best_val_predictions.csv`

### Step 4: Generate the final submission
Use the tuned parameters directly:

```bash
python generate_submission.py --models runs/train/traffic_sign_strong/weights/best.pt --config runs/tuning/tuning_summary.json --output submission.csv
```

For an ensemble:

```bash
python generate_submission.py --models runs/train/traffic_sign_strong/weights/best.pt runs/train/traffic_sign_yolov8m/weights/best.pt --config runs/tuning/tuning_summary.json --output submission.csv
```

## Useful Validation Commands
Evaluate a validation CSV:

```bash
python score_submission.py --csv runs/tuning/best_val_predictions.csv
```

Run the original baseline export script:

```bash
python baseline_infer.py --model runs/train/traffic_sign_strong/weights/best.pt --test-dir test/images --output submission.csv
```

## Practical Tips For Higher mAP@0.5
- Prefer `yolov8s.pt` or `yolov8m.pt` over `yolov8n.pt` if you have a GPU.
- Keep `imgsz` relatively large because traffic signs are often small targets.
- Train at least two runs with different model sizes or seeds, then tune them together as an ensemble.
- Use `build_balanced_dataset.py` before training, because rare classes are heavily underrepresented.
- Do not submit the first threshold you try; always run `search_best_inference.py` on `val/` first.
- If overfitting appears late in training, lower `epochs` or increase `close_mosaic`.
- If `python -c "import torch; print(torch.cuda.is_available())"` prints `False`, your current interpreter is using CPU-only PyTorch and will not train on GPU until you reinstall CUDA-enabled PyTorch.

## Minimal Baseline Commands
Original training example:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416
```

Original submission generation example:

```bash
python baseline_infer.py --model runs/detect/train/weights/best.pt --test-dir test/images --output submission.csv
```
