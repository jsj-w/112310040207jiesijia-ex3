from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a stronger traffic sign detector with Ultralytics YOLO.",
    )
    parser.add_argument("--data", default=None, help="Dataset YAML. Defaults to generated/balanced_data.yaml if it exists.")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO checkpoint or local weights path.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size. Lower it if VRAM or RAM is tight.")
    parser.add_argument(-"--device", default="", help="Training device, for example 0, 0,1 or cpu.")
    parser.add_argument("--project", default="runs/train", help="Ultralytics project directory.")
    parser.add_argument("--name", default="traffic_sign_strong", help="Ultralytics run name.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted training run.")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=0.003, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning-rate fraction.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout used by some heads.")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain.")
    parser.add_argument("--cls", type=float, default=0.7, help="Class loss gain.")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain.")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic probability.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp probability.")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste probability.")
    parser.add_argument("--close-mosaic", type=int, default=8, help="Disable mosaic in the final N epochs.")
    parser.add_argument("--scale", type=float, default=0.35, help="Scale augmentation strength.")
    parser.add_argument("--translate", type=float, default=0.1, help="Translate augmentation strength.")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation strength.")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear augmentation strength.")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation strength.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability.")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV value augmentation.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate scheduling.")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--single-cls", action="store_true", help="Train as a single-class detector.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    local_config_dir = project_root / ".ultralytics"
    local_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_config_dir))
    data_path = resolve_data_path(project_root, args.data)

    try:
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install dependencies first, for example:\n"
            "pip install -r requirements.txt"
        ) from exc

    runtime_device = resolve_runtime_device(args.device, torch)
    args.project = str(resolve_project_dir(project_root, args.project))
    apply_safe_defaults(args=args, runtime_device=runtime_device)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": runtime_device,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "cache": args.cache,
        "resume": args.resume,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "close_mosaic": args.close_mosaic,
        "scale": args.scale,
        "translate": args.translate,
        "degrees": args.degrees,
        "shear": args.shear,
        "perspective": args.perspective,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "cos_lr": args.cos_lr,
        "multi_scale": args.multi_scale,
        "amp": args.amp,
        "single_cls": args.single_cls,
        "plots": True,
        "save": True,
        "exist_ok": True,
        "verbose": True,
    }

    print(f"Using dataset config: {data_path}")
    print(f"Using device: {runtime_device}")
    print(
        "Training settings: "
        f"model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, workers={args.workers}, mosaic={args.mosaic}, "
        f"mixup={args.mixup}, copy_paste={args.copy_paste}"
    )
    print(f"Starting training with model: {args.model}")
    model.train(**train_kwargs)


def resolve_data_path(project_root: Path, requested_path: str | None) -> Path:
    if requested_path:
        return Path(requested_path).resolve()
    balanced_path = project_root / "generated" / "balanced_data.yaml"
    if balanced_path.exists():
        return balanced_path
    return (project_root / "data.yaml").resolve()


def resolve_project_dir(project_root: Path, requested_path: str) -> Path:
    project_path = Path(requested_path)
    if project_path.is_absolute():
        return project_path
    return (project_root / project_path).resolve()


def resolve_runtime_device(requested_device: str, torch_module: object) -> str:
    requested = requested_device.strip()
    if requested:
        if requested != "cpu" and not torch_module.cuda.is_available():
            raise SystemExit(
                "You requested GPU training, but the current PyTorch build cannot see CUDA.\n"
                "Your machine has an NVIDIA GPU, so this usually means you installed a CPU-only torch.\n"
                "Install a CUDA-enabled PyTorch build in this interpreter, then rerun with --device 0.\n"
                "Quick check:\n"
                "python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\""
            )
        return requested
    return "0" if torch_module.cuda.is_available() else "cpu"


def apply_safe_defaults(args: argparse.Namespace, runtime_device: str) -> None:
    if runtime_device != "cpu":
        return

    adjustments: list[str] = []

    if args.model == "yolov8s.pt":
        args.model = "yolov8n.pt"
        adjustments.append("model->yolov8n.pt")
    if args.epochs > 30:
        args.epochs = 30
        adjustments.append("epochs->30")
    if args.imgsz > 640:
        args.imgsz = 640
        adjustments.append("imgsz->640")
    if args.batch > 2:
        args.batch = 2
        adjustments.append("batch->2")
    if args.workers != 0:
        args.workers = 0
        adjustments.append("workers->0")
    if args.mosaic > 0.5:
        args.mosaic = 0.5
        adjustments.append("mosaic->0.5")
    if args.mixup != 0.0:
        args.mixup = 0.0
        adjustments.append("mixup->0.0")
    if args.copy_paste != 0.0:
        args.copy_paste = 0.0
        adjustments.append("copy_paste->0.0")
    if args.scale > 0.35:
        args.scale = 0.35
        adjustments.append("scale->0.35")

    if adjustments:
        print("CPU-safe mode enabled:", ", ".join(adjustments))


if __name__ == "__main__":
    main()
