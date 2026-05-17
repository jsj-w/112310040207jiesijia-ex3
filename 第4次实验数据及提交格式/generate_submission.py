from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from traffic_sign_utils import (
    Detection,
    find_latest_best_weight,
    find_latest_tuning_config,
    iter_image_paths,
    weighted_boxes_fusion,
    write_submission_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final submission.csv from one or more trained YOLO models.",
    )
    parser.add_argument("--models", nargs="+", default=None, help="One or more YOLO weight files.")
    parser.add_argument("--image-dir", default="test/images", help="Image directory to run inference on.")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Prediction image size.")
    parser.add_argument("--batch", type=int, default=4, help="Prediction batch size.")
    parser.add_argument("--device", default="", help="Prediction device, for example 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.003, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.55, help="NMS or fusion IoU threshold.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per image.")
    parser.add_argument("--augment", action="store_true", help="Enable Ultralytics test-time augmentation.")
    parser.add_argument(
        "--fusion",
        choices=("wbf", "concat"),
        default="wbf",
        help="Use weighted boxes fusion or keep raw concatenated predictions.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional tuning_summary.json or a JSON file with best inference settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    local_config_dir = project_root / ".ultralytics"
    local_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_config_dir))
    try:
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install dependencies first, for example:\n"
            "pip install -r requirements.txt"
        ) from exc

    args.models = resolve_model_paths(project_root=project_root, requested_models=args.models)
    if args.config:
        apply_config_file(args)
    elif has_default_tuning_config(project_root):
        args.config = str(find_latest_tuning_config(project_root))
        apply_config_file(args)
        print(f"Auto-detected tuning config: {args.config}")
    if not args.device:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        if args.imgsz > 640:
            args.imgsz = 640
        if args.batch > 4:
            args.batch = 4

    image_dir = (project_root / args.image_dir).resolve() if not Path(args.image_dir).is_absolute() else Path(args.image_dir).resolve()
    output_path = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    image_paths = iter_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    models = [YOLO(str(model_path)) for model_path in args.models]
    predictions = predict_dataset(
        models=models,
        image_paths=image_paths,
        conf=args.conf,
        iou=args.iou,
        augment=args.augment,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        max_det=args.max_det,
        fusion=args.fusion,
    )
    write_submission_csv(predictions, output_path)
    print(f"Using models: {[str(model_path) for model_path in args.models]}")
    print(f"Submission written to: {output_path}")
    print(f"Rows: {len(predictions)}")


def apply_config_file(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    best = payload.get("best", payload)
    args.conf = float(best.get("conf", args.conf))
    args.iou = float(best.get("iou", args.iou))
    args.augment = bool(best.get("augment", args.augment))
    args.imgsz = int(best.get("imgsz", args.imgsz))
    args.batch = int(best.get("batch", args.batch))
    args.device = str(best.get("device", args.device))
    args.max_det = int(best.get("max_det", args.max_det))
    args.fusion = str(best.get("fusion", args.fusion))


def resolve_model_paths(project_root: Path, requested_models: list[str] | None) -> list[Path]:
    if requested_models:
        return [Path(model_path).resolve() for model_path in requested_models]

    latest_weight = find_latest_best_weight(project_root)
    print(f"Auto-detected latest model: {latest_weight}")
    return [latest_weight]


def has_default_tuning_config(project_root: Path) -> bool:
    try:
        find_latest_tuning_config(project_root)
    except FileNotFoundError:
        return False
    return True


def predict_dataset(
    models: list[object],
    image_paths: list[Path],
    conf: float,
    iou: float,
    augment: bool,
    imgsz: int,
    batch: int,
    device: str,
    max_det: int,
    fusion: str,
) -> list[Detection]:
    predictions_by_image: dict[str, list[Detection]] = {path.name: [] for path in image_paths}

    for model in models:
        for chunk_paths in chunked_paths(image_paths, batch):
            chunk_sources = [str(path) for path in chunk_paths]
            results = model.predict(
                source=chunk_sources,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                batch=min(batch, len(chunk_sources)),
                device=device,
                max_det=max_det,
                augment=augment,
                save=False,
                verbose=False,
            )
            # Ultralytics can replace `result.path` with batch-local names like
            # image0.jpg/image1.jpg, so keep the original file names from the
            # input chunk to preserve valid submission image_ids.
            for result_index, result in enumerate(results):
                image_id = chunk_paths[result_index].name
                image_predictions = predictions_by_image.setdefault(image_id, [])
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    image_predictions.append(
                        Detection(
                            image_id=image_id,
                            class_id=int(box.cls[0].item()),
                            x_center=float(x_center),
                            y_center=float(y_center),
                            width=float(width),
                            height=float(height),
                            confidence=float(box.conf[0].item()),
                        )
                    )

    final_predictions: list[Detection] = []
    for image_id in sorted(predictions_by_image):
        detections = predictions_by_image[image_id]
        if fusion == "wbf" and len(models) > 1:
            final_predictions.extend(weighted_boxes_fusion(detections, iou_threshold=iou))
        else:
            final_predictions.extend(sorted(detections, key=lambda item: -item.confidence))
    return final_predictions


def chunked_paths(image_paths: list[Path], chunk_size: int) -> list[list[Path]]:
    safe_chunk_size = max(1, chunk_size)
    return [
        image_paths[index:index + safe_chunk_size]
        for index in range(0, len(image_paths), safe_chunk_size)
    ]


if __name__ == "__main__":
    main()
