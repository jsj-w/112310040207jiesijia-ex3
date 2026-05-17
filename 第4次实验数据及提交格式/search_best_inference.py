from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

from traffic_sign_utils import (
    Detection,
    compute_map50,
    find_latest_best_weight,
    iter_image_paths,
    save_json,
    weighted_boxes_fusion,
    write_submission_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for the best validation-time inference settings for mAP@0.5.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more YOLO weight files. Multiple models will be fused.",
    )
    parser.add_argument("--image-dir", default="val/images", help="Validation image directory.")
    parser.add_argument("--label-dir", default="val/labels", help="Validation label directory.")
    parser.add_argument("--output-dir", default="runs/tuning", help="Directory for tuning artifacts.")
    parser.add_argument("--imgsz", type=int, default=640, help="Prediction image size.")
    parser.add_argument("--batch", type=int, default=4, help="Prediction batch size.")
    parser.add_argument("--device", default="", help="Prediction device, for example 0 or cpu.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per image.")
    parser.add_argument("--conf-values", default="0.001,0.003,0.005,0.01,0.02", help="Comma-separated confidence thresholds.")
    parser.add_argument("--iou-values", default="0.45,0.5,0.55,0.6,0.65", help="Comma-separated NMS IoU thresholds.")
    parser.add_argument(
        "--augment-values",
        default="false",
        help="Comma-separated boolean values that control Ultralytics test-time augmentation.",
    )
    parser.add_argument(
        "--fusion",
        choices=("wbf", "concat"),
        default="wbf",
        help="Use weighted boxes fusion or keep raw concatenated predictions.",
    )
    parser.add_argument("--save-best-csv", action="store_true", help="Write the best validation prediction CSV.")
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

    resolved_model_paths = resolve_model_paths(project_root=project_root, requested_models=args.models)
    image_dir = (project_root / args.image_dir).resolve() if not Path(args.image_dir).is_absolute() else Path(args.image_dir).resolve()
    label_dir = (project_root / args.label_dir).resolve() if not Path(args.label_dir).is_absolute() else Path(args.label_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    from traffic_sign_utils import read_yolo_annotations

    ground_truths = read_yolo_annotations(label_dir=label_dir, image_dir=image_dir)
    models = [YOLO(str(model_path)) for model_path in resolved_model_paths]
    if not args.device:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        if args.imgsz > 640:
            args.imgsz = 640
        if args.batch > 4:
            args.batch = 4
    conf_values = parse_float_list(args.conf_values)
    iou_values = parse_float_list(args.iou_values)
    augment_values = parse_bool_list(args.augment_values)
    total_runs = len(conf_values) * len(iou_values) * len(augment_values)

    best_record: dict[str, object] | None = None
    best_predictions: list[Detection] = []
    records: list[dict[str, object]] = []

    print(f"Using models: {[str(model_path) for model_path in resolved_model_paths]}")
    print(f"Validation images: {len(image_paths)}")
    print(f"Search combinations: {total_runs}")

    for run_index, (conf_threshold, iou_threshold, augment) in enumerate(
        itertools.product(conf_values, iou_values, augment_values),
        start=1,
    ):
        print(
            f"[{run_index}/{total_runs}] "
            f"conf={conf_threshold}, iou={iou_threshold}, augment={augment} -> running inference..."
        )
        started_at = time.perf_counter()
        predictions = predict_dataset(
            models=models,
            image_paths=image_paths,
            conf=conf_threshold,
            iou=iou_threshold,
            augment=augment,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            max_det=args.max_det,
            fusion=args.fusion,
        )
        scores = compute_map50(ground_truths=ground_truths, predictions=predictions)
        elapsed_seconds = time.perf_counter() - started_at
        record = {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "augment": augment,
            "fusion": args.fusion,
            "map50_present_classes": scores["map50_present_classes"],
            "map50_all_classes": scores["map50_all_classes"],
            "elapsed_seconds": elapsed_seconds,
        }
        records.append(record)
        print(record)

        if best_record is None or record["map50_present_classes"] > best_record["map50_present_classes"]:
            best_record = record
            best_predictions = predictions

    if best_record is None:
        raise RuntimeError("No tuning runs were executed.")

    best_record["models"] = [str(model_path) for model_path in resolved_model_paths]
    best_record["imgsz"] = args.imgsz
    best_record["batch"] = args.batch
    best_record["device"] = args.device
    best_record["max_det"] = args.max_det

    save_json(
        {
            "best": best_record,
            "all_runs": records,
        },
        output_dir / "tuning_summary.json",
    )

    if args.save_best_csv:
        write_submission_csv(best_predictions, output_dir / "best_val_predictions.csv")

    print("Best validation setting:")
    print(best_record)


def resolve_model_paths(project_root: Path, requested_models: list[str] | None) -> list[Path]:
    if requested_models:
        return [Path(model_path).resolve() for model_path in requested_models]

    latest_weight = find_latest_best_weight(project_root)
    print(f"Auto-detected latest model: {latest_weight}")
    return [latest_weight]


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
            # input chunk to preserve correct validation image_ids.
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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_list(value: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    parsed: list[bool] = []
    for item in value.split(","):
        key = item.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported boolean value: {item}")
        parsed.append(mapping[key])
    return parsed


def chunked_paths(image_paths: list[Path], chunk_size: int) -> list[list[Path]]:
    safe_chunk_size = max(1, chunk_size)
    return [
        image_paths[index:index + safe_chunk_size]
        for index in range(0, len(image_paths), safe_chunk_size)
    ]


if __name__ == "__main__":
    main()
