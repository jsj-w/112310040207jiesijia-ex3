from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES = [
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class Detection:
    image_id: str
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        x1 = max(0.0, self.x_center - half_w)
        y1 = max(0.0, self.y_center - half_h)
        x2 = min(1.0, self.x_center + half_w)
        y2 = min(1.0, self.y_center + half_h)
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(
        cls,
        image_id: str,
        class_id: int,
        box: tuple[float, float, float, float],
        confidence: float,
    ) -> "Detection":
        x1, y1, x2, y2 = box
        x1 = min(max(x1, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0)
        y2 = min(max(y2, 0.0), 1.0)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return cls(
            image_id=image_id,
            class_id=class_id,
            x_center=x1 + width / 2.0,
            y_center=y1 + height / 2.0,
            width=width,
            height=height,
            confidence=confidence,
        )


def iter_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_submission_csv(predictions: Iterable[Detection], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "class_id",
                "x_center",
                "y_center",
                "width",
                "height",
                "confidence",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(asdict(prediction))


def read_submission_csv(csv_path: Path) -> list[Detection]:
    predictions: list[Detection] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions.append(
                Detection(
                    image_id=row["image_id"],
                    class_id=int(row["class_id"]),
                    x_center=float(row["x_center"]),
                    y_center=float(row["y_center"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                    confidence=float(row["confidence"]),
                )
            )
    return predictions


def read_yolo_annotations(
    label_dir: Path,
    image_dir: Path | None = None,
) -> dict[str, list[Detection]]:
    targets: dict[str, list[Detection]] = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        image_id = resolve_image_id_from_label(label_path=label_path, image_dir=image_dir)
        detections: list[Detection] = []
        content = label_path.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                class_id, x_center, y_center, width, height = line.split()
                detections.append(
                    Detection(
                        image_id=image_id,
                        class_id=int(class_id),
                        x_center=float(x_center),
                        y_center=float(y_center),
                        width=float(width),
                        height=float(height),
                        confidence=1.0,
                    )
                )
        targets[image_id] = detections
    return targets


def resolve_image_id_from_label(label_path: Path, image_dir: Path | None) -> str:
    if image_dir is not None:
        exact_image_path = image_dir / f"{label_path.stem}.jpg"
        if exact_image_path.exists():
            return exact_image_path.name
        matches = sorted(image_dir.glob(f"{label_path.stem}.*"))
        if matches:
            return matches[0].name
    return label_path.stem


def group_by_image(predictions: Iterable[Detection]) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.image_id, []).append(prediction)
    return grouped


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_map50(
    ground_truths: dict[str, list[Detection]],
    predictions: Iterable[Detection],
    num_classes: int = len(CLASS_NAMES),
) -> dict[str, object]:
    predictions_by_class: dict[int, list[Detection]] = {class_id: [] for class_id in range(num_classes)}
    gt_by_class: dict[int, dict[str, list[Detection]]] = {class_id: {} for class_id in range(num_classes)}

    for image_id, detections in ground_truths.items():
        for detection in detections:
            gt_by_class[detection.class_id].setdefault(image_id, []).append(detection)

    for prediction in predictions:
        if 0 <= prediction.class_id < num_classes:
            predictions_by_class[prediction.class_id].append(prediction)

    ap_by_class: dict[int, float] = {}
    gt_count_by_class: dict[int, int] = {}

    for class_id in range(num_classes):
        gt_for_class = gt_by_class[class_id]
        gt_count = sum(len(items) for items in gt_for_class.values())
        gt_count_by_class[class_id] = gt_count
        predictions_for_class = sorted(
            predictions_by_class[class_id],
            key=lambda item: item.confidence,
            reverse=True,
        )

        if gt_count == 0:
            ap_by_class[class_id] = 0.0
            continue

        matched = {
            image_id: [False] * len(items)
            for image_id, items in gt_for_class.items()
        }
        true_positive: list[float] = []
        false_positive: list[float] = []

        for prediction in predictions_for_class:
            candidates = gt_for_class.get(prediction.image_id, [])
            best_iou = 0.0
            best_index = -1
            prediction_box = prediction.to_xyxy()

            for index, target in enumerate(candidates):
                if matched[prediction.image_id][index]:
                    continue
                overlap = iou_xyxy(prediction_box, target.to_xyxy())
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index

            if best_iou >= 0.5 and best_index >= 0:
                matched[prediction.image_id][best_index] = True
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)

        cumulative_tp: list[float] = []
        cumulative_fp: list[float] = []
        running_tp = 0.0
        running_fp = 0.0

        for tp_value, fp_value in zip(true_positive, false_positive):
            running_tp += tp_value
            running_fp += fp_value
            cumulative_tp.append(running_tp)
            cumulative_fp.append(running_fp)

        recalls = [tp / gt_count for tp in cumulative_tp]
        precisions = [
            tp / max(tp + fp, 1e-9)
            for tp, fp in zip(cumulative_tp, cumulative_fp)
        ]
        ap_by_class[class_id] = compute_average_precision(recalls, precisions)

    present_classes = [class_id for class_id, count in gt_count_by_class.items() if count > 0]
    present_class_map = (
        sum(ap_by_class[class_id] for class_id in present_classes) / len(present_classes)
        if present_classes
        else 0.0
    )
    all_class_map = sum(ap_by_class.values()) / num_classes if num_classes else 0.0

    return {
        "map50_present_classes": present_class_map,
        "map50_all_classes": all_class_map,
        "ap_by_class": ap_by_class,
        "gt_count_by_class": gt_count_by_class,
    }


def compute_average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    area = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            area += (mrec[index] - mrec[index - 1]) * mpre[index]
    return area


def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.55,
) -> list[Detection]:
    kept: list[Detection] = []
    for class_id in range(len(CLASS_NAMES)):
        class_detections = sorted(
            [detection for detection in detections if detection.class_id == class_id],
            key=lambda item: item.confidence,
            reverse=True,
        )
        selected: list[Detection] = []
        while class_detections:
            best = class_detections.pop(0)
            selected.append(best)
            best_box = best.to_xyxy()
            class_detections = [
                candidate
                for candidate in class_detections
                if iou_xyxy(best_box, candidate.to_xyxy()) < iou_threshold
            ]
        kept.extend(selected)
    return sorted(kept, key=lambda item: (item.image_id, -item.confidence, item.class_id))


def weighted_boxes_fusion(
    detections: list[Detection],
    iou_threshold: float = 0.55,
) -> list[Detection]:
    fused: list[Detection] = []
    by_image = group_by_image(detections)

    for image_id, image_detections in by_image.items():
        for class_id in range(len(CLASS_NAMES)):
            remaining = sorted(
                [item for item in image_detections if item.class_id == class_id],
                key=lambda item: item.confidence,
                reverse=True,
            )
            while remaining:
                anchor = remaining.pop(0)
                anchor_box = anchor.to_xyxy()
                cluster = [anchor]
                leftovers: list[Detection] = []

                for candidate in remaining:
                    if iou_xyxy(anchor_box, candidate.to_xyxy()) >= iou_threshold:
                        cluster.append(candidate)
                    else:
                        leftovers.append(candidate)
                remaining = leftovers

                score_sum = sum(item.confidence for item in cluster)
                if score_sum <= 0.0:
                    continue

                x1 = sum(item.to_xyxy()[0] * item.confidence for item in cluster) / score_sum
                y1 = sum(item.to_xyxy()[1] * item.confidence for item in cluster) / score_sum
                x2 = sum(item.to_xyxy()[2] * item.confidence for item in cluster) / score_sum
                y2 = sum(item.to_xyxy()[3] * item.confidence for item in cluster) / score_sum
                confidence = max(item.confidence for item in cluster)
                fused.append(
                    Detection.from_xyxy(
                        image_id=image_id,
                        class_id=class_id,
                        box=(x1, y1, x2, y2),
                        confidence=min(confidence, 1.0),
                    )
                )

    return sorted(fused, key=lambda item: (item.image_id, -item.confidence, item.class_id))


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def find_latest_best_weight(project_root: Path) -> Path:
    weight_paths = sorted(
        project_root.glob("runs/train/*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not weight_paths:
        raise FileNotFoundError(
            "No trained model was found under runs/train/*/weights/best.pt. "
            "Run train_detector.py first."
        )
    return weight_paths[0]


def find_latest_tuning_config(project_root: Path) -> Path:
    default_path = project_root / "runs" / "tuning" / "tuning_summary.json"
    if default_path.exists():
        return default_path

    config_paths = sorted(
        project_root.glob("runs/**/tuning_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not config_paths:
        raise FileNotFoundError(
            "No tuning_summary.json was found. Run search_best_inference.py first."
        )
    return config_paths[0]
