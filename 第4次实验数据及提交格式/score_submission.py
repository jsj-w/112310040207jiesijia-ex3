from __future__ import annotations

import argparse
from pathlib import Path

from traffic_sign_utils import compute_map50, read_submission_csv, read_yolo_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a prediction CSV against YOLO labels using mAP@0.5.",
    )
    parser.add_argument("--csv", required=True, help="Prediction CSV to evaluate.")
    parser.add_argument("--label-dir", default="val/labels", help="YOLO label directory.")
    parser.add_argument("--image-dir", default="val/images", help="Image directory used to resolve image_id names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    label_dir = Path(args.label_dir).resolve()
    image_dir = Path(args.image_dir).resolve()

    predictions = read_submission_csv(csv_path)
    ground_truths = read_yolo_annotations(label_dir=label_dir, image_dir=image_dir)
    scores = compute_map50(ground_truths=ground_truths, predictions=predictions)

    print(f"CSV: {csv_path}")
    print(f"mAP@0.5 (present classes): {scores['map50_present_classes']:.6f}")
    print(f"mAP@0.5 (all classes):     {scores['map50_all_classes']:.6f}")


if __name__ == "__main__":
    main()
