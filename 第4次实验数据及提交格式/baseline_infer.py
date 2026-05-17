from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--test-dir", default="test/images", help="Directory of test images")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    local_config_dir = project_root / ".ultralytics"
    local_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_config_dir))
    from ultralytics import YOLO

    model = YOLO(args.model)
    image_paths = sorted(
        [p for p in Path(args.test_dir).iterdir() if p.is_file()]
    )

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        results = model.predict(source=[str(p) for p in image_paths], conf=args.conf, save=False, verbose=False)
        for result_index, result in enumerate(results):
            image_id = image_paths[result_index].name
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow(
                    {
                        "image_id": image_id,
                        "class_id": int(box.cls[0].item()),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": float(box.conf[0].item()),
                    }
                )


if __name__ == "__main__":
    main()
