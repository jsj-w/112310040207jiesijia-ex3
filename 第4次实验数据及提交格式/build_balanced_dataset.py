from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from traffic_sign_utils import CLASS_NAMES, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO training list that oversamples rare traffic sign classes.",
    )
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument(
        "--output-dir",
        default="generated",
        help="Directory that will receive balanced_train.txt and balanced_data.yaml",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=0.55,
        help="Rare classes are repeated until they reach this fraction of the most common class",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=8,
        help="Maximum number of times one image can appear in the balanced train list",
    )
    parser.add_argument(
        "--min-target-count",
        type=int,
        default=120,
        help="Minimum target box count for each class before oversampling stops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    train_image_dir = root / "train" / "images"
    train_label_dir = root / "train" / "labels"
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_box_counts: Counter[int] = Counter()
    image_classes: dict[Path, set[int]] = {}

    for label_path in sorted(train_label_dir.glob("*.txt")):
        image_path = train_image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            fallback_matches = sorted(train_image_dir.glob(f"{label_path.stem}.*"))
            if not fallback_matches:
                continue
            image_path = fallback_matches[0]

        classes_in_image: set[int] = set()
        content = label_path.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                class_id = int(line.split()[0])
                class_box_counts[class_id] += 1
                classes_in_image.add(class_id)
        image_classes[image_path.resolve()] = classes_in_image

    if not class_box_counts:
        raise RuntimeError("No labels were found in train/labels.")

    max_count = max(class_box_counts.values())
    target_count = max(int(max_count * args.target_fraction), args.min_target_count)

    repeat_by_class: dict[int, int] = {}
    for class_id in range(len(CLASS_NAMES)):
        count = class_box_counts.get(class_id, 0)
        if count <= 0:
            repeat_by_class[class_id] = 1
            continue
        repeat_factor = math.ceil(target_count / count)
        repeat_by_class[class_id] = min(max(repeat_factor, 1), args.max_repeat)

    repeated_images: list[str] = []
    repeat_histogram: Counter[int] = Counter()

    for image_path, classes_in_image in sorted(image_classes.items(), key=lambda item: item[0].name):
        if classes_in_image:
            repeat_factor = max(repeat_by_class[class_id] for class_id in classes_in_image)
        else:
            repeat_factor = 1
        repeat_histogram[repeat_factor] += 1
        relative_image_path = image_path.relative_to(root).as_posix()
        repeated_images.extend([relative_image_path] * repeat_factor)

    train_list_path = output_dir / "balanced_train.txt"
    train_list_path.write_text("\n".join(repeated_images) + "\n", encoding="utf-8")

    data_yaml_path = output_dir / "balanced_data.yaml"
    data_yaml_path.write_text(
        build_data_yaml(output_dir=output_dir),
        encoding="utf-8",
    )

    summary = {
        "root": str(root),
        "target_count": target_count,
        "class_box_counts": {
            CLASS_NAMES[class_id]: class_box_counts.get(class_id, 0)
            for class_id in range(len(CLASS_NAMES))
        },
        "repeat_by_class": {
            CLASS_NAMES[class_id]: repeat_by_class[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
        "repeat_histogram": dict(sorted(repeat_histogram.items())),
        "original_train_images": len(image_classes),
        "balanced_train_rows": len(repeated_images),
    }
    save_json(summary, output_dir / "balanced_summary.json")

    print(f"Balanced training list written to: {train_list_path}")
    print(f"Balanced data config written to:   {data_yaml_path}")
    print(f"Original train images: {len(image_classes)}")
    print(f"Balanced train rows:   {len(repeated_images)}")


def build_data_yaml(output_dir: Path) -> str:
    lines = [
        "path: .",
        f"train: {output_dir.name}/balanced_train.txt",
        "val: val/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
