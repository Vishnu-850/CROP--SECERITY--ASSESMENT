import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "Disease Detection"
    / "Dataset"
    / "Disease Detection Dataset"
    / "train"
    / "Tomato___Bacterial_spot"
)


def resolve_files(dataset_dir, limit=None):
    files = sorted(
        [
            str(path)
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No image files found in dataset directory: {dataset_dir}")
    return files


def severity_param(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])

    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_brown, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No leaf contour found in image: {img_path}")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    leaf_contour = contours[0]

    inten_yellow = []
    inten_brown = []

    for cnt in contours_yellow:
        mask = np.zeros_like(yellow_mask)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        intensity = cv2.mean(hsv_img, mask=mask)[2]
        inten_yellow = np.append(inten_yellow, intensity)
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    for cnt in contours_brown:
        mask = np.zeros_like(brown_mask)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        intensity = cv2.mean(hsv_img, mask=mask)[2]
        inten_brown = np.append(inten_brown, intensity)
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    area_damaged1 = 0
    area_damaged2 = 0

    for cnt in contours_yellow:
        area_damaged1 += cv2.contourArea(cnt)

    for cnt in contours_brown:
        area_damaged2 += cv2.contourArea(cnt)

    area_damaged = area_damaged1 + area_damaged2
    leaf_area = cv2.contourArea(leaf_contour)

    return leaf_area, area_damaged, inten_brown, inten_yellow


def main():
    parser = argparse.ArgumentParser(description="Compute disease severity index for tomato leaf images.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Directory containing the training images")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of number of files to process for a quick run")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("severity_plot.png"), help="Path for the generated chart")
    args = parser.parse_args()

    files = resolve_files(args.dataset_dir, args.limit)

    leaf_area = []
    area_damaged = []
    intensity_yellow = []
    intensity_brown = []

    for image_path in files:
        severity = severity_param(image_path)
        leaf_area = np.append(leaf_area, severity[0])
        area_damaged = np.append(area_damaged, severity[1])
        if len(severity[2]) == 0:
            intensity_brown = np.append(intensity_brown, 0)
        else:
            intensity_brown = np.append(intensity_brown, np.max(severity[2]))
        if len(severity[3]) == 0:
            intensity_yellow = np.append(intensity_yellow, 0)
        else:
            intensity_yellow = np.append(intensity_yellow, np.max(severity[3]))

    severity_index = []
    for i in range(len(files)):
        w1 = 0.17
        w2 = 0.05
        w3 = 0.78
        damaged = area_damaged[i]
        brown_intensity = intensity_brown[i]
        yellow_intensity = intensity_yellow[i]

        if damaged == 0 and brown_intensity == 0 and yellow_intensity == 0:
            severity_index = np.append(severity_index, 0)
        else:
            severity_index = np.append(
                severity_index,
                (w1 * damaged + w2 * brown_intensity + w3 * yellow_intensity) / (damaged + brown_intensity + yellow_intensity),
            )

    severity = []
    for value in severity_index:
        if value == 0:
            severity.append(0)
        elif 0 < value <= 0.19:
            severity.append(1)
        elif 0.19 < value <= 0.39:
            severity.append(2)
        elif 0.39 < value <= 0.59:
            severity.append(3)
        elif 0.59 < value <= 0.79:
            severity.append(4)
        elif 0.79 < value <= 1:
            severity.append(5)
        else:
            severity.append(0)

    severity = np.asarray(severity, dtype=int)
    value_counts = np.bincount(severity)

    x = np.arange(len(value_counts))
    plt.bar(x, value_counts)
    plt.xticks(x)
    plt.xlabel("No. of Samples")
    plt.ylabel("Severity")
    plt.title("Severity using Image Processing")
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Processed {len(files)} images; plot saved to {args.output}")


if __name__ == "__main__":
    main()