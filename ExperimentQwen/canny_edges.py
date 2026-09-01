"""Convert all PNGs in object_photos to Canny edge images.

Output: ExperimentQwen/object_canny/<name>.png

Usage:
    pip install opencv-python
    python ExperimentQwen/canny_edges.py
"""

from pathlib import Path
import cv2

IN_DIR  = Path(__file__).parent / "object_photos"
OUT_DIR = Path(__file__).parent / "object_canny"

BLUR_KERNEL = 3   # Gaussian blur before Canny (odd number)
THRESHOLD1  = 50  # lower hysteresis threshold
THRESHOLD2  = 150 # upper hysteresis threshold


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(IN_DIR.glob("*.png"))
    print(f"Found {len(images)} PNG files\n")

    for index, src in enumerate(images, 1):
        dest = OUT_DIR / src.name
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[{index:>2}/{len(images)}] skip  {src.name}  (unreadable)")
            continue

        blurred = cv2.GaussianBlur(img, (BLUR_KERNEL, BLUR_KERNEL), 0)
        edges   = cv2.Canny(blurred, THRESHOLD1, THRESHOLD2)

        # White background, black edges (invert default black-bg output)
        edges = cv2.bitwise_not(edges)

        cv2.imwrite(str(dest), edges)
        print(f"[{index:>2}/{len(images)}] {src.name} -> {dest.name}")

    print(f"\nDone. Canny images saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
