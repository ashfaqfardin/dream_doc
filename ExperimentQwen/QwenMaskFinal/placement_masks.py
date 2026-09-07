"""Utilities for reading the manually drawn, depth-aware placement masks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


LABEL_VALUES = (85, 170, 255)


@dataclass(frozen=True)
class PlacementRectangle:
    """An inclusive-exclusive image rectangle assigned to one object."""

    object_index: int
    label: int
    box: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


def quantize_labels(mask: Image.Image) -> np.ndarray:
    """Snap anti-aliased grayscale pixels to 0, 85, 170, or 255."""

    gray = np.asarray(mask.convert("L"), dtype=np.int16)
    palette = np.asarray((0, *LABEL_VALUES), dtype=np.int16)
    nearest = np.abs(gray[..., None] - palette).argmin(axis=-1)
    return palette[nearest].astype(np.uint8)


def _largest_component(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates of the largest 4-connected foreground component."""

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best_y = np.empty(0, dtype=np.int32)
    best_x = np.empty(0, dtype=np.int32)

    for start_y, start_x in zip(*np.nonzero(binary & ~visited)):
        if visited[start_y, start_x]:
            continue
        queue = deque(((int(start_y), int(start_x)),))
        visited[start_y, start_x] = True
        component_y: list[int] = []
        component_x: list[int] = []
        while queue:
            y, x = queue.popleft()
            component_y.append(y)
            component_x.append(x)
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and binary[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(component_x) > len(best_x):
            best_y = np.asarray(component_y, dtype=np.int32)
            best_x = np.asarray(component_x, dtype=np.int32)
    return best_y, best_x


def extract_rectangles(mask_path: str | Path) -> list[PlacementRectangle]:
    """Recover full rectangles from labels, including depth-wise overlaps.

    Each label's visible pixels are reduced to their axis-aligned bounding box.
    The complete box is then used for placement, so a foreground rectangle
    overwriting part of a background rectangle does not create an irregular
    placement mask.
    """

    path = Path(mask_path)
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.int16).copy()

    rectangles = []
    for object_index, label in enumerate(LABEL_VALUES, start=1):
        # Use only the solid interior color here. Nearest-palette quantization
        # is unsuitable for geometry: an anti-aliased transition from black to
        # 170 or 255 passes through 85 and can falsely join distant boxes.
        solid = np.abs(gray - label) <= 2
        ys, xs = _largest_component(solid)
        if not len(xs):
            raise ValueError(f"{path} has no visible pixels for object {object_index} (label {label})")
        rectangles.append(
            PlacementRectangle(
                object_index=object_index,
                label=label,
                box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )
        )
    return rectangles


def rectangle_mask(size: tuple[int, int], rectangle: PlacementRectangle) -> Image.Image:
    """Rasterize one recovered placement rectangle as a binary mask."""

    width, height = size
    x0, y0, x1, y1 = rectangle.box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Rectangle {rectangle.box} lies outside image size {size}")
    array = np.zeros((height, width), dtype=np.uint8)
    array[y0:y1, x0:x1] = 255
    return Image.fromarray(array)
