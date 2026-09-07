"""Utilities for reading the manually drawn, depth-aware placement masks."""
from __future__ import annotations

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


def extract_rectangles(mask_path: str | Path) -> list[PlacementRectangle]:
    """Recover full rectangles from labels, including depth-wise overlaps.

    Each label's visible pixels are reduced to their axis-aligned bounding box.
    The complete box is then used for placement, so a foreground rectangle
    overwriting part of a background rectangle does not create an irregular
    placement mask.
    """

    path = Path(mask_path)
    with Image.open(path) as image:
        labels = quantize_labels(image)

    rectangles = []
    for object_index, label in enumerate(LABEL_VALUES, start=1):
        ys, xs = np.nonzero(labels == label)
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
