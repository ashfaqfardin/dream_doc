"""Render recovered placement rectangles over their corresponding base images."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from placement_masks import extract_rectangles


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / "BaseImages"
MASK_DIR = HERE / "PlacementMasks"
PROMPTS = HERE / "e5_prompts.json"
OUTPUT_DIR = HERE / "PlacementMaskOverlays"


def label_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 0), width=6)
    text_box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    label_y = max(0, y0 - text_h - 12)
    draw.rectangle((x0, label_y, min(x1, x0 + text_w + 16), y0), fill=(255, 0, 0))
    draw.text((x0 + 8, label_y + 4), text, fill=(255, 255, 255), font=font, stroke_width=1)


def main() -> None:
    with PROMPTS.open(encoding="utf-8") as handle:
        cases = json.load(handle)["prompts"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=20)
    overlays = []
    for case in cases:
        case_id = int(case["id"])
        base_path = BASE_DIR / f"base_{case_id:03d}.png"
        mask_path = MASK_DIR / f"base_{case_id:03d}.png"
        if not base_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair: {base_path}, {mask_path}")

        image = Image.open(base_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        rectangles = extract_rectangles(mask_path)
        if len(rectangles) != len(case["objects"]):
            raise ValueError(f"Case {case_id}: rectangle/object count mismatch")

        for rectangle, item in zip(rectangles, case["objects"]):
            label_box(
                draw,
                rectangle.box,
                f"{rectangle.object_index}: {item['name']} [{rectangle.label}]",
                font,
            )

        target = OUTPUT_DIR / f"base_{case_id:03d}_placement_overlay.png"
        image.save(target)
        overlays.append(image.copy())

    thumb_size = (384, 384)
    sheet = Image.new("RGB", (thumb_size[0] * 5, thumb_size[1] * 2), "white")
    for index, overlay in enumerate(overlays):
        thumb = overlay.copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((index % 5) * thumb_size[0], (index // 5) * thumb_size[1]))
    sheet.save(OUTPUT_DIR / "all_placement_overlays.png")
    print(f"Saved {len(overlays)} overlays to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
