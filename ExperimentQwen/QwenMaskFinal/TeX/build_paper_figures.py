"""Build the fixed-size qualitative figures used by the TeX chapters."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE.parent / "results"
QWEN = RESULTS / "QwenMaskFinal_04"
KONTEXT = RESULTS / "KontextMaskFinal_01"
SIZE = (1920, 800)
BG = "#FFFFFF"
INK = "#172033"
MUTED = "#586174"
BORDER = "#9AA3B2"
QWEN_COLOR = "#9A5B13"
KONTEXT_COLOR = "#13795B"
FONT_FILE = Path("C:/Windows/Fonts/arialbi.ttf")


def font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_FILE.is_file():
        raise FileNotFoundError(f"Arial Bold Italic was not found at {FONT_FILE}")
    return ImageFont.truetype(str(FONT_FILE), size)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, face, fill=INK):
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=face, fill=fill)


def image_card(source: Path, side: int, radius: int = 22, border: int = 3) -> Image.Image:
    image = Image.open(source).convert("RGB")
    width, height = image.size
    scale = max(side / width, side / height)
    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, side - 1, side - 1), radius=radius, fill=255)
    card = Image.new("RGB", (side, side), BG)
    card.paste(image, mask=mask)
    ImageDraw.Draw(card).rounded_rectangle(
        (border // 2, border // 2, side - 1 - border // 2, side - 1 - border // 2),
        radius=radius,
        outline=BORDER,
        width=border,
    )
    return card


def sequence(root: Path, case_id: int, backbone: str, output: Path, accent: str):
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    centered(draw, (960, 30), f"{backbone}: Sequential Object Insertion", font(38))
    steps = sorted((root / f"case_{case_id:03d}" / "steps").glob("*_final.png"))
    files = [root / f"case_{case_id:03d}" / "base.png", *steps]
    labels = ["Base scene", "Turn 1: Couch", "Turn 2: Floor lamp", "Turn 3: Potted plant"]
    side, gap = 400, 55
    start = (SIZE[0] - side * 4 - gap * 3) // 2
    for index, (path, label) in enumerate(zip(files, labels)):
        x = start + index * (side + gap)
        centered(draw, (x + side // 2, 105), label, font(25), accent if index else MUTED)
        canvas.paste(image_card(path, side, 26, 3), (x, 155))
        if index < 3:
            y = 355
            draw.line((x + side + 10, y, x + side + gap - 14, y), fill=INK, width=4)
            draw.polygon(
                [(x + side + gap - 14, y), (x + side + gap - 28, y - 9), (x + side + gap - 28, y + 9)],
                fill=INK,
            )
    centered(draw, (960, 704), "Each turn restores all pixels outside its interaction mask", font(23), MUTED)
    canvas.save(output, quality=96)


def all_cases(output: Path):
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    centered(draw, (960, 14), "Matched Qualitative Comparison Across Ten Scenes", font(34))
    row_specs = [
        ("Input", QWEN, "base.png", MUTED),
        ("Qwen", QWEN, "FINAL.png", QWEN_COLOR),
        ("Kontext", KONTEXT, "FINAL.png", KONTEXT_COLOR),
    ]
    side, gap, start_x = 176, 9, 80
    row_y = [120, 340, 560]
    for case_id in range(1, 11):
        x = start_x + (case_id - 1) * (side + gap)
        centered(draw, (x + side // 2, 75), f"Case {case_id}", font(20))
    for y, (label, root, name, color) in zip(row_y, row_specs):
        draw.text((12, y + side // 2 - 12), label, font=font(20), fill=color)
        for case_id in range(1, 11):
            x = start_x + (case_id - 1) * (side + gap)
            path = root / f"case_{case_id:03d}" / name
            canvas.paste(image_card(path, side, 15, 2), (x, y))
    canvas.save(output, quality=96)


def challenging(output: Path):
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    centered(draw, (960, 18), "Representative Placement Challenges", font(36))
    cases = [(2, "Appliances"), (4, "Floor contact"), (7, "Thin objects"), (8, "Depth relation"), (10, "Small identity")]
    side, gap = 290, 72
    start = (1920 - 5 * side - 4 * gap) // 2
    for j, (case_id, title) in enumerate(cases):
        x = start + j * (side + gap)
        centered(draw, (x + side // 2, 76), f"Case {case_id}: {title}", font(19))
        canvas.paste(image_card(QWEN / f"case_{case_id:03d}" / "FINAL.png", side, 22, 3), (x, 125))
        canvas.paste(image_card(KONTEXT / f"case_{case_id:03d}" / "FINAL.png", side, 22, 3), (x, 455))
    draw.text((22, 250), "Qwen", font=font(21), fill=QWEN_COLOR)
    draw.text((12, 580), "Kontext", font=font(21), fill=KONTEXT_COLOR)
    canvas.save(output, quality=96)


def project_experiment_failures(output: Path):
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    centered(draw, (960, 16), "Failure Modes in the Project Experiment Series", font(36))
    source = FIGURES / "development_sources"
    items = [
        ("Mask-collage run 1", "Global texture corruption", source / "global_edit_corruption.png"),
        ("Mask-collage run 2", "Duplicated or framed content", source / "reference_card_artifact.png"),
        ("Kontext attention concat.", "Identity dilution as context grows", source / "attention_concat_identity_dilution.png"),
        ("E14 correspondence", "Localized response, scene takeover", source / "correspondence_scene_takeover.png"),
    ]
    side, gap = 390, 70
    start = (1920 - side * 4 - gap * 3) // 2
    for index, (title, finding, path) in enumerate(items):
        x = start + index * (side + gap)
        centered(draw, (x + side // 2, 84), title, font(23), INK)
        canvas.paste(image_card(path, side, 25, 3), (x, 135))
        draw.rounded_rectangle((x, 555, x + side, 670), radius=18, fill="#F5F6F8", outline=BORDER, width=2)
        centered(draw, (x + side // 2, 580), finding, font(18), MUTED)
    centered(
        draw,
        (960, 725),
        "Project finding: reliable preservation requires explicit local ownership",
        font(23),
        KONTEXT_COLOR,
    )
    canvas.save(output, quality=96)


def rounded_box(draw, box, fill, outline, title, detail="", title_size=22):
    draw.rounded_rectangle(box, radius=20, fill=fill, outline=outline, width=3)
    cx = (box[0] + box[2]) // 2
    centered(draw, (cx, box[1] + 24), title, font(title_size), INK)
    if detail:
        centered(draw, (cx, box[1] + 61), detail, font(16), MUTED)


def arrow(draw, start, end):
    draw.line((*start, *end), fill=INK, width=4)
    draw.polygon([(end[0], end[1]), (end[0] - 13, end[1] - 8), (end[0] - 13, end[1] + 8)], fill=INK)


def pipeline(output: Path):
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    centered(draw, (960, 20), "Training-Free Local Collage Harmonization", font(38))
    centered(draw, (960, 68), "Reference identity + local generation + hard non-disturbance", font(20), MUTED)

    # Inputs
    inputs = [("Current scene", "I(i-1)"), ("Placement", "L(i)"), ("Reference", "O(i)")]
    for j, (name, symbol) in enumerate(inputs):
        rounded_box(draw, (35, 150 + j * 170, 275, 270 + j * 170), "#EAF2FF", "#356AE6", name, symbol)

    # Three major stages
    draw.rounded_rectangle((340, 120, 805, 655), radius=26, fill="#FFF8EC", outline="#D98919", width=3)
    centered(draw, (573, 142), "1. Deterministic Preparation", font(26), QWEN_COLOR)
    rounded_box(draw, (390, 220, 580, 330), "#FFF1D8", "#D98919", "RMBG cutout", "RGB + alpha", 20)
    rounded_box(draw, (610, 220, 755, 330), "#FFF1D8", "#D98919", "Fit", "Scale + anchor", 20)
    rounded_box(draw, (390, 405, 580, 520), "#FFF1D8", "#D98919", "Collage", "C(i)", 20)
    rounded_box(draw, (610, 405, 755, 520), "#FFF1D8", "#D98919", "Mask", "M(i)", 20)

    draw.rounded_rectangle((855, 120, 1275, 655), radius=26, fill="#FAF3FF", outline="#7D35D8", width=3)
    centered(draw, (1065, 142), "2. Local Harmonization", font(26), "#6D28B8")
    rounded_box(draw, (920, 230, 1210, 350), "#F2E5FF", "#7D35D8", "Square context crop", "Object + local scene", 21)
    rounded_box(draw, (920, 425, 1210, 555), "#EEDFFF", "#7D35D8", "Frozen editor", "Qwen or Kontext", 23)

    draw.rounded_rectangle((1325, 120, 1880, 655), radius=26, fill="#F1FBF5", outline="#16805A", width=3)
    centered(draw, (1602, 142), "3. Hard Spatial Preservation", font(26), KONTEXT_COLOR)
    rounded_box(draw, (1395, 220, 1810, 335), "#E3F6EB", "#16805A", "Inward mask feather", "Zero outside hard support", 22)
    rounded_box(draw, (1395, 390, 1810, 520), "#E3F6EB", "#16805A", "Masked RGB composition", "I(i) = (1-M)I(i-1) + M I-hat(i)", 22)
    rounded_box(draw, (1480, 570, 1725, 685), "#D9F3E4", "#16805A", "Edited scene", "I(i)", 22)

    arrow(draw, (275, 210), (390, 462)); arrow(draw, (275, 550), (390, 275))
    arrow(draw, (275, 380), (610, 275)); arrow(draw, (580, 275), (610, 275))
    arrow(draw, (500, 330), (500, 405)); arrow(draw, (680, 330), (680, 405))
    arrow(draw, (755, 462), (920, 290)); arrow(draw, (580, 462), (920, 290))
    arrow(draw, (1065, 350), (1065, 425)); arrow(draw, (1210, 490), (1395, 455))
    arrow(draw, (755, 462), (1395, 277)); arrow(draw, (1602, 335), (1602, 390)); arrow(draw, (1602, 520), (1602, 570))

    # Evaluation strip
    draw.rounded_rectangle((340, 700, 1880, 780), radius=18, fill="#F5F6F8", outline=BORDER, width=2)
    labels = ["DINO identity", "Colour retention", "Edge structure", "Spatial localization", "Non-disturbance", "Cross-turn stability"]
    for j, label in enumerate(labels):
        centered(draw, (470 + j * 255, 728), label, font(17), INK)
    canvas.save(output, quality=96)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sequence(QWEN, 1, "Qwen-Image-Edit-2509", FIGURES / "qwen_case001_sequence.png", QWEN_COLOR)
    sequence(KONTEXT, 1, "FLUX.1 Kontext Dev", FIGURES / "kontext_case001_sequence.png", KONTEXT_COLOR)
    all_cases(FIGURES / "all_cases_comparison.png")
    challenging(FIGURES / "challenging_cases.png")
    project_experiment_failures(FIGURES / "project_experiment_failures.png")
    pipeline(FIGURES / "pipeline_figure.png")
    for path in sorted(FIGURES.glob("*.png")):
        with Image.open(path) as image:
            assert image.size == SIZE, (path, image.size)
            image.convert("RGB").save(path.with_suffix(".pdf"), "PDF", resolution=300.0)
            print(f"{path.name} -> {path.with_suffix('.pdf').name}: {image.size} at 300 DPI")


if __name__ == "__main__":
    main()
