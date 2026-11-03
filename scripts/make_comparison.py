import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

root = Path(__file__).resolve().parents[1]
folder = root / "results" / "2026-11-03"
record = json.loads((folder / "results.json").read_text(encoding="utf-8"))
scores = {row["prompt"]: row for row in record["results"]}
source = Image.open(root / "external" / "Grad-Eclip" / record["image"]).convert("RGB")
source = ImageOps.fit(source, (224, 224), method=Image.Resampling.BICUBIC)

panels = [("Input image", source, "")]
for prompt in ("dog", "car"):
    row = scores[prompt]
    picture = Image.open(folder / row["heatmap_file"]).convert("RGB")
    score = f'cosine {row["original_cosine"]:.4f} -> {row["single_head_cosine"]:.4f}'
    panels.append((f"Grad-ECLIP: {prompt}", picture, score))

font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 19)
small = ImageFont.truetype(font_path, 15)
panel_width, margin, gap, height = 280, 24, 20, 394
board = Image.new("RGB", (margin * 2 + panel_width * 3 + gap * 2, height), "white")
draw = ImageDraw.Draw(board)
for index, (title, picture, score) in enumerate(panels):
    x = margin + index * (panel_width + gap)
    draw.text((x, 18), title, fill="black", font=font)
    board.paste(picture.resize((panel_width, panel_width), Image.Resampling.BICUBIC), (x, 52))
    if score:
        draw.text((x, 342), score, fill="#333333", font=small)
draw.text(
    (margin, 371),
    "Each heat map normalized separately for display. One-image pilot, not a faithfulness test.",
    fill="#555555",
    font=small,
)
board.save(folder / "comparison.png")
print(folder / "comparison.png")
