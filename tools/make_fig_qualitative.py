"""Build the qualitative figure from the k=1 run's own visualisation PNGs.

Each run directory contains a 2x2 panel of 672x672 images: Original (top left),
Ground Truth (top right), Anomaly Map (bottom left), Overlay (bottom right).
This script reads the saved files under k1_raw/ and keeps only Original and
Overlay for three categories that matter in the report.
"""
import pathlib

from PIL import Image, ImageDraw, ImageFont

WS = pathlib.Path(__file__).resolve().parent.parent
FONT = pathlib.Path("C:/Users/michael/.workbuddy-ai/binaries/python/envs/default/"
                    "Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")
AUROC = {"bottle": "1.0000", "cable": "0.9196", "capsule": "0.6745"}
CATS = list(AUROC)

PANEL = 672
GAP = 8
HEAD = 46

font = ImageFont.truetype(str(FONT), 30)
canvas = Image.new(
    "RGB", (2 * PANEL + 3 * GAP, len(CATS) * (PANEL + HEAD + GAP) + GAP), "white"
)
draw = ImageDraw.Draw(canvas)

for row, cat in enumerate(CATS):
    hits = sorted((WS / "k1_raw" / cat).rglob("visualizations/*_example_0.png"))
    assert len(hits) == 1, f"expected one visualisation for {cat}, found {len(hits)}"
    img = Image.open(hits[0]).convert("RGB")
    assert img.size == (2 * PANEL, 2 * PANEL), img.size

    y = GAP + row * (PANEL + HEAD + GAP)
    draw.text((GAP + 6, y + 5), f"{cat}   |   image AUROC {AUROC[cat]}   |   k = 1",
              fill="black", font=font)

    canvas.paste(img.crop((0, 0, PANEL, PANEL)), (GAP, y + HEAD))
    canvas.paste(img.crop((PANEL, PANEL, 2 * PANEL, 2 * PANEL)),
                 (2 * GAP + PANEL, y + HEAD))

out = WS / "figures" / "fig_qualitative_k1.png"
canvas.save(out, optimize=True)
print(f"wrote {out} {canvas.size} {out.stat().st_size} bytes")
