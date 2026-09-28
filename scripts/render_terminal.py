"""Render lines from a results/ log as a terminal-style PNG for the blog.

Only draws text it is given, which should come straight from a log in results/, so every
number in a screenshot traces back to a file.

    python scripts/render_terminal.py OUT.png "title" "$ command" < lines.txt
"""

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM, PROMPT = (24, 26, 33), (220, 223, 228), (130, 136, 150), (122, 162, 247)
GOOD, BAD, BAR = (158, 206, 106), (247, 118, 142), (40, 43, 54)
FONT = "C:/Windows/Fonts/CascadiaMono.ttf"


def color_for(line: str):
    if re.search(r"\bFAIL\b|COMPILE FAILED|✗|cosine vs CPU 0\.", line):
        return BAD
    if re.search(r"\bPASS\b|✓|cosine vs CPU 1\.0", line):
        return GOOD
    if line.startswith("#"):
        return DIM
    return FG


def render(out: Path, title: str, command: str, lines: list[str], size: int = 18) -> None:
    font = ImageFont.truetype(FONT, size)
    char_w = font.getbbox("M")[2]
    line_h = int(size * 1.45)
    pad, bar_h = 24, 38
    width = int(max(len(l) for l in lines + [command, title]) * char_w + 2 * pad)
    height = bar_h + pad + line_h * (len(lines) + 1) + pad
    img = Image.new("RGB", (max(width, 700), height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, bar_h], fill=BAR)
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
    d.text((100, 9), title, font=ImageFont.truetype(FONT, 15), fill=DIM)
    y = bar_h + pad
    d.text((pad, y), command, font=font, fill=PROMPT)
    for line in lines:
        y += line_h
        d.text((pad, y), line, font=font, fill=color_for(line))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


if __name__ == "__main__":
    out, title, command = sys.argv[1], sys.argv[2], sys.argv[3]
    render(Path(out), title, command, [l.rstrip("\n") for l in sys.stdin])
