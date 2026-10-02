"""Font loading with sensible fallbacks, so the code runs on any machine."""
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_CANDIDATES = [
    Path(__file__).parent / "assets" / "font.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    Path("/Library/Fonts/Times New Roman Bold.ttf"),
    Path("C:/Windows/Fonts/timesbd.ttf"),
]


@lru_cache(maxsize=64)
def get_font(size: int, path: str | None = None) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, size)
    for p in _CANDIDATES:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)
