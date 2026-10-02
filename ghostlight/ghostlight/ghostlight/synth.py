"""Synthetic worn gravestones with known ground truth.

Used for two things:
  1. Developing and testing the pipeline without a real stone.
  2. The WornBench benchmark: measuring accuracy as erosion increases.

A stone is a height map (carved letters are V-shaped grooves), an albedo map
(stone colour plus lichen blotches) and a set of images rendered under known
raking lights, exactly like a real Ghostlight capture.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import get_font


@dataclass
class SyntheticStone:
    text_lines: list[str]
    height: np.ndarray          # HxW, carving depth in pixel units (negative = groove)
    albedo: np.ndarray          # HxW in [0, 1]
    stroke_mask: np.ndarray     # HxW bool, where the original carving was
    line_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # x0,y0,x1,y1
    wear: float = 0.0


def _fractal_noise(shape, rng, octaves=5, persistence=0.55):
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        small = rng.standard_normal((s, int(s * w / h) + 1)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    out /= total
    return (out - out.mean()) / (out.std() + 1e-8)


def render_text_mask(lines, size, font_size, line_gap=1.45, font_path=None):
    """Render centred text lines; returns mask (float 0..1) and per-line boxes."""
    w, h = size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    font = get_font(font_size, font_path)
    heights = []
    widths = []
    for ln in lines:
        x0, y0, x1, y1 = draw.textbbox((0, 0), ln, font=font)
        widths.append(x1 - x0)
        heights.append(y1 - y0)
    step = int(font_size * line_gap)
    total = step * (len(lines) - 1) + (heights[-1] if heights else 0)
    y = (h - total) // 2
    boxes = []
    for ln, tw in zip(lines, widths):
        x = (w - tw) // 2
        bx0, by0, bx1, by1 = draw.textbbox((x, y), ln, font=font)
        draw.text((x, y), ln, fill=255, font=font)
        boxes.append((bx0, by0, bx1, by1))
        y += step
    return np.asarray(img, np.float32) / 255.0, boxes


def make_stone(
    lines,
    size=(768, 512),
    font_size=46,
    depth=4.0,
    wear=0.5,
    lichen=0.5,
    seed=0,
) -> SyntheticStone:
    """Create a carved stone. wear in [0,1]: 0 = crisp, 1 = almost gone."""
    rng = np.random.default_rng(seed)
    w, h = size
    mask, boxes = render_text_mask(lines, size, font_size)
    stroke = mask > 0.5

    # V-groove profile: deepest at the stroke centre line.
    dist = cv2.distanceTransform(stroke.astype(np.uint8), cv2.DIST_L2, 5)
    half = max(1.0, float(np.percentile(dist[stroke], 95))) if stroke.any() else 1.0
    groove = np.clip(dist / half, 0, 1)
    height = -depth * groove

    # Erosion: soften edges, lose depth, pit the surface.
    sigma = 0.5 + 5.0 * wear
    height = cv2.GaussianBlur(height, (0, 0), sigma)
    height *= (1.0 - 0.8 * wear)
    pits = np.clip(_fractal_noise((h, w), rng) - 1.2, 0, None)
    height -= pits * (0.6 + 1.5 * wear)

    # Stone surface roughness (always present, independent of wear).
    height += 0.35 * _fractal_noise((h, w), rng, octaves=6) + 2.0 * cv2.GaussianBlur(
        _fractal_noise((h, w), rng, octaves=3), (0, 0), 40)

    # Albedo: stone colour variation + lichen blotches (dark/light patches).
    albedo = 0.72 + 0.06 * _fractal_noise((h, w), rng)
    blot = _fractal_noise((h, w), rng, octaves=4)
    lichen_mask = cv2.GaussianBlur((blot > (1.0 - 1.4 * lichen)).astype(np.float32), (0, 0), 3)
    albedo = albedo * (1 - 0.55 * lichen_mask) + 0.15 * lichen_mask * (_fractal_noise((h, w), rng) > 0)
    albedo = np.clip(albedo, 0.08, 1.0).astype(np.float32)

    return SyntheticStone(lines, height.astype(np.float32), albedo, stroke, boxes, wear)


def normals_from_height(height: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float32))
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def light_dir(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], np.float32)


def render(height, albedo, light, ambient=0.05, noise=0.01, rng=None):
    n = normals_from_height(height)
    shade = np.clip(n @ light, 0, None)
    img = albedo * shade + ambient * albedo
    if noise and rng is not None:
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def capture(stone: SyntheticStone, n_lights=36, elevations=(15, 25, 40), seed=1):
    """Simulate a Ghostlight capture: images under a sweep of known light directions."""
    rng = np.random.default_rng(seed)
    lights, images = [], []
    for i in range(n_lights):
        az = 360.0 * i / n_lights + rng.uniform(-3, 3)
        el = elevations[i % len(elevations)] + rng.uniform(-3, 3)
        L = light_dir(az, el)
        lights.append(L)
        images.append(render(stone.height, stone.albedo, L, rng=rng))
    return np.stack(images), np.stack(lights)


def plain_photo(stone: SyntheticStone, seed=2):
    """What a person sees: soft overhead daylight. Letters barely show, lichen dominates."""
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(stone.albedo)
    dirs = [light_dir(a, e) for a in range(0, 360, 30) for e in (55, 75)] + [np.array([0, 0, 1.0])]
    for L in dirs:
        acc += render(stone.height, stone.albedo, L, ambient=0.0, noise=0)
    acc /= len(dirs)
    return np.clip(acc + rng.normal(0, 0.01, acc.shape), 0, 1).astype(np.float32)
