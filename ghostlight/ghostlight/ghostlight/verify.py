"""Analysis by synthesis: does a proposed reading physically fit the stone?

Given a line of the reconstructed groove map and a candidate reading, we render
the candidate as virtual carving (V-shaped grooves, softened to match the
observed wear), search over size, position and wear, and measure how well each
predicted letter lines up with the real grooves. Every letter gets its own
support score, so the agent learns exactly which characters the stone supports.

This is the guard against hallucination: a reading the language model 'likes'
is rejected if its strokes are not physically present in the stone.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .fonts import get_font


@dataclass
class Verification:
    text: str
    score: float
    char_scores: list[float] = field(default_factory=list)
    font_size: int = 0
    offset: tuple[int, int] = (0, 0)
    wear_sigma: float = 0.0

    def weakest(self, k=3):
        pairs = [(i, c, s) for i, (c, s) in enumerate(zip(self.text, self.char_scores)) if not c.isspace()]
        return sorted(pairs, key=lambda p: p[2])[:k]

    def as_dict(self):
        return {
            "text": self.text,
            "score": round(float(self.score), 4),
            "letters": [
                {"pos": i, "char": c, "support": round(float(s), 3)}
                for i, (c, s) in enumerate(zip(self.text, self.char_scores)) if not c.isspace()
            ],
        }


def render_line(text: str, font_size: int, font_path=None, wear_sigma=2.0):
    """Render a line as virtual carving: V-groove depth profile, eroded by wear_sigma.
    Returns the template and the x-extent of each character."""
    font = get_font(font_size, font_path)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    m = int(3 * wear_sigma) + 3
    w, h = x1 - x0 + 2 * m, y1 - y0 + 2 * m
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((m - x0, m - y0), text, fill=255, font=font)
    mask = (np.asarray(img) > 127).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half = float(np.percentile(dist[mask > 0], 95)) if mask.any() else 1.0
    prof = np.clip(dist / max(half, 1.0), 0, 1)
    if wear_sigma > 0:
        prof = cv2.GaussianBlur(prof, (0, 0), wear_sigma)
    prof = prof / (prof.max() + 1e-8)
    extents = []
    for i in range(len(text)):
        a = probe.textlength(text[:i], font=font) - x0 + m
        b = probe.textlength(text[: i + 1], font=font) - x0 + m
        extents.append((int(round(a)), int(round(b))))
    return prof.astype(np.float32), extents


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _pad_to_fit(G, th, tw):
    H, W = G.shape
    py, px = max(0, th - H + 6), max(0, tw - W + 6)
    if not py and not px:
        return G, 0, 0
    Gp = cv2.copyMakeBorder(G, py // 2, py - py // 2, px // 2, px - px // 2, cv2.BORDER_CONSTANT, value=0)
    return Gp, py // 2, px // 2


def _align(G, text, size_range, sigmas, font_path):
    best = None
    for fs in size_range:
        for sg in sigmas:
            T, ext = render_line(text, fs, font_path, sg)
            Gp, oy, ox = _pad_to_fit(G, *T.shape)
            res = cv2.matchTemplate(Gp, T, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, fs, sg, loc, T, ext, Gp, oy, ox)
    return best


def score_at(Gp, T, ext, text, x, y, shift=2, xshift=4):
    """Per-letter support: NCC between each predicted letter and the stone in the
    letter's own cell. The search window follows the carving left to right, so
    uneven hand-cut spacing (or a slightly wrong size estimate) doesn't pile up
    into misalignment at the end of a long line."""
    th, tw = T.shape
    H, W = Gp.shape
    scores = []
    drift = 0
    for (a, b), ch in zip(ext, text):
        if ch.isspace():
            scores.append(1.0)
            continue
        a2, b2 = max(0, a - 2), min(tw, b + 2)
        tcell = T[:, a2:b2]
        if tcell.std() < 1e-6:
            scores.append(0.0)
            continue
        y0, y1 = max(0, y - shift), min(H, y + th + shift)
        cx = x + a2 + drift
        x0, x1 = max(0, cx - xshift), min(W, cx + (b2 - a2) + xshift)
        region = Gp[y0:y1, x0:x1]
        if region.shape[0] < tcell.shape[0] or region.shape[1] < tcell.shape[1]:
            scores.append(0.0)
            continue
        res = np.nan_to_num(cv2.matchTemplate(region, tcell, cv2.TM_CCOEFF_NORMED))
        _, mx, _, (bx, _) = cv2.minMaxLoc(res)
        scores.append(float(mx))
        if mx > 0.5:   # only trust confident letters to update the drift
            drift += (x0 + bx) - cx
    return scores


def verify_line(groove_line: np.ndarray, text: str, size_range=None, sigmas=(1.0, 2.5, 4.0),
                font_path=None) -> Verification:
    """groove_line: crop of the groove map around one text line (with margin)."""
    G = groove_line.astype(np.float32)
    H, _ = G.shape
    if not text.strip():
        return Verification(text, 0.0, [])
    if size_range is None:
        size_range = range(max(8, int(H * 0.55)), max(10, int(H * 1.15)), 2)
    mx, fs, sg, (x, y), T, ext, Gp, oy, ox = _align(G, text, size_range, sigmas, font_path)
    # refine size at 1px around the coarse best
    mx2 = _align(G, text, range(fs - 1, fs + 2), (sg,), font_path)
    if mx2[0] > mx:
        mx, fs, sg, (x, y), T, ext, Gp, oy, ox = mx2
    chars = score_at(Gp, T, ext, text, x, y)
    letters = [s for c, s in zip(text, chars) if not c.isspace()]
    score = float(np.mean(letters))
    return Verification(text, score, chars, fs, (x - ox, y - oy), sg)


def compare(groove_line: np.ndarray, candidates: list[str], font_path=None, **kw) -> list[Verification]:
    """Score several readings of the same line; best first.

    All candidates are judged under ONE shared geometry (letter size and wear),
    fitted to the stone using every candidate, so the only thing that differs
    between them is the letters themselves. This makes single-letter questions
    (is it a 1 or a 4?) far more reliable."""
    if len(candidates) == 1:
        return [verify_line(groove_line, candidates[0], font_path=font_path, **kw)]
    G = groove_line.astype(np.float32)
    H = G.shape[0]
    sizes = kw.get("size_range") or range(max(8, int(H * 0.55)), max(10, int(H * 1.15)), 2)
    sigmas = kw.get("sigmas") or (1.0, 2.5, 4.0)
    # Shared geometry from the best-fitting candidate.
    fits = [_align(G, c, sizes, sigmas, font_path) for c in candidates]
    best = max(fits, key=lambda f: f[0])
    fs, sg = best[1], best[2]
    refined = _align(G, candidates[fits.index(best)], range(fs - 1, fs + 2), (sg,), font_path)
    fs, sg = refined[1], refined[2]
    anchor = refined[3]  # where the best-fitting reading sits on the stone
    out = []
    for c in candidates:
        mx, _, _, (x, y), T, ext, Gp, oy, ox = _align(G, c, (fs,), (sg,), font_path)
        trials = [(x, y), anchor]
        best_v = None
        for (tx, ty) in trials:
            chars = score_at(Gp, T, ext, c, tx, ty)
            letters = [v for ch, v in zip(c, chars) if not ch.isspace()]
            score = float(np.mean(letters))
            if best_v is None or score > best_v.score:
                best_v = Verification(c, score, chars, fs, (tx - ox, ty - oy), sg)
        out.append(best_v)
    return sorted(out, key=lambda v: v.score, reverse=True)


def ambiguous(ranked: list[Verification], tol=0.01) -> list[Verification]:
    """Readings the stone alone cannot tell apart from the best one."""
    if not ranked:
        return []
    return [v for v in ranked if ranked[0].score - v.score <= tol]


def crop_line(groove: np.ndarray, box, margin=10):
    x0, y0, x1, y1 = box
    H, W = groove.shape
    return groove[max(0, y0 - margin):min(H, y1 + margin), max(0, x0 - margin):min(W, x1 + margin)]
