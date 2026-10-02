"""Turning a real phone video into calibrated images.

Setup: phone on a small tripod facing the stone; a torch (or second phone's
flashlight) swept in low arcs around it; a shiny ball (ball bearing / glossy
marble) somewhere in frame. The bright highlight on the ball tells us where the
light was in every frame, which is the same calibration professional RTI uses.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Sphere:
    cx: float
    cy: float
    r: float


def read_frames(video_path: str, max_frames=120, min_brightness_change=0.02):
    """Read a capture video, keeping up to max_frames evenly spaced, distinct frames."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, n // max_frames) if n else 1
    frames, prev, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev is None or np.abs(g - prev).mean() > min_brightness_change:
                frames.append(g)
                prev = g
        i += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 1, 1), np.float32)


def stabilise(frames: np.ndarray) -> np.ndarray:
    """Remove small camera wobble by aligning every frame to the median image (ECC)."""
    ref = np.median(frames, axis=0).astype(np.float32)
    ref_e = cv2.Canny((ref * 255).astype(np.uint8), 40, 120).astype(np.float32)
    out = []
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for f in frames:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            fe = cv2.Canny((f * 255).astype(np.uint8), 40, 120).astype(np.float32)
            _, warp = cv2.findTransformECC(ref_e, fe, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
            f = cv2.warpAffine(f, warp, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            pass
        out.append(f)
    return np.stack(out)


def find_sphere(frames: np.ndarray) -> Sphere | None:
    """Locate the calibration ball: a circle that holds a moving bright highlight."""
    var = frames.max(axis=0) - np.median(frames, axis=0)
    img = cv2.GaussianBlur((np.median(frames, axis=0) * 255).astype(np.uint8), (0, 0), 2)
    h, w = img.shape
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w / 4, param1=100, param2=25,
                               minRadius=max(6, w // 60), maxRadius=w // 6)
    if circles is None:
        return None
    best, best_s = None, -1.0
    for cx, cy, r in circles[0]:
        yy, xx = np.ogrid[:h, :w]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.9 * r) ** 2
        s = float(var[m].mean()) if m.any() else 0.0
        if s > best_s:
            best, best_s = Sphere(float(cx), float(cy), float(r)), s
    return best


def light_from_highlight(frame: np.ndarray, sphere: Sphere) -> np.ndarray | None:
    """Mirror-ball geometry: the highlight sits where the sphere normal bisects the
    view direction and the light direction, so L = 2(n.v)n - v with v = (0,0,1)."""
    h, w = frame.shape
    x0, x1 = int(max(0, sphere.cx - sphere.r)), int(min(w, sphere.cx + sphere.r + 1))
    y0, y1 = int(max(0, sphere.cy - sphere.r)), int(min(h, sphere.cy + sphere.r + 1))
    patch = cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 1.0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - sphere.cx) ** 2 + (yy - sphere.cy) ** 2 <= (0.97 * sphere.r) ** 2
    vals = np.where(inside, patch, 0)
    thr = vals.max() * 0.9
    if vals.max() < 0.3:
        return None
    sel = vals >= thr
    hx, hy = xx[sel].mean(), yy[sel].mean()
    nx, ny = (hx - sphere.cx) / sphere.r, (hy - sphere.cy) / sphere.r
    nz2 = 1 - nx * nx - ny * ny
    if nz2 <= 0:
        return None
    n = np.array([nx, ny, np.sqrt(nz2)], np.float32)
    v = np.array([0, 0, 1], np.float32)
    L = 2 * (n @ v) * n - v
    return L / np.linalg.norm(L)


def calibrate(frames: np.ndarray, sphere: Sphere | None = None):
    """Returns (images, lights) for frames where the light could be located."""
    sphere = sphere or find_sphere(frames)
    if sphere is None:
        raise ValueError("Calibration ball not found. Make sure the shiny ball is fully in frame.")
    imgs, lights = [], []
    for f in frames:
        L = light_from_highlight(f, sphere)
        if L is not None and L[2] > 0.05:
            imgs.append(f)
            lights.append(L)
    if len(imgs) < 8:
        raise ValueError(f"Only {len(imgs)} usable frames; sweep the light more slowly and keep it low.")
    return np.stack(imgs), np.stack(lights), sphere


def mask_sphere(images: np.ndarray, sphere: Sphere, pad=1.15) -> np.ndarray:
    """Hide the ball before reconstruction so its glare can't leak into the stone."""
    out = images.copy()
    h, w = images.shape[1:]
    yy, xx = np.ogrid[:h, :w]
    m = (xx - sphere.cx) ** 2 + (yy - sphere.cy) ** 2 <= (pad * sphere.r) ** 2
    out[:, m] = np.median(images[:, ~m], axis=1)[:, None]
    return out
