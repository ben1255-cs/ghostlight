"""Witness 1, part A: rebuilding the stone's surface from a light sweep.

Robust photometric stereo: for every pixel we observe its brightness under many
known light directions and solve for the surface normal and albedo. Shadows and
glare are excluded per pixel. Runs on NumPy by default, or on an NVIDIA GPU via
PyTorch (CUDA) when available, which is how it runs on Nebius Serverless Jobs.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Surface:
    normals: np.ndarray   # HxWx3 unit normals
    albedo: np.ndarray    # HxW (colour, including lichen) -- deliberately ignored for reading
    height: np.ndarray    # HxW integrated relative height
    relief: np.ndarray    # HxW high-pass height: carving with stone shape removed
    groove: np.ndarray    # HxW in [0,1]: how strongly each pixel looks carved


def _solve_numpy(I, L, low_q, high_q):
    n, h, w = I.shape
    flat = I.reshape(n, -1)
    lo = np.quantile(flat, low_q, axis=0)
    hi = np.quantile(flat, high_q, axis=0)
    W = ((flat >= lo) & (flat <= hi)).astype(np.float32)          # N x P
    A = np.einsum("np,ni,nj->pij", W, L, L) + 1e-6 * np.eye(3)     # P x 3 x 3
    b = np.einsum("np,ni,np->pi", W, L, flat)                      # P x 3
    g = np.linalg.solve(A, b[..., None])[..., 0]
    return g.reshape(h, w, 3)


def _solve_torch(I, L, low_q, high_q):
    import torch
    dev = "cuda"
    n, h, w = I.shape
    flat = torch.as_tensor(I.reshape(n, -1), device=dev)
    Lt = torch.as_tensor(L, device=dev)
    lo = torch.quantile(flat, low_q, dim=0)
    hi = torch.quantile(flat, high_q, dim=0)
    W = ((flat >= lo) & (flat <= hi)).float()
    A = torch.einsum("np,ni,nj->pij", W, Lt, Lt) + 1e-6 * torch.eye(3, device=dev)
    b = torch.einsum("np,ni,np->pi", W, Lt, flat)
    g = torch.linalg.solve(A, b)
    return g.reshape(h, w, 3).cpu().numpy()


def _use_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def frankot_chellappa(normals: np.ndarray) -> np.ndarray:
    """Integrate a normal field into a height map (least squares, Fourier domain)."""
    nz = np.clip(normals[..., 2], 0.05, None)
    p = -normals[..., 0] / nz
    q = -normals[..., 1] / nz
    h, w = p.shape
    wx, wy = np.meshgrid(np.fft.fftfreq(w) * 2 * np.pi, np.fft.fftfreq(h) * 2 * np.pi)
    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0
    Z = (-1j * wx * np.fft.fft2(p) - 1j * wy * np.fft.fft2(q)) / denom
    Z[0, 0] = 0
    return np.real(np.fft.ifft2(Z)).astype(np.float32)


def reconstruct(images: np.ndarray, lights: np.ndarray, low_q=0.2, high_q=0.95,
                relief_sigma=12.0, use_gpu=None) -> Surface:
    """images: N x H x W in [0,1]; lights: N x 3 unit vectors (from the sphere or known)."""
    images = images.astype(np.float32)
    lights = lights.astype(np.float32)
    if use_gpu is None:
        use_gpu = _use_gpu()
    g = _solve_torch(images, lights, low_q, high_q) if use_gpu else _solve_numpy(images, lights, low_q, high_q)
    albedo = np.linalg.norm(g, axis=2)
    normals = g / (albedo[..., None] + 1e-8)
    normals[..., 2] = np.abs(normals[..., 2])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8

    height = frankot_chellappa(normals)
    relief = height - cv2.GaussianBlur(height, (0, 0), relief_sigma)
    groove = groove_map(relief)
    return Surface(normals.astype(np.float32), albedo.astype(np.float32), height, relief.astype(np.float32), groove)


def groove_map(relief: np.ndarray) -> np.ndarray:
    """Carving = locally low. Normalise robustly to [0,1]."""
    g = -relief
    lo, hi = np.percentile(g, [50, 99.5])
    return np.clip((g - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def relight(surface: Surface, azimuth_deg: float, elevation_deg: float, use_albedo=False) -> np.ndarray:
    """Virtual raking light. Uniform albedo by default: this is the 'lichen trick',
    colour is thrown away so only the carved shape is shown."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    L = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], np.float32)
    shade = np.clip(surface.normals @ L, 0, None)
    if use_albedo:
        shade = shade * surface.albedo
    lo, hi = np.percentile(shade, [1, 99])
    return np.clip((shade - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def enhanced(surface: Surface) -> np.ndarray:
    """Relief image for reading: grooves dark, surface light."""
    return (1.0 - surface.groove).astype(np.float32)


def find_lines(groove: np.ndarray, min_height=12, thresh_frac=0.35, pad=6):
    """Locate text lines from the horizontal projection of the groove map.
    Returns boxes (x0, y0, x1, y1)."""
    g = cv2.GaussianBlur(groove, (0, 0), 2)
    strong = (g > 0.35).astype(np.float32)
    prof = strong.mean(axis=1)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    if prof.max() <= 0:
        return []
    on = prof > thresh_frac * prof.max()
    boxes, y = [], 0
    H, W = groove.shape
    while y < H:
        if on[y]:
            y0 = y
            while y < H and on[y]:
                y += 1
            y1 = y
            if y1 - y0 >= min_height:
                cols = np.convolve(strong[y0:y1].mean(axis=0), np.ones(9) / 9, mode="same")
                xs = np.where(cols > 0.04)[0]
                if xs.size:
                    # Split into clusters on big gaps; keep the one carrying most ink
                    # (drops pits and edge damage that happen to share the row band).
                    gap = 3 * (y1 - y0)
                    splits = np.where(np.diff(xs) > gap)[0]
                    groups = np.split(xs, splits + 1)
                    best = max(groups, key=lambda gr: cols[gr].sum())
                    boxes.append((max(0, int(best[0]) - pad), max(0, y0 - pad),
                                  min(W, int(best[-1]) + pad), min(H, y1 + pad)))
        y += 1
    return boxes


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
