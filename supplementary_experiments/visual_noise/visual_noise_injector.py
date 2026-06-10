"""RGB-level physical sensor noise for embodied perception (Appendix C).

Injects corruptions that mirror real on-robot camera degradation, operating
directly on the rendered RGB frame (not on the structured metadata stream
used by the main §3.4 noise engine).  This answers the reviewer concern that
the benchmark's noise is metadata-only.

Implemented with numpy + Pillow ONLY (both already project dependencies) so
the server needs no extra install (no OpenCV).  Each function takes and
returns an HxWx3 uint8 array; ``inject_physical_sensor_noise`` is the public
entry point and also accepts/saves file paths.

Corruption types
----------------
  motion_blur       — horizontal directional blur (chassis / head shake)
  low_illumination  — global intensity attenuation (night / lights-off)
  gaussian_sensor   — additive Gaussian electronic (dark-current) noise
  defocus_blur      — isotropic Gaussian blur (out-of-focus optics)  [bonus]

``intensity`` ∈ [0, 1] scales severity for every type.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter

NOISE_TYPES = ("motion_blur", "low_illumination", "gaussian_sensor", "defocus_blur")


# ─── individual corruptions (numpy uint8 in / out) ───────────────────────

def _motion_blur(img: np.ndarray, intensity: float) -> np.ndarray:
    """Horizontal motion blur via averaging k horizontally-shifted copies."""
    k = max(3, int(round(25 * intensity)))
    if k % 2 == 0:
        k += 1
    half = k // 2
    acc = np.zeros_like(img, dtype=np.float32)
    # pad on width axis so shifted copies stay aligned
    padded = np.pad(img, ((0, 0), (half, half), (0, 0)), mode="edge").astype(np.float32)
    w = img.shape[1]
    for off in range(k):
        acc += padded[:, off:off + w, :]
    acc /= k
    return np.clip(acc, 0, 255).astype(np.uint8)


def _low_illumination(img: np.ndarray, intensity: float) -> np.ndarray:
    """Attenuate brightness (up to 80% darker at intensity=1)."""
    factor = 1.0 - 0.8 * intensity
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _gaussian_sensor(img: np.ndarray, intensity: float,
                     seed: int | None = None) -> np.ndarray:
    """Additive zero-mean Gaussian noise (electronic dark-current)."""
    sigma = 50.0 * intensity
    rng = np.random.default_rng(seed)
    # use float32 to avoid the np.int96 overflow bug in the original draft
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _defocus_blur(img: np.ndarray, intensity: float) -> np.ndarray:
    """Isotropic Gaussian blur via Pillow."""
    radius = max(0.5, 6.0 * intensity)
    pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(pil, dtype=np.uint8)


_DISPATCH = {
    "motion_blur":      _motion_blur,
    "low_illumination": _low_illumination,
    "gaussian_sensor":  _gaussian_sensor,
    "defocus_blur":     _defocus_blur,
}


# ─── public entry point ──────────────────────────────────────────────────

def inject_physical_sensor_noise(
    image: "str | os.PathLike | np.ndarray | Image.Image",
    noise_type: str = "motion_blur",
    intensity: float = 0.5,
    seed: int | None = 0,
    out_path: str | None = None,
) -> np.ndarray:
    """Apply one RGB-level corruption.

    Parameters
    ----------
    image : path | ndarray | PIL.Image
        Input camera frame.
    noise_type : str   one of NOISE_TYPES
    intensity : float  in [0, 1]
    seed : int | None  RNG seed (only used by gaussian_sensor) for repro
    out_path : str | None  if given, the corrupted frame is also written here

    Returns
    -------
    np.ndarray  HxWx3 uint8 corrupted frame.
    """
    if noise_type not in _DISPATCH:
        raise ValueError(f"Unknown noise_type {noise_type!r}; "
                         f"expected one of {NOISE_TYPES}")

    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        if not os.path.exists(str(image)):
            raise FileNotFoundError(f"Cannot load camera stream: {image}")
        arr = np.asarray(Image.open(str(image)).convert("RGB"))

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {arr.shape}")

    fn = _DISPATCH[noise_type]
    out = (fn(arr, intensity, seed) if noise_type == "gaussian_sensor"
           else fn(arr, intensity))

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        Image.fromarray(out).save(out_path)
    return out


# ─── quick visual sanity check ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Corrupt one image for inspection.")
    p.add_argument("image")
    p.add_argument("--type", default="motion_blur", choices=NOISE_TYPES)
    p.add_argument("--intensity", type=float, default=0.7)
    p.add_argument("--out", default="sample_corrupted.png")
    a = p.parse_args()
    inject_physical_sensor_noise(a.image, a.type, a.intensity, out_path=a.out)
    print(f"wrote {a.out}  ({a.type} @ {a.intensity})")
