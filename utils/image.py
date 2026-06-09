"""
utils/image.py — Manipulation d'images (base64, crop OBB, prétraitement)
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Encodage / décodage base64
# ─────────────────────────────────────────────────────────────
def decode_image(b64_str: str) -> np.ndarray:
    """Décoder une image base64 (avec ou sans en-tête data:image) → RGB numpy."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


def img_to_b64(img_rgb: np.ndarray) -> str:
    """Encoder une image RGB numpy → JPEG base64."""
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────
# Extraction de crop orienté (OBB)
# ─────────────────────────────────────────────────────────────
def extract_rotated_crop(
    img_rgb: np.ndarray,
    cx: float, cy: float,
    bw: float, bh: float,
    angle_rad: float,
) -> np.ndarray | None:
    """Extraire un crop redressé selon l'angle OBB."""
    angle_deg = float(angle_rad * 180.0 / np.pi)
    img_h, img_w = img_rgb.shape[:2]

    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img_rgb, M, (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    half_w = max(int(bw / 2), 4)
    half_h = max(int(bh / 2), 4)
    x1 = max(0, int(cx) - half_w)
    y1 = max(0, int(cy) - half_h)
    x2 = min(img_w, int(cx) + half_w)
    y2 = min(img_h, int(cy) + half_h)

    if x2 <= x1 or y2 <= y1:
        return None
    return rotated[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────
# Prétraitement par champ
# ─────────────────────────────────────────────────────────────
def preprocess_crop(crop_rgb: np.ndarray, field: str) -> np.ndarray:
    """Appliquer le prétraitement adapté au type de champ."""
    try:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

        if field in ("note", "matricule"):
            # Agrandissement ×4 + binarisation Otsu (chiffres)
            h, w = gray.shape
            resized = cv2.resize(
                gray, (max(w * 4, 64), max(h * 4, 64)),
                interpolation=cv2.INTER_CUBIC,
            )
            _, binary = cv2.threshold(
                resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        else:
            # Agrandissement ×2 + CLAHE + débruitage + seuillage adaptatif (texte)
            h, w = gray.shape
            resized = cv2.resize(
                gray, (max(w * 2, 64), max(h * 2, 64)),
                interpolation=cv2.INTER_CUBIC,
            )
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            binary   = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    except Exception:
        return crop_rgb
