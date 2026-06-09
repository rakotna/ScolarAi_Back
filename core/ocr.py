"""
core/ocr.py — Reconnaissance TrOCR + post-traitement par champ
"""

import re
import torch
from PIL import Image

from config.settings import settings
import core.models as _m


# ─────────────────────────────────────────────────────────────
# Post-traitements spécifiques
# ─────────────────────────────────────────────────────────────
def _clean_note(text: str) -> str:
    cleaned = text.strip().replace(",", ".")
    try:
        val = float(cleaned)
        if 0 <= val <= 20:
            return str(int(val)) if val == int(val) else str(round(val, 1)).replace(".", ",")
        digits = re.sub(r"\D", "", cleaned)
        for i in range(1, len(digits)):
            c = float(digits[:i] + "." + digits[i:])
            if 0 <= c <= 20:
                return str(round(c, 1)).replace(".", ",")
    except ValueError:
        pass
    return text.strip()


def _clean_matricule(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    return digits if 1 <= len(digits) <= 4 else text.strip()


def postprocess(text: str, field: str) -> str:
    text = text.strip()
    if field == "note":
        return _clean_note(text)
    if field == "matricule":
        return _clean_matricule(text)
    return text


# ─────────────────────────────────────────────────────────────
# Inférence TrOCR
# ─────────────────────────────────────────────────────────────
def run_trocr(pil_image: Image.Image, field: str) -> str:
    pixel_values = _m.processor(
        images=pil_image, return_tensors="pt"
    ).pixel_values.to(settings.DEVICE)

    with torch.no_grad():
        generated = _m.model_ocr.generate(
            pixel_values, max_new_tokens=64, num_beams=4
        )
    text = _m.processor.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()

    return postprocess(text, field)
