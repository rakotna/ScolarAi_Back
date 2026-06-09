"""
core/dataset.py — Lecture / écriture du dataset ScolarAI (metadata.jsonl + crops)
"""

import json
from datetime import datetime
from pathlib import Path
from PIL import Image

from config.settings import settings
from utils.image import decode_image


# ─────────────────────────────────────────────────────────────
# Comptage
# ─────────────────────────────────────────────────────────────
def count_corrections() -> int:
    if not settings.METADATA_FILE.exists():
        return 0
    with open(settings.METADATA_FILE, "r") as f:
        return sum(1 for line in f if line.strip())


# ─────────────────────────────────────────────────────────────
# Lecture complète
# ─────────────────────────────────────────────────────────────
def load_all_records() -> list[dict]:
    records = []
    if not settings.METADATA_FILE.exists():
        return records
    with open(settings.METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ─────────────────────────────────────────────────────────────
# Sauvegarde d'une correction
# ─────────────────────────────────────────────────────────────
def save_correction(
    field: str,
    session_id: str,
    text_corrige: str,
    text_original: str,
    crop_b64: str,
    confidence: float,
    modifie: bool,
) -> str:
    """
    Sauvegarde l'image crop + label .txt + entrée metadata.jsonl.
    Retourne le nom de fichier sans extension.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename  = f"{field}_{session_id}_{timestamp}"

    # Image crop
    crop_rgb = decode_image(crop_b64)
    Image.fromarray(crop_rgb).save(
        str(settings.CROPS_DIR / f"{filename}.png"), format="PNG"
    )

    # Label texte (format TrOCR)
    (settings.CROPS_DIR / f"{filename}.txt").write_text(
        text_corrige, encoding="utf-8"
    )

    # Metadata
    meta = {
        "id":             filename,
        "session_id":     session_id,
        "field":          field,
        "text_corrige":   text_corrige,
        "text_original":  text_original,
        "modifie":        modifie,
        "confidence":     confidence,
        "crop_file":      f"{filename}.png",
        "timestamp":      datetime.now().isoformat(),
    }
    with open(settings.METADATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return filename
