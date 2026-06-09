"""
core/models.py — Chargement unique des modèles IA (YOLO + TrOCR)
"""

import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from config.settings import settings

# Références globales (initialisées par load_models())
model_yolo: YOLO = None
model_ocr:  VisionEncoderDecoderModel = None
processor:  AutoProcessor = None


def load_models():
    """Charger YOLO et TrOCR une seule fois au démarrage."""
    global model_yolo, model_ocr, processor

    print(f"Device: {settings.DEVICE}")
    if settings.DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── TrOCR ─────────────────────────────────────────────────
    print("Chargement TrOCR V5...")
    processor = AutoProcessor.from_pretrained(
        settings.TROCR_DIR, local_files_only=True
    )
    model_ocr = VisionEncoderDecoderModel.from_pretrained(
        settings.TROCR_DIR, local_files_only=True
    ).to(settings.DEVICE)
    model_ocr.eval()
    print("TrOCR V5 ✅")

    # ── YOLO OBB ──────────────────────────────────────────────
    print("Chargement YOLO OBB...")
    model_yolo = YOLO(settings.YOLO_PATH)
    print(f"YOLO ✅  — Classes: {model_yolo.names}")
