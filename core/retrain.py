"""
core/retrain.py — Fine-tuning TrOCR avec les corrections utilisateurs
"""

from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

from config.settings import settings
from core.dataset import load_all_records
import core.models as _m


# ─────────────────────────────────────────────────────────────
# Dataset PyTorch
# ─────────────────────────────────────────────────────────────
class CorrectionDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r      = self.records[idx]
        img    = Image.open(settings.CROPS_DIR / r["crop_file"]).convert("RGB")
        label  = r["text_corrige"]

        inputs = _m.processor(
            images=img, return_tensors="pt", padding=True, truncation=True
        )
        labels = _m.processor.tokenizer(
            label, return_tensors="pt",
            padding="max_length", max_length=64, truncation=True
        ).input_ids
        labels[labels == _m.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels":       labels.squeeze(),
        }


# ─────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────
def run_retrain(epochs: int = 5, lr: float = 5e-5):
    """Fine-tune TrOCR avec toutes les corrections valides."""
    print(f"\n{'='*50}\nRÉENTRAÎNEMENT TrOCR\n{'='*50}")

    # Filtrer les records valides
    records = [
        r for r in load_all_records()
        if (settings.CROPS_DIR / r["crop_file"]).exists()
        and r.get("text_corrige")
    ]
    print(f"Dataset : {len(records)} exemples valides")
    if not records:
        print("Aucun exemple valide — abandon.")
        return

    loader    = DataLoader(CorrectionDataset(records), batch_size=4, shuffle=True)
    _m.model_ocr.train()
    optimizer = AdamW(_m.model_ocr.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            pv   = batch["pixel_values"].to(settings.DEVICE)
            lbl  = batch["labels"].to(settings.DEVICE)
            loss = _m.model_ocr(pixel_values=pv, labels=lbl).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")

    _m.model_ocr.eval()

    # Sauvegarder le nouveau modèle
    v_new = Path(settings.TROCR_DIR).parent / "trocr_malagasy_v6"
    _m.model_ocr.save_pretrained(str(v_new))
    _m.processor.save_pretrained(str(v_new))
    print(f"✅  Nouveau modèle sauvegardé : {v_new}")
    print("=" * 50)
