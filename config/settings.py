"""
config/settings.py — Configuration centralisée ScolarAI
"""

from pathlib import Path
import torch


class Settings:
    # ── Chemins modèles ────────────────────────────────────────
    TROCR_DIR: str  = "/content/gdrive/MyDrive/Colab Notebooks/trocr_malagasy_v5"
    YOLO_PATH: str  = "/content/gdrive/MyDrive/Colab Notebooks/model_best/best.pt"

    # ── Dataset Google Drive ───────────────────────────────────
    DATASET_DIR: Path = Path("/content/gdrive/MyDrive/ScolarAI_Dataset")

    @property
    def CROPS_DIR(self) -> Path:
        return self.DATASET_DIR / "crops"

    @property
    def METADATA_FILE(self) -> Path:
        return self.DATASET_DIR / "metadata.jsonl"

    @property
    def EXPORT_DIR(self) -> Path:
        return self.DATASET_DIR / "exports"

    # ── Serveur ────────────────────────────────────────────────
    PORT: int = 8000
    NGROK_TOKEN: str = "386TIPPyrgsQsLZXDiShtj0y8mO_88u84aDzQXMwrG1vgoqFt"

    # ── Device IA ──────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Couleurs par champ (pour Flutter) ─────────────────────
    FIELD_COLORS: dict = {
        "nom":       "#6C63FF",
        "prenom":    "#00D4FF",
        "matiere":   "#FFBE0B",
        "note":      "#00FF94",
        "matricule": "#FF6B6B",
        "classe":    "#FF9F43",
    }

    def init_dirs(self):
        """Créer les dossiers dataset si absents."""
        for d in [self.DATASET_DIR, self.CROPS_DIR, self.EXPORT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.init_dirs()
