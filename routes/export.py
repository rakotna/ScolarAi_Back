python
"""
routes/export.py — GET /export/{format}
Exporte le dataset au format demandé : json, csv, txt, stats.
"""

import csv
import json
import traceback
from collections import Counter
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse

from config.settings import settings
from core.dataset import load_all_records

router = APIRouter()


def _get_records_or_error():
    """Charge les records et retourne une erreur JSON si vide."""
    if not settings.METADATA_FILE.exists():
        return None, JSONResponse(
            {"status": "error", "message": "Pas de données"}, status_code=404
        )
    records = load_all_records()
    if not records:
        return None, JSONResponse(
            {"status": "error", "message": "Dataset vide"}, status_code=404
        )
    return records, None


@router.get("/export/{fmt}")
async def export_dataset(fmt: str):
    """
    GET /export/json   → fichier JSON complet
    GET /export/csv    → fichier CSV (field, text, confidence, date)
    GET /export/txt    → paires image\ttexte (format TrOCR)
    GET /export/stats  → statistiques du dataset
    """
    try:
        records, err = _get_records_or_error()
        if err:
            return err

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── JSON ──────────────────────────────────────────────
        if fmt == "json":
            out = settings.EXPORT_DIR / f"scolarai_dataset_{ts}.json"
            payload = {
                "version":  "ScolarAI 6.0",
                "exported": datetime.now().isoformat(),
                "total":    len(records),
                "modifies": sum(1 for r in records if r.get("modifie")),
                "records":  records,
            }
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return FileResponse(
                str(out), media_type="application/json",
                filename=out.name,
            )

        # ── CSV ───────────────────────────────────────────────
        elif fmt == "csv":
            out = settings.EXPORT_DIR / f"scolarai_dataset_{ts}.csv"
            _FIELDS = ["id", "field", "text_corrige", "text_original",
                       "modifie", "confidence", "timestamp", "session_id"]
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDS)
                writer.writeheader()
                for r in records:
                    writer.writerow({k: r.get(k, "") for k in _FIELDS})
            return FileResponse(
                str(out), media_type="text/csv", filename=out.name
            )

        # ── TXT (format TrOCR) ────────────────────────────────
        elif fmt == "txt":
            out = settings.EXPORT_DIR / f"scolarai_labels_{ts}.txt"
            with open(out, "w", encoding="utf-8") as f:
                for r in records:
                    if r.get("crop_file") and r.get("text_corrige"):
                        f.write(f"{r['crop_file']}\t{r['text_corrige']}\n")
            return FileResponse(
                str(out), media_type="text/plain", filename=out.name
            )

        # ── Stats ─────────────────────────────────────────────
        elif fmt == "stats":
            n_modifies = sum(1 for r in records if r.get("modifie"))
            fields     = Counter(r.get("field", "") for r in records)
            conf_moy   = {
                field: round(
                    sum(r.get("confidence", 0) for r in records if r.get("field") == field)
                    / max(count, 1) * 100, 1
                )
                for field, count in fields.items()
            }
            return JSONResponse({
                "status":          "success",
                "total":           len(records),
                "modifies":        n_modifies,
                "confirmes":       len(records) - n_modifies,
                "taux_correction": round(n_modifies / max(len(records), 1) * 100, 1),
                "par_champ":       dict(fields),
                "confiance_moy":   conf_moy,
                "crops_count":     len(list(settings.CROPS_DIR.glob("*.png"))),
            })

        else:
            return JSONResponse(
                {"status": "error", "message": "Format invalide. Utilisez : json, csv, txt, stats"},
                status_code=400,
            )

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
