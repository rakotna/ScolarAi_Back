"""
routes/save.py — POST /save
Sauvegarde les corrections utilisateurs dans Google Drive.
"""

import traceback
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.dataset import save_correction, count_corrections

router = APIRouter()


@router.post("/save")
async def save(data: dict):
    """
    Body attendu :
    {
      "session_id": "uuid",
      "corrections": {
        "nom": {
          "text_original": "Eddrey",
          "text_corrige":  "Edrey",
          "crop_b64":      "...",
          "confidence":    0.71,
          "modifie":       true
        },
        ...
      }
    }
    """
    try:
        session_id  = data.get("session_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        corrections = data.get("corrections", {})
        saved: list[str]   = []
        skipped: list[str] = []

        print(f"\n/save — session: {session_id} | {len(corrections)} champs")

        for field, info in corrections.items():
            text_corrige  = info.get("text_corrige", "").strip()
            text_original = info.get("text_original", "").strip()
            crop_b64      = info.get("crop_b64", "")
            confidence    = float(info.get("confidence", 0.0))
            modifie       = bool(info.get("modifie", False))

            if not text_corrige or not crop_b64:
                skipped.append(field)
                continue

            save_correction(
                field=field,
                session_id=session_id,
                text_corrige=text_corrige,
                text_original=text_original,
                crop_b64=crop_b64,
                confidence=confidence,
                modifie=modifie,
            )
            saved.append(field)
            icon = "✏️" if modifie else "✅"
            print(f"  [{field}] {icon}: '{text_original}' → '{text_corrige}'")

        total = count_corrections()
        print(f"  Sauvegardé: {saved} | Ignoré: {skipped} | Total: {total}")

        return JSONResponse({
            "status":            "success",
            "saved":             saved,
            "skipped":           skipped,
            "total_corrections": total,
            "message":           f"{len(saved)} correction(s) sauvegardée(s)",
        })

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
