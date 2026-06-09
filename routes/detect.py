"""
routes/detect.py — POST /detect
YOLO OBB : détecte les champs du bulletin.
"identite" × 2 → triées par Y → nom (haut) + prenom (bas).
"""

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config.settings import settings
from utils.image import decode_image, img_to_b64, extract_rotated_crop
import core.models as _m

router = APIRouter()

# Couleur dédiée pour "identite" avant séparation
_IDENTITE_COLOR = "#A78BFA"


@router.post("/detect")
async def detect(data: dict):
    try:
        b64 = data.get("image", "")
        if not b64:
            return JSONResponse(
                {"status": "error", "message": "Pas d'image"}, status_code=400
            )

        print(f"\n/detect — b64: {len(b64)} chars")
        img_rgb = decode_image(b64)
        h, w    = img_rgb.shape[:2]

        results = _m.model_yolo.predict(img_rgb, conf=0.20, verbose=False)
        obb     = results[0].obb

        raw: list[dict] = []

        if obb is not None and len(obb) > 0:
            print(f"  OBB : {len(obb)} détections brutes")
            xywhr_all = obb.xywhr

            for i in range(len(obb)):
                conf_score            = float(obb.conf[i])
                field                 = _m.model_yolo.names[int(obb.cls[i])]
                cx, cy, bw, bh, a_rad = xywhr_all[i].tolist()
                angle_deg             = a_rad * 180.0 / 3.14159265

                x1, y1, x2, y2 = obb.xyxy[i].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop_rgb = extract_rotated_crop(img_rgb, cx, cy, bw, bh, a_rad)
                if crop_rgb is None or crop_rgb.size == 0:
                    crop_rgb = img_rgb[y1:y2, x1:x2]

                corners = obb.xyxyxyxy[i].tolist()

                raw.append({
                    "field":      field,
                    "confidence": round(conf_score, 3),
                    "crop_b64":   img_to_b64(crop_rgb),
                    "cy_norm":    cy / h,
                    "box": {
                        "left":   round(x1 / w, 4),
                        "top":    round(y1 / h, 4),
                        "width":  round((x2 - x1) / w, 4),
                        "height": round((y2 - y1) / h, 4),
                    },
                    "obb": {
                        "cx":      round(cx / w, 4),
                        "cy":      round(cy / h, 4),
                        "w":       round(bw / w, 4),
                        "h":       round(bh / h, 4),
                        "angle":   round(angle_deg, 2),
                        "corners": [
                            {"x": round(c[0] / w, 4), "y": round(c[1] / h, 4)}
                            for c in corners
                        ],
                    },
                })
        else:
            print("  Aucune détection OBB")

        # ── Meilleure box par champ (hors identite) ──────────
        best: dict[str, dict] = {}
        for box in raw:
            if box["field"] == "identite":
                continue
            f = box["field"]
            if f not in best or box["confidence"] > best[f]["confidence"]:
                best[f] = box

        final: list[dict] = list(best.values())

        # ── Séparation identite → nom / prenom (tri par Y) ───
        identite_boxes = sorted(
            [b for b in raw if b["field"] == "identite"],
            key=lambda x: x["cy_norm"],
        )

        labels = ("nom", "prenom")
        for label, box in zip(labels, identite_boxes):
            mapped        = dict(box)
            mapped["field"] = label
            mapped["color"] = settings.FIELD_COLORS[label]
            final.append(mapped)
            print(f"  [{label}] cy={box['cy_norm']:.3f} conf={box['confidence']:.2f}")

        # Couleur + nettoyage cy_norm
        for box in final:
            box.setdefault("color", settings.FIELD_COLORS.get(box["field"], "#FFFFFF"))
            box.pop("cy_norm", None)

        print(f"  → {len(final)} boxes : {[b['field'] for b in final]}")

        return JSONResponse({
            "status": "success",
            "boxes":  final,
            "count":  len(final),
        })

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
