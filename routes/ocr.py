"""
routes/ocr.py — POST /ocr
Lance TrOCR sur un dictionnaire de crops base64 { field: crop_b64 }.
"""

import traceback
from PIL import Image

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.image import decode_image, img_to_b64, preprocess_crop
from core.ocr import run_trocr

router = APIRouter()


@router.post("/ocr")
async def ocr(data: dict):
    try:
        crops   = data.get("crops", {})
        results = {}
        print(f"\n/ocr — {len(crops)} champs : {list(crops.keys())}")

        for field, crop_b64 in crops.items():
            if not crop_b64:
                results[field] = {"text": "", "crop_b64": ""}
                continue
            try:
                crop_rgb  = decode_image(crop_b64)
                processed = preprocess_crop(crop_rgb, field)
                text      = run_trocr(Image.fromarray(processed), field)
                results[field] = {
                    "text":     text,
                    "crop_b64": img_to_b64(crop_rgb),
                }
                print(f"  {field}: '{text}'")
            except Exception as ef:
                print(f"  {field} erreur: {ef}")
                results[field] = {"text": "", "crop_b64": ""}

        return JSONResponse({"status": "success", "results": results})

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
