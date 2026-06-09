"""
routes/status.py — GET /status
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config.settings import settings
from core.dataset import count_corrections
import core.models as _m

router = APIRouter()


@router.get("/")
@router.get("/status")
def status():
    return JSONResponse({
        "status":      "running",
        "device":      settings.DEVICE,
        "version":     "ScolarAI 6.0",
        "classes":     list(_m.model_yolo.names.values()),
        "corrections": count_corrections(),
    })
