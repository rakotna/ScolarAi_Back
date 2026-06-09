"""
Factory de l'application FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.status  import router as status_router
from routes.detect  import router as detect_router
from routes.ocr     import router as ocr_router
from routes.save    import router as save_router
from routes.export  import router as export_router
from routes.retrain import router as retrain_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="ScolarAI API",
        version="6.0",
        description="OCR automatique de bulletins scolaires (YOLO OBB + TrOCR)",
    )

    # CORS ouvert (à restreindre en production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enregistrement des routes
    app.include_router(status_router)
    app.include_router(detect_router)
    app.include_router(ocr_router)
    app.include_router(save_router)
    app.include_router(export_router)
    app.include_router(retrain_router)

    return app
