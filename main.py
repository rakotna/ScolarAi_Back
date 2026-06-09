"""
ScolarAI API v6.0 — Point d'entrée principal
"""

import os
import time
import threading
import uvicorn
from pyngrok import ngrok, conf

from app import create_app
from config.settings import settings
from core.models import load_models

# ── Auth ngrok ─────────────────────────────────────────────────
conf.get_default().auth_token = settings.NGROK_TOKEN

# ── Créer l'application ────────────────────────────────────────
app = create_app()

def main():
    # Charger les modèles IA
    load_models()

    # Libérer le port si occupé
    os.system("pkill -f uvicorn 2>/dev/null || true")
    os.system("fuser -k 8000/tcp 2>/dev/null || true")
    time.sleep(2)

    # Démarrer uvicorn en arrière-plan
    threading.Thread(
        target=lambda: uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings.PORT,
            log_level="info",
        ),
        daemon=True,
    ).start()
    time.sleep(3)

    # Tunnel ngrok
    tunnel = ngrok.connect(settings.PORT)
    public_url = tunnel.public_url

    print("\n" + "=" * 60)
    print(f"  ScolarAI API v6.0 — {settings.DEVICE.upper()}")
    print(f"  URL publique : {public_url}")
    print("=" * 60)
    print(f"  GET  {public_url}/status")
    print(f"  POST {public_url}/detect       ← YOLO OBB")
    print(f"  POST {public_url}/ocr          ← TrOCR")
    print(f"  POST {public_url}/save         ← Corrections Drive")
    print(f"  GET  {public_url}/export/json  ← Export JSON")
    print(f"  GET  {public_url}/export/csv   ← Export CSV")
    print(f"  GET  {public_url}/export/txt   ← Export TXT")
    print(f"  GET  {public_url}/export/stats ← Statistiques")
    print(f"  POST {public_url}/retrain      ← Réentraîner TrOCR")
    print("=" * 60)
    print(f'\n  apiUrl = "{public_url}"')
    print("=" * 60)

    # Boucle keepalive
    from core.dataset import count_corrections
    while True:
        time.sleep(60)
        print(f"[alive] {settings.DEVICE} | corrections: {count_corrections()}")


if __name__ == "__main__":
    main()
