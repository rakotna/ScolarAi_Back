# ScolarAI API v6.0

## 📋 Description

**ScolarAI** est une API backend Python haute performance dédiée à l'automatisation de la reconnaissance optique de caractères (OCR) sur les bulletins scolaires. Elle utilise des modèles d'intelligence artificielle de pointe pour détecter et extraire les informations textuelles des documents scolaires avec une grande précision.

### Technologies Clés
- **YOLO OBB** : Détection orientée des zones de texte
- **TrOCR** : Reconnaissance optique de caractères transformateur
- **FastAPI** : Framework web async haute performance
- **Ngrok** : Tunnel public pour accès à distance

---

## 🎯 Fonctionnalités Principales

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/status` | GET | Vérifier l'état de l'API |
| `/detect` | POST | Détection YOLO OBB des zones de texte |
| `/ocr` | POST | Reconnaissance optique (TrOCR) |
| `/save` | POST | Sauvegarde des corrections sur Google Drive |
| `/export/json` | GET | Export des données en JSON |
| `/export/csv` | GET | Export des données en CSV |
| `/export/txt` | GET | Export des données en TXT |
| `/export/stats` | GET | Statistiques d'utilisation |
| `/retrain` | POST | Réentraînement du modèle TrOCR |

---

## 🚀 Installation et Configuration

### Prérequis
- Python 3.8+
- pip ou conda
- Jeton ngrok (pour tunnelage public)

### Installation

```bash
# Cloner le repository
git clone https://github.com/rakotna/ScolarAi_Back.git
cd ScolarAi_Back

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

Créer un fichier `.env` à la racine du projet :

```env
NGROK_TOKEN=your_ngrok_token_here
PORT=8000
DEVICE=gpu  # ou 'cpu' selon votre système
```

Les paramètres sont gérés via `config/settings.py`.

---

## 📦 Structure du Projet

```
ScolarAi_Back/
├── main.py                 # Point d'entrée principal
├── app.py                  # Factory FastAPI
├── config/
│   └── settings.py        # Configuration centralisée
├── core/
│   ├── models.py          # Chargement des modèles IA
│   └── dataset.py         # Gestion des jeux de données
├── routes/
│   ├── status.py          # Endpoint /status
│   ├── detect.py          # Endpoint /detect (YOLO OBB)
│   ├── ocr.py             # Endpoint /ocr (TrOCR)
│   ├── save.py            # Endpoint /save (Google Drive)
│   ├── export.py          # Endpoints /export/*
│   └── retrain.py         # Endpoint /retrain
└── utils/                 # Utilitaires et helpers
```

---

## 🔧 Démarrage de l'API

### Mode Développement

```bash
python main.py
```

L'API démarrera automatiquement :
- Serveur local : `http://localhost:8000`
- URL publique via ngrok : affichée dans le terminal

### Mode Production

Adapter les paramètres dans `.env` et déployer avec un gestionnaire de processus (gunicorn, supervisord, etc.).

---

## 📝 Exemples d'Utilisation

### 1. Vérifier l'état de l'API

```bash
curl http://localhost:8000/status
```

### 2. Détecter des zones de texte

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d @image_data.json
```

### 3. Reconnaissance OCR

```bash
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d @image_data.json
```

### 4. Exporter les résultats

```bash
# Exporter en JSON
curl http://localhost:8000/export/json > results.json

# Exporter en CSV
curl http://localhost:8000/export/csv > results.csv

# Afficher les statistiques
curl http://localhost:8000/export/stats
```

---

## 🔐 CORS et Sécurité

⚠️ **Attention** : Actuellement, CORS est configuré avec `allow_origins=["*"]` pour le développement.

**En production**, restreindre les origines autorisées :

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 🤖 Modèles IA

### YOLO OBB (You Only Look Once - Oriented Bounding Box)
- Détecte et localise les zones de texte orientées
- Entraîné sur des images de bulletins scolaires
- Sortie : coordonnées et angles des boîtes de détection

### TrOCR (Transformer-based OCR)
- Reconnaissance optique basée sur transformeurs
- Traitement fin : peut être réentraîné avec des données custom
- Endpoint de réentraînement : `POST /retrain`

---

## 📊 Monitoring

L'API inclut un système de keepalive qui affiche toutes les 60 secondes :
- État du système (GPU/CPU)
- Nombre de corrections/prédictions effectuées

```
[alive] gpu | corrections: 42
```

---

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| Port 8000 déjà utilisé | API libère automatiquement le port au démarrage |
| Jeton ngrok invalide | Vérifier `NGROK_TOKEN` dans `.env` |
| Modèles non chargés | Vérifier `core/models.py` et les chemins des modèles |
| Erreur CUDA | Basculer vers CPU : `DEVICE=cpu` dans `.env` |

---

## 📚 Documentation API

Une documentation interactive est disponible via Swagger UI :
```
http://localhost:8000/docs
```

---

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

---

## 📄 Licence

Ce projet est sous licence propriétaire. Pour plus d'informations, voir le fichier LICENSE.

---

## 📞 Support et Contact

Pour des questions ou des problèmes :
- 📧 Email : [contact@example.com]
- 🐛 Issues : [Ouvrir une issue](https://github.com/rakotna/ScolarAi_Back/issues)
- 📖 Wiki : [GitHub Wiki](https://github.com/rakotna/ScolarAi_Back/wiki)

---

## 🎓 Version

**Version actuelle** : 6.0  
**Dernière mise à jour** : Juin 2026

---

**Développé avec ❤️ pour l'automatisation intelligente des processus éducatifs**
