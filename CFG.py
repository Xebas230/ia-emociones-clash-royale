import os

# Leer API key desde variable de entorno si está definida.
# Si no está definida, intentar cargar desde el archivo local `Api_key.py`
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    try:
        from Api_key import API_KEY as _API_KEY_FILE
        API_KEY = _API_KEY_FILE
    except Exception:
        API_KEY = None

MODEL_PATH = r"runs/detect/train7/weights/best.pt"
emoji_to_path = {
    "Anticipation": "imgs/Anticipation.png",
    "Kiss": "imgs/Kiss.png",
    "Laugh": "imgs/Laugh.png",
    "Sleepy": "imgs/Sleepy.png",
    "Tongue": "imgs/Tongue.png",
}
emoji_resolution = {
    'x': 512,
    'y': 512,
}
