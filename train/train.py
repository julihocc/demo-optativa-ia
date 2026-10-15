"""Entrena un MLP sobre dibujos de Quick, Draw! y exporta los pesos a web/model/weights.json."""
import urllib.parse
import urllib.request

import numpy as np

from common import ROOT, fit_and_export

# (categoria de Quick, Draw!, emoji, nombre, nivel de dificultad en el juego segun la complejidad del dibujo)
CATEGORIES = [
    ("smiley face", "😀", "cara sonriente", 3),
    ("sun", "☀️", "sol", 2),
    ("moon", "🌙", "luna", 1),
    ("star", "⭐", "estrella", 1),
    ("cloud", "☁️", "nube", 1),
    ("lightning", "⚡", "rayo", 1),
    ("snowflake", "❄️", "copo de nieve", 3),
    ("umbrella", "☂️", "paraguas", 2),
    ("rainbow", "🌈", "arcoíris", 2),
    ("pizza", "🍕", "pizza", 3),
]
N = 10000  # dibujos por categoria
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
CACHE = ROOT / "data" / "quickdraw"


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def load(cat):
    """Descarga solo las primeras N filas del .npy (peticion HTTP Range) y las guarda en cache."""
    path = CACHE / f"{cat.replace(' ', '_')}_{N}.npy"
    if path.exists():
        return np.load(path)
    url = URL.format(urllib.parse.quote(cat))
    head = fetch_range(url, 0, 1023)
    header_len = 10 + int.from_bytes(head[8:10], "little")  # formato .npy v1
    raw = fetch_range(url, header_len, header_len + N * 784 - 1)
    X = np.frombuffer(raw, dtype=np.uint8).reshape(N, 784)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    print("Descargado", cat)
    return X


if __name__ == "__main__":
    X = np.concatenate([load(c) for c, *_ in CATEGORIES]).astype(np.float32) / 255.0
    y = np.repeat(np.arange(len(CATEGORIES)), N)
    labels = [{"emoji": e, "nombre": n, "nivel": lvl} for _, e, n, lvl in CATEGORIES]
    fit_and_export(X, y, ROOT / "web" / "model" / "weights.json", test_size=0.1, labels=labels)
