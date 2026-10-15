"""Entrena un MLP sobre dibujos de Quick, Draw! y exporta los pesos a web/emoji/model/weights.json."""
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# (categoria de Quick, Draw!, emoji, nombre)
CATEGORIES = [
    ("smiley face", "😀", "cara sonriente"),
    ("sun", "☀️", "sol"),
    ("moon", "🌙", "luna"),
    ("star", "⭐", "estrella"),
    ("cloud", "☁️", "nube"),
    ("lightning", "⚡", "rayo"),
    ("snowflake", "❄️", "copo de nieve"),
    ("umbrella", "☂️", "paraguas"),
    ("rainbow", "🌈", "arcoíris"),
    ("pizza", "🍕", "pizza"),
]
N = 10000  # dibujos por categoria
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
ROOT = Path(__file__).resolve().parent.parent
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
    X = np.concatenate([load(c) for c, _, _ in CATEGORIES]).astype(np.float32) / 255.0
    y = np.repeat(np.arange(len(CATEGORIES)), N)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=0)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, random_state=0, verbose=True)
    clf.fit(Xtr, ytr)
    print(f"Precision en test: {clf.score(Xte, yte):.4f}")

    out = ROOT / "web" / "emoji" / "model" / "weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    layers = [
        {"W": np.round(W, 5).tolist(), "b": np.round(b, 5).tolist()}
        for W, b in zip(clf.coefs_, clf.intercepts_)
    ]
    labels = [{"emoji": e, "nombre": n} for _, e, n in CATEGORIES]
    out.write_text(json.dumps({"labels": labels, "layers": layers}, ensure_ascii=False), encoding="utf-8")
    print("Pesos exportados en", out)
