"""Entrena un MLP sobre MNIST y exporta los pesos a web/model/weights.json."""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0
y = y.astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=10000, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, random_state=0, verbose=True)
clf.fit(Xtr, ytr)
print(f"Precision en test: {clf.score(Xte, yte):.4f}")

out = Path(__file__).resolve().parent.parent / "web" / "model" / "weights.json"
out.parent.mkdir(parents=True, exist_ok=True)
layers = [
    {"W": np.round(W, 5).tolist(), "b": np.round(b, 5).tolist()}
    for W, b in zip(clf.coefs_, clf.intercepts_)
]
out.write_text(json.dumps({"layers": layers}))
print("Pesos exportados en", out)
