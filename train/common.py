"""Codigo compartido por los scripts de entrenamiento: ajuste del MLP y exportacion a JSON."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent


def export_mlp(clf, out, labels=None):
    """Guarda pesos y sesgos (redondeados) de un MLPClassifier como JSON para la web."""
    layers = [
        {"W": np.round(W, 5).tolist(), "b": np.round(b, 5).tolist()}
        for W, b in zip(clf.coefs_, clf.intercepts_)
    ]
    data = {"layers": layers} if labels is None else {"labels": labels, "layers": layers}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("Pesos exportados en", out)


def fit_and_export(X, y, out, test_size, labels=None):
    """Entrena un MLP (128, 64), muestra la precision en test y exporta los pesos a `out`."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, random_state=0, verbose=True)
    clf.fit(Xtr, ytr)
    print(f"Precision en test: {clf.score(Xte, yte):.4f}")
    export_mlp(clf, out, labels)
