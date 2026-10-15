"""Entrena un MLP sobre MNIST y exporta los pesos a web/model/weights.json."""
import numpy as np
from sklearn.datasets import fetch_openml

from common import ROOT, fit_and_export

if __name__ == "__main__":
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)
    fit_and_export(X, y, ROOT / "web" / "model" / "weights.json", test_size=10000)
