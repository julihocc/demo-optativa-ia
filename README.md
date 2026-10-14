# Demo: reconocedor de dígitos

Dibuja un dígito con el ratón y la barra lateral muestra la probabilidad de cada dígito (0–9).

```bash
pip install -r requirements.txt
python train/train.py                 # entrena (MNIST) y genera web/model/weights.json
python -m http.server -d web 8000     # servir en local
```
Abrir http://localhost:8000. La inferencia corre en el navegador (JS puro, sin dependencias).
