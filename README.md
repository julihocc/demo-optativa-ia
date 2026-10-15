# Demo: reconocedores de dígitos y emojis

Dibuja con el ratón y la barra lateral muestra la probabilidad de cada clase.

- **Dígitos** (0–9): MLP entrenado con MNIST.
- **Emojis** (😀 ☀️ 🌙 ⭐ ☁️ ⚡ ❄️ ☂️ 🌈 🍕): MLP entrenado con dibujos de [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset).

```bash
pip install -r requirements.txt
python train/train.py                 # entrena (MNIST) y genera web/model/weights.json
python train/train_emoji.py           # descarga Quick, Draw! (~78 MB en data/) y genera web/emoji/model/weights.json
python -m http.server -d web 8000     # servir en local
```

Abrir <http://localhost:8000> (dígitos) o <http://localhost:8000/emoji/> (emojis). La inferencia corre en el navegador (JS puro, sin dependencias).
