# Demo: reconocedores de dígitos y emojis

Dibuja con el ratón y la barra lateral muestra la probabilidad de cada clase.

- **Dígitos** (0–9): MLP entrenado con MNIST.
- **Emojis** (😀 ☀️ 🌙 ⭐ ☁️ ⚡ ❄️ ☂️ 🌈 🍕): MLP entrenado con dibujos de [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset).

## Arranque

```bash
python run.py
```

Sirve la web en <http://localhost:8000> (dígitos) y <http://localhost:8000/emoji/> (emojis) y abre el navegador.
Los pesos entrenados ya están en el repositorio, así que no hace falta instalar nada más. Si falta algún
`weights.json`, `run.py` instala las dependencias (`requirements.txt`) y entrena solo ese modelo.

| Opción         | Efecto                                                           |
| -------------- | ---------------------------------------------------------------- |
| `--train`      | reentrena ambos modelos aunque ya existan los pesos              |
| `--port N`     | puerto del servidor (por defecto 8000)                           |
| `--no-browser` | no abre el navegador                                             |

## Entrenar a mano

```bash
pip install -r requirements.txt
python train/train.py          # MNIST -> web/model/weights.json
python train/train_emoji.py    # Quick, Draw! (~78 MB en data/) -> web/emoji/model/weights.json
```

La inferencia corre en el navegador (JS puro, sin dependencias); el código común está en `web/common.js`
(modelo, barras, lienzo) y `train/common.py` (entrenamiento y exportación).
