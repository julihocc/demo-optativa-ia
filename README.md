# Demo: juego de dibujar emojis contra una red neuronal

Dibuja con el ratón el emoji que te pida la app. Un MLP entrenado con dibujos de
[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) (😀 ☀️ 🌙 ⭐ ☁️ ⚡ ❄️ ☂️ 🌈 🍕)
decide en tiempo real, en el navegador, si lo reconoce; la barra lateral muestra la probabilidad de cada clase.

## Cómo se juega

- Tienes **1 minuto** para acertar tantos emojis como puedas. Aciertas cuando la red da ≥ 60 % al emoji pedido.
- Hay **3 niveles** según la complejidad del dibujo; cada emoji da `10 × nivel` puntos y subes de nivel al llegar a 40 y 120 puntos:

| Nivel | Emojis                    |
| ----- | ------------------------- |
| 1     | ⭐ 🌙 ☁️ ⚡               |
| 2     | ☀️ ☂️ 🌈                  |
| 3     | 😀 ❄️ 🍕                  |

- **Saltar** cambia de emoji si te atascas; el récord se guarda en el navegador.

## Arranque

```bash
python run.py
```

Sirve la web en <http://localhost:8000> y abre el navegador.
Los pesos entrenados ya están en el repositorio, así que no hace falta instalar nada más. Si falta
`web/model/weights.json`, `run.py` instala las dependencias (`requirements.txt`) y entrena el modelo.

| Opción         | Efecto                                                           |
| -------------- | ---------------------------------------------------------------- |
| `--train`      | reentrena el modelo aunque ya existan los pesos                  |
| `--port N`     | puerto del servidor (por defecto 8000)                           |
| `--no-browser` | no abre el navegador                                             |

## Entrenar a mano

```bash
pip install -r requirements.txt
python train/train.py    # Quick, Draw! (~78 MB en data/) -> web/model/weights.json
```

La inferencia corre en el navegador (JS puro, sin dependencias): `web/common.js` (modelo, barras, lienzo),
`web/game.js` (reglas del juego) y `web/app.js`. El nivel de cada emoji se define en `train/train.py`
(`CATEGORIES`) y viaja en las etiquetas de `weights.json`.
