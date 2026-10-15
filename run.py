"""Arranca el juego: entrena el modelo si falta, sirve web/ y abre el navegador.

    python run.py                # usa los pesos existentes, entrena solo si faltan
    python run.py --train        # reentrena el modelo
    python run.py --port 9000 --no-browser
"""
import argparse
import functools
import importlib.util
import subprocess
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

# (nombre, script de entrenamiento, pesos que genera, ruta en la web)
MODELS = [
    ("emojis", ROOT / "train" / "train.py", WEB / "model" / "weights.json", "/"),
]


def ensure_dependencies():
    if all(importlib.util.find_spec(m) for m in ("numpy", "sklearn")):
        return
    print("Instalando dependencias (requirements.txt)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)


def train(force):
    pending = [(n, s) for n, s, w, _ in MODELS if force or not w.exists()]
    if not pending:
        return
    ensure_dependencies()
    for name, script in pending:
        print(f"Entrenando modelo de {name}...")
        subprocess.run([sys.executable, str(script)], check=True)


class Handler(SimpleHTTPRequestHandler):
    # En Windows el tipo MIME de .js sale del registro y puede ser text/plain,
    # lo que impide cargar modulos ES en el navegador.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # evita pesos antiguos tras reentrenar
        super().end_headers()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--train", action="store_true", help="reentrenar aunque ya existan los pesos")
    ap.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    args = ap.parse_args()

    train(args.train)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(WEB)))
    except OSError as e:
        sys.exit(f"No se pudo usar el puerto {args.port}: {e}. Prueba con --port <otro>.")
    base = f"http://localhost:{args.port}"
    for name, _, _, path in MODELS:
        print(f"  {name}: {base}{path}")
    print("Ctrl+C para salir.")
    if not args.no_browser:
        webbrowser.open(base + "/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
