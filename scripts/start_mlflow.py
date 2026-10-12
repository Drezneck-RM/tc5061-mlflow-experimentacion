"""Abrir el servidor local del proyecto; mantener esta terminal en ejecución."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("El puerto debe estar entre 1 y 65535.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Iniciar MLflow para consultar los experimentos de este proyecto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="Mostrar esta ayuda y salir.")
    parser.add_argument("--port", type=port_number, default=5001,
                        help="Puerto local de la interfaz MLflow.")
    args = parser.parse_args(argv)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    command = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///" + (ROOT / "mlruns.sqlite").as_posix(),
        "--default-artifact-root", artifacts.as_uri(),
        "--no-serve-artifacts", "--host", "127.0.0.1", "--port", str(args.port),
        "--workers", "1",
    ]
    print(f"MLflow local: http://127.0.0.1:{args.port}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
