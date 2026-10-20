import argparse
from pathlib import Path

from .calibrate import capture, calibrate


def main():
    """Herramienta de línea de comandos para calibrar la cámara utilizando imágenes de un tablero de ajedrez."""
    parser = argparse.ArgumentParser(
        prog="miia-pec1",
        description="Calibrar la cámara utilizando imágenes de un tablero de ajedrez.",
        epilog="Ejemplo de uso:\n calibrate --rows 6 --cols 7 --image_dir /ruta/a/imagenes",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponibles")

    # Opciones compartidas
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--rows", type=int, default=6, help="Número de filas del tablero de ajedrez."
    )
    common_parser.add_argument(
        "--cols", type=int, default=7, help="Número de columnas del tablero de ajedrez."
    )

    # Comando de calibración
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        parents=[common_parser],
        help="Calibrar la cámara utilizando imágenes de un tablero de ajedrez.",
    )
    calibrate_parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directorio que contiene las imágenes de calibración.",
    )
    calibrate_parser.add_argument(
        "--outfile",
        type=Path,
        default="calibration.npz",
        help="Directorio donde se guardarán los resultados de la calibración.",
    )

    # Comando para capturar imágenes de calibración (aún no implementado)
    capture_parser = subparsers.add_parser(
        "capture",
        parents=[common_parser],
        help="Capturar imágenes de calibración utilizando la cámara.",
    )
    capture_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID del dispositivo de la cámara a utilizar.",
    )
    capture_parser.add_argument(
        "--output",
        type=Path,
        default=".",
        help="Directorio donde se guardarán las imágenes capturadas.",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "calibrate":
        if args.image_dir.exists() is False:
            print(f"El directorio {args.image_dir} no existe.")
            raise SystemExit(1)

        calibrate(
            rows=args.rows,
            cols=args.cols,
            image_dir=args.image_dir,
            outfile=args.outfile,
        )
    elif args.command == "capture":
        capture(
            device=args.device, rows=args.rows, cols=args.cols, output_dir=args.output
        )
    else:
        parser.print_help()
        raise SystemExit(1)

