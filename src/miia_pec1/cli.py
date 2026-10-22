import argparse
from pathlib import Path

from .calibrate import calibrate, capture
from .meanshift import meanshift_tracking, setup_roi


def main():
    """Herramienta de línea de comandos para calibrar la cámara utilizando imágenes de un tablero de ajedrez."""
    parser = argparse.ArgumentParser(
        prog="miia-pec1",
        description="Calibrar la cámara utilizando imágenes de un tablero de ajedrez.",
        epilog="Ejemplo de uso:\n calibrate --rows 6 --cols 7 --image_dir /ruta/a/imagenes",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcomandos disponibles")

    # Opciones compartidas
    common_parser_1 = argparse.ArgumentParser(add_help=False)
    common_parser_1.add_argument(
        "-r",
        "--rows",
        type=int,
        default=6,
        help="Número de filas del tablero de ajedrez.",
    )
    common_parser_1.add_argument(
        "-c",
        "--cols",
        type=int,
        default=7,
        help="Número de columnas del tablero de ajedrez.",
    )

    common_parser_2 = argparse.ArgumentParser(add_help=False)
    common_parser_2.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="ID del dispositivo de la cámara a utilizar.",
    )

    # Comando de calibración
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        parents=[common_parser_1],
        help="Calibrar la cámara utilizando imágenes de un tablero de ajedrez.",
    )
    calibrate_parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        required=True,
        help="Directorio que contiene las imágenes de calibración.",
    )
    calibrate_parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default="calibration.npz",
        help="Fichero donde se guardan los resultados de la calibración.",
    )

    # Comando para capturar imágenes de calibración
    capture_parser = subparsers.add_parser(
        "capture",
        parents=[common_parser_1, common_parser_2],
        help="Capturar imágenes de calibración utilizando la cámara.",
    )
    capture_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=".",
        help="Directorio donde se guardan las imágenes capturadas.",
    )

    # Comando para seguimiento con meanshift
    meanshift_parser = subparsers.add_parser(
        "meanshift",
        parents=[common_parser_2],
        help="Seguimiento de objetos utilizando el algoritmo meanshift.",
    )
    meanshift_parser.add_argument(
        "-i",
        "--image-ref",
        type=Path,
        required=True,
        help="Imagen de referencia para el seguimiento.",
    )
    meanshift_parser.add_argument(
        "--track-window",
        type=str,
        required=True,
        help="Definición de la ventana de seguimiento en la imagen de referencia en el formato 'x,y,width,height'.",
    )
    meanshift_parser.add_argument(
        "--mask-min",
        type=str,
        required=True,
        help="Valor mínimo del rango de la máscara HSV. En el formato 'H,S,V'.",
    )
    meanshift_parser.add_argument(
        "--mask-max",
        type=str,
        required=True,
        help="Valor máximo del rango de la máscara HSV. En el formato 'H,S,V'.",
    )
    meanshift_parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default=None,
        help="Fichero donde se guarda el vídeo de seguimiento.",
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
    elif args.command == "meanshift":

        def validate_window(win_str):
            try:
                win = tuple(map(int, win_str.split(",")))
                if len(win) != 4:
                    raise ValueError
                return win
            except ValueError:
                print(
                    "La ventana de seguimiento debe estar en el formato 'x,y,width,height'."
                )
                raise SystemExit(1)

        def validate_mask(mask_str):
            try:
                mask = tuple(map(int, mask_str.split(",")))
                if len(mask) != 3:
                    raise ValueError
                return mask
            except ValueError:
                print("La máscara debe estar en el formato 'H,S,V'.")
                raise SystemExit(1)

        track_window = validate_window(args.track_window)
        mask_min = validate_mask(args.mask_min)
        mask_max = validate_mask(args.mask_max)

        roi_hist = setup_roi(
            args.image_ref,
            track_window=track_window,
            mask_min=mask_min,
            mask_max=mask_max,
        )
        meanshift_tracking(args.device, roi_hist, track_window, outfile=args.outfile)

    else:
        parser.print_help()
        raise SystemExit(1)
