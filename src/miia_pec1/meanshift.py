import sys

import cv2 as cv
import numpy as np

# Criterio de terminación para el algoritmo meanshift
# finaliza cuando: haya alcanzado 10 iteraciones o el desplzamiento de la ventana sea menor a 1 punto
CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
FONT_FACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def setup_roi(img_ref, *, track_window, mask_min, mask_max):
    """
    Genera el modelo de color de la región de interés (ROI) para el seguimiento con meanshift.

    Parámetros:
    ===========
    img_ref: Ruta a la imagen de referencia
    track_window: Ventana de seguimiento en formato (x, y, width, height)
    mask_min: Valor mínimo del rango de la máscara HSV en el formato (H, S, V)
    mask_max: Valor máximo del rango de la máscara HSV en el formato (H, S, V)

    Returns:
    ========
    roi_hist: Histograma normalizado del canal H de la ROI.
    """
    x, y, width, height = track_window
    frame = cv.imread(img_ref)
    roi = frame[y : y + height, x : x + width]

    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array(mask_min), np.array(mask_max))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    return roi_hist


def meanshift_tracking(
    device, roi_hist, track_window, *, term_crit=CRITERIA, outfile=None
):
    """
    Realiza el seguimiento de un objeto utilizando el algoritmo meanshift.

    Parámetros:
    ===========
    device: Dispositivo de la cámara (por ejemplo, 0 para la cámara predeterminada)
    roi_hist: Histograma normalizado del canal H de la ROI
    track_window: Ventana de seguimiento en formato (x, y, width, height)
    term_crit: Criterio de terminación para el algoritmo meanshift
    outfile: Fichero donde se guarda el vídeo de seguimiento (opcional)

    Returns:
    ========
    None
    """
    cap = cv.VideoCapture(device)
    if not cap.isOpened():
        print(
            f"No se pudo abrir la cámara con el dispositivo {device}.", file=sys.stderr
        )
        raise SystemExit(1)
    if outfile is not None:
        # Crear el directorio padre si no existe
        outfile.parent.mkdir(parents=True, exist_ok=True)
        # Añadir la extensión
        outfile = outfile.with_suffix(".mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0
        out = cv.VideoWriter(outfile, fourcc, fps, (w, h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            _, track_window = cv.meanShift(dst, track_window, term_crit)

            # Draw it on image
            x, y, w, h = track_window
            img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv.putText(
                img2,
                "Presiona 'ESC' para salir",
                (10, h - 10),
                FONT_FACE,
                FONT_SIZE,
                GREEN,
                1,
            )
            cv.imshow("img2", img2)

            if outfile is not None:
                out.write(frame)

            # La tecla 'Esc' termina el seguimiento
            if cv.waitKey(30) & 0xFF == 27:
                break

    finally:
        cap.release()
        if outfile is not None:
            out.release()
        cv.destroyAllWindows()
