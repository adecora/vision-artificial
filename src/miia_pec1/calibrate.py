import os
import platform
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

# Criterio de terminación para el algoritmo de detección de esquinas de terminación
# finaliza cuando: haya realizado 30 iteraciones o la posición estimada cambia menos de 0.001 píxeles entre iteracciones
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FONT_FACE = cv.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Específico para mi configuración en Arch Linux
if platform.system() == "Linux":
    info = platform.freedesktop_os_release()
    if info.get("NAME") == "Arch Linux":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
        os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"


def capture(device, output_dir, rows, cols):
    """
    Función para capturar imágenes de calibración utilizando la cámara.

    Parámetros:
    ===========
    device: Dispositivo de la cámara (por ejemplo, 0 para la cámara predeterminada)
    output_dir: Directorio donde se guardarán las imágenes capturadas
    rows: Número de filas del tablero de ajedrez
    cols: Número de columnas del tablero de ajedrez

    Returns:
    ========
    int: Código de salida (0 si se completó correctamente)
    """
    cap = cv.VideoCapture(device)
    if not cap.isOpened():
        print(
            f"No se pudo abrir la cámara con el dispositivo {device}.", file=sys.stderr
        )
        raise SystemExit(1)

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar la imagen.")
                break

            h, w = frame.shape[:2]
            img = frame.copy()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                # Dibuja y muestra las esquinas detectadas
                cv.drawChessboardCorners(img, (cols, rows), corners, ret)

            cv.putText(
                img,
                "Presiona 'c' para capturar, 'q' para salir",
                (10, h - 10),
                FONT_FACE,
                FONT_SIZE,
                GREEN,
                1,
            )
            if ret:
                cv.putText(
                    img,
                    "Patrón detectado",
                    (10, 30),
                    FONT_FACE,
                    FONT_SIZE,
                    GREEN,
                    1,
                )
            else:
                cv.putText(
                    img,
                    "Patrón no detectado",
                    (10, 30),
                    FONT_FACE,
                    FONT_SIZE,
                    RED,
                    1,
                )
            cv.imshow("Captura de Imagen", img)

            key = cv.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("c"):
                if ret:
                    output_file = output_dir / f"capture_{idx:02}.jpg"
                    cv.imwrite(str(output_file), frame)
                    print(f"Imagen guardada en {output_file}")
                    idx += 1
                else:
                    print("Captura fallida: patrón no detectado.")
    finally:
        cap.release()
        cv.destroyAllWindows()

    return 0


def calibrate(rows, cols, image_dir, outfile):
    """
    Función para calibrar la cámara utilizando un conjunto de imágenes con un patrón de tablero de ajedrez.

    Parámetros:
    ===========
    rows: Número de filas del tablero de ajedrez
    cols: Número de columnas del tablero de ajedrez
    image_dir: Directorio que contiene las imágenes del tablero de ajedrez
    outfile: Archivo donde se guardarán los parámetros de calibración

    Returns:
    ========
    int: Código de salida (0 si se completó correctamente)
    """
    # preparación de los puntos del objeto, como (0,0,0), (1,0,0), (2,0,0) ....,(cols-1,rows-1,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays para guardar los puntos del objeto y los puntos de la imagen
    objpoints = []  # puntos 3D en el espacio del mundo real
    imgpoints = []  # puntos 2D en el plano de la imagen
    imgnames = []  # nombres de las imágenes procesadas

    images = sorted(image_dir.glob("*.jpg"))

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Encuentra las esquinas del tablero de ajedrez
        ret, corners = get_corner_locations(gray, pattern_size=(cols, rows))

        # Si se encuentran, agrega los puntos del objeto y los puntos de la imagen
        if ret:
            imgnames.append(str(fname))
            objpoints.append(objp)
            imgpoints.append(corners)

            # Dibuja y muestra las esquinas detectadas
            cv.drawChessboardCorners(img, (cols, rows), corners, ret)
            cv.imshow("Esquinas del Tablero", img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    # Calibración de la cámara
    rms, mtx, dist, rvects, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Cálculo del error de calibración por imagen
    errors = []
    for obj, img, rvec, tvec, name in zip(
        objpoints, imgpoints, rvects, tvecs, imgnames
    ):
        projected, _ = cv.projectPoints(obj, rvec, tvec, mtx, dist)
        distances = projected.reshape(-1, 2) - img.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(distances**2, axis=1)))))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        outfile,
        corners=imgpoints,
        size=[cols, rows],
        fnames=imgnames,
        rms=rms,
        mtx=mtx,
        dist=dist,
        rvects=rvects,
        tvecs=tvecs,
        errors=errors,
    )

    return 0


def get_corner_locations(img, *, pattern_size=(7, 6), winSize=(11, 11)):
    """
    Función auxiliar para obtener las ubicaciones de las esquinas del tablero de ajedrez en una imagen.

    Parámetros:
    ===========
    img: Imagen que contiene el tablero de ajedrez

    Returns:
    ========
    ret: Booleano que indica si se encontraron las esquinas
    corners: Coordenadas de las esquinas detectadas (refinadas a subpíxel si se encontraron)
    """
    # Encuentra las esquinas del tablero de ajedrez
    ret, corners = cv.findChessboardCorners(img, pattern_size, None)
    if ret:
        # Refina las esquinas detectadas a subpíxel de precisión
        corners2 = cv.cornerSubPix(img, corners, winSize, (-1, -1), CRITERIA)
        return ret, corners2
    return ret, corners


def draw(img, corners, imgpts):
    """
    Dibuja un sistema de coordenadas 3D en la imagen.

    Parámetros:
    ===========
    img: Imagen en la que se dibujará
    corners: Esquinas detectadas del tablero de ajedrez
    imgpts: Puntos proyectados de los ejes 3D

    Returns:
    ========
    img: Imagen con los ejes dibujados
    """
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), RED, 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), BLUE, 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), GREEN, 5)
    return img


def draw_cube(img, corners, imgpts):
    """
    Dibuja un cubo 3D en la imagen.

    Parámetros:
    ===========
    img: Imagen en la que se dibujará
    corners: Esquinas detectadas del tablero de ajedrez
    imgpts: Puntos proyectados de los vértices del cubo

    Returns:
    ========
    img: Imagen con el cubo dibujado
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, GREEN, -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), BLUE, 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, RED, 3)

    return img
