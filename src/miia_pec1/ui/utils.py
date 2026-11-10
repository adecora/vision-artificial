import cv2 as cv
import numpy as np
import pandas as pd


def is_gray(img):
    """
    Verifica si una imagen es en escala de grises.

    Parámetros:
    ===========
    img: Imagen de entrada (numpy.ndarray)

    Retorna:
    ===========
    bool: True si la imagen es en escala de grises, False en caso contrario
    """
    if not isinstance(img, np.ndarray):
        return False
    if img.ndim == 2:
        return True
    if img.ndim != 3 or img.shape[2] == 1:
        return img.ndim == 3

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    if np.array_equal(R, G) and np.array_equal(R, B):
        return True
    return False


def plot_histogram(img):
    """
    Calcula y devuelve el histograma de una imagen.

    Parámetros:
    ===========
    img: Imagen de entrada (numpy.ndarray)

    Retorna:
    ===========
    numpy.ndarray: Histograma de la imagen
    """
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        return None
    if img.ndim == 2:
        canales = [("Gris", img)]
    elif is_gray(img):
        canales = [("Gris", img[:, :, 0])]
    else:
        canales = [
            ("Rojo", img[:, :, 0]),
            ("Verde", img[:, :, 1]),
            ("Azul", img[:, :, 2]),
        ]

    histograms = []
    for nombre, canal in canales:
        hist = cv.calcHist([canal], [0], None, [256], [0, 256])
        histograms.append(
            pd.DataFrame(
                {
                    "Intensidad": list(range(256)),
                    "Frecuencia": hist.flatten(),
                    "Canal": nombre,
                }
            )
        )

    return pd.concat(histograms, ignore_index=True)


def roi_bounds(image, xmin, xmax, ymin, ymax):
    """
    Calcula los límites de la región de interés (ROI) en una imagen.

    Parámetros:
    ===========
    image: Imagen de entrada
    xmin: Porcentaje mínimo en X
    xmax: Porcentaje máximo en X
    ymin: Porcentaje mínimo en Y
    ymax: Porcentaje máximo en Y

    Retorna:
    ===========
    tuple: Coordenadas (x0, y0, x1, y1) de la ROI o None si la ROI es inválida
    """
    height, width = image.shape[:2]
    x0, x1 = sorted((round(width * xmin / 100), round(width * xmax / 100)))
    y0, y1 = sorted((round(height * ymin / 100), round(height * ymax / 100)))
    if x0 == x1 or y0 == y1:
        return None
    return x0, y0, x1, y1


def crop_roi(image, xmin, xmax, ymin, ymax):
    """
    Recorta la región de interés (ROI) de una imagen.

    Parámetros:
    ===========
    image: Imagen de entrada
    xmin: Porcentaje mínimo en X
    xmax: Porcentaje máximo en X
    ymin: Porcentaje mínimo en Y
    ymax: Porcentaje máximo en Y

    Retorna:
    ===========
    numpy.ndarray: Imagen recortada o None si la ROI es inválida
    """
    bounds = roi_bounds(image, xmin, xmax, ymin, ymax)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1].copy()
