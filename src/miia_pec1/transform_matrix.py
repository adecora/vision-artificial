import cv2 as cv
import numpy as np

TRANSLATE = lambda t_x, t_y: np.array([[1, 0, t_x], [0, 1, t_y], [0, 0, 1]])
ROTATE = lambda degrees: np.array(
    [
        [np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees)), 0],
        [-np.sin(np.deg2rad(degrees)), np.cos(np.deg2rad(degrees)), 0],
        [0, 0, 1],
    ]
)
SCALE = lambda f_x, f_y: np.array([[f_x, 0, 0], [0, f_y, 0], [0, 0, 1]])
SHEAR_X = lambda degrees: np.array(
    [[1, np.tan(np.deg2rad(degrees)), 0], [0, 1, 0], [0, 0, 1]]
)
SHEAR_Y = lambda degrees: np.array(
    [[1, 0, 0], [np.tan(np.deg2rad(degrees)), 1, 0], [0, 0, 1]]
)


def rotate_bound(image, angle):
    """
    Función auxiliar para adapatar las dimesiones de salida
    al rotar las imagenes
    Ver: https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    Parámetros:
    ===========
    image: Imagen a rotar
    angle: Ángulo de rotación en grados

    Returns:
    ========
    np.ndarray: Imagen rotada en sentido horario con dimensiones ajustadas
    """
    # Centro de la imagen original
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    # Generar la matriz de rotación
    # Ver: https://docs.opencv.org/5.0/main_modules/geometry_shape.html#getrotationmatrix2d
    # (ángulo negativo para hacer el giro en sentido horario)
    M = cv.getRotationMatrix2D((cX, cY), -angle, scale=1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Cálculo de las nuevas dimensiones de la imagen
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Ajusta el centro del giro al centro de las nuevas dimensiones de la imagen
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Aplica la trasnformación
    return cv.warpAffine(image, M, (nW, nH))
