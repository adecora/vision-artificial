from typing import no_type_check

import cv2 as cv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

RANDOM_COLORS = list(mcolors.CSS4_COLORS.keys())
GREEN = (0, 255, 0)


def plot_2D_transformations(x, x_t, *, ax=None, title=None, **kwargs):
    """
    Función auxiliar para representar transformaciones 2D.

    Parámetros:
    ===========
    x: Punto o conjunto de puntos 2D
    x_t: Punto o conjunto de puntos 2D transformados

    ax: Eje de Matplotlib donde se representará la transformación (opcional)
    title: Título del gráfico (opcional)
    **kwargs: Argumentos adicionales para la función de representación

    Returns:
    ========
    ax: Eje de Matplotlib donde se representó la transformación (devuelto para encadenamiento)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if title:
        ax.set_title(title, y=1.02)

    if len(x.shape) == 1:
        ax.scatter(x[0], x[1], marker="o", s=100, color="tab:blue", label="Original")
        ax.scatter(
            x_t[0], x_t[1], marker="o", s=100, color="tab:green", label="Transformado"
        )
    else:
        ax.add_patch(
            Polygon(
                x,
                closed=True,
                fill=False,
                linewidth=2,
                linestyle="--",
                color="tab:blue",
                label="Original",
            )
        )
        ax.add_patch(
            Polygon(
                x_t,
                closed=True,
                fill=False,
                linewidth=2,
                linestyle="--",
                color="tab:green",
                label="Transformado",
            )
        )

    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    if "x_lim" in kwargs:
        ax.set_xlim(kwargs["x_lim"])
    if "y_lim" in kwargs:
        ax.set_ylim(kwargs["y_lim"])

    ax.spines[["top", "right"]].set_visible(False)
    # Ver: https://python-course.eu/numerical-programming/spines-and-ticks-in-matplotlib.php
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))

    return ax


def plot_image(img, *, ax=None, title=None, axis=False, convert=None):
    """
    Función auxiliar para pintar imágenes.

    Parámetros:
    ===========
    img: Imagen

    ax: Eje de Matplotlib donde se representará la transformación (opcional)
    title: Título del gráfico (opcional)
    convert: Código conversión del espacio de colores de OpenCV
        Ver: https://docs.opencv.org/5.0/main_modules/imgproc_color_conversions.html#colorconversioncodes

    Returns:
    ========
    ax: Eje de Matplotlib donde se representó la imagen (devuelto para encadenamiento)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if title:
        ax.set_title(title)
    if convert:
        img = cv.cvtColor(img, convert)

    if len(img.shape) == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)

    if axis:
        ax.xaxis.tick_top()
        return ax

    ax.axis("off")
    return ax

@no_type_check
def plot_3d_gray(img):
    """
    Función auxiliar para pintar una imagen en 3D en escala de grises.
    Los valores de intensidad de la imagen se representan en el eje Z. Se invierte el eje Z para que los valores más altos
    simulen cuencas.

    Parámetros:
    ===========
    img: Imagen en escala de grises

    Returns:
    ========
    ax: Eje de Matplotlib donde se representó la imagen en 3D
    """
    fig, ax =  plt.subplots(subplot_kw=dict(projection='3d'))

    if img.ndim > 2:
        img = img[:, :, 0]

    h, w = img.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))


    hist = ax.plot_surface(xx, yy, img, cmap='coolwarm')
    ax.invert_zaxis()
    ax.set_xlabel(r'width')
    ax.set_ylabel(r'height')
    ax.set_zlabel(r'intensity')

    fig.colorbar(hist, ax=ax, shrink=0.5, aspect=5)

    return ax


def plot_list_images(images, *, shape=None, figsize=(15, 10)):
    """
    Función auxiliar para pintar una lista de imágenes.

    Parámetros:
    ===========
    images: Lista o tupla de imágenes a representar
    shape: Tupla (filas, columnas) que define la disposición de las subplots (opcional)
    figsize: Tamaño de la figura (opcional)

    Returns:
    ========
    axes: Array de ejes de Matplotlib donde se representaron las imágenes
    """
    if not isinstance(images, (list, tuple)):
        raise ValueError("images debe ser una lista o tupla de imágenes")

    if shape is None:
        shape = (1, len(images))

    fig, axes = plt.subplots(*shape, figsize=figsize)

    axes = np.array(axes).reshape(-1)
    for ax, img in zip(axes, images):
        plot_image(img, ax=ax)

    fig.tight_layout()
    return axes


def plot_arrow(ax, inicio, fin, *, lw=2, style="-|>", **kwargs):
    """
    Función auxiliar para pintar flechas en los gráficos.

    Parámetros:
    ===========
    ax: Eje de Matplotlib donde se representará la flecha
    inicio: Coordenadas del punto de inicio de la flecha
    fin: Coordenadas del punto final de la flecha
    lw: Ancho de línea de la flecha (opcional)
    style: Estilo de la flecha (opcional)
    **kwargs: Argumentos adicionales para la función de representación
    """
    ax.add_patch(
        FancyArrowPatch(
            inicio, fin, mutation_scale=20, linewidth=lw, arrowstyle=style, **kwargs
        )
    )


def draw_flow_hsv(flow):
    """
    Dibuja el flujo óptico en formato HSV.
    El ángulo del flujo se codifica en el canal H y la magnitud en el canal V.

    Parámetros:
    ===========
    flow: Flujo óptico calculado con OpenCV

    Returns:
    ========
    Imagen en formato BGR que representa el flujo óptico
    """
    h, w = flow.shape[:2]
    ch = 3
    hsv = np.zeros((h, w, ch), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    brg = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return brg


def draw_flow(img, flow, *, step=16):
    """
    Dibuja el flujo óptico sobre la imagen original.

    Parámetros:
    ===========
    img: Imagen original sobre la que se dibujará el flujo
    flow: Flujo óptico calculado con OpenCV
    step: Paso para muestrear el flujo (opcional)
        Determina la densidad de las flechas que se dibujarán sobre la imagen

    Returns:
    ========
    Imagen con el flujo óptico representado
    """
    h, w = img.shape[:2]

    y, x = (
        np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
    )
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img = cv.polylines(img, lines, 0, GREEN)
    for (x1, y1), (x2, y2) in lines:
        img = cv.circle(img, (x1, y1), 1, GREEN, -1)

    return img


def draw_flow_warp(img, flow):
    """
    Aproxima el fotograma actual deformando el fotograma anterior
    mediante el campo de flujo óptico.

    Parámetros:
    ===========
    img: Fotograma anterior de la secuencia
    flow: Flujo óptico calculado con OpenCV

    Returns:
    ========
    Aproximación del fotograma actual obtenida mediante mapeo inverso del flujo óptico
    """
    h, w = img.shape[:2]

    row_coords, col_coords = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Aproximación al mapa inverso.
    # Para cada píxel de destino, se estima de qué posición del fotograma anterior debe obtenerse su intensidad.
    map_x = col_coords - u
    map_y = row_coords - v
    # Aproximación del fotograma actual
    img_warp = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

    return img_warp


def draw_flow_registration(gray, old_gray, flow):
    """
    Dibuja el flujo óptico aplicado a la imagen gris y genera secuencias de imágenes para visualización.

    Parámetros:
    ===========
    gray: Imagen gris actual
    old_gray: Imagen gris anterior

    Returns:
    ========
    seq_img: Superposición RGB de los fotogramas sin registrar
    reg_img: Superposición RGB después de compensar el movimiento
    target_img: Fotograma de referencia representado en RGB
    """
    h, w = gray.shape

    row_coords, col_coords = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Aproximación al mapa inverso.
    # Para cada píxel de destino, se estima de qué posición del fotograma anterior debe obtenerse su intensidad.
    map_x = col_coords + u
    map_y = row_coords + v
    # Aproximación del fotograma anterior
    gray_warp = cv.remap(gray, map_x, map_y, cv.INTER_LINEAR)

    # Superposición RGB de los fotogramas sin registrar
    seq_img = np.zeros((h, w, 3), dtype=np.uint8)
    seq_img[..., 0] = gray
    seq_img[..., 1] = old_gray
    seq_img[..., 2] = old_gray

    # Superposición RGB después de compensar el movimiento
    reg_img = np.zeros((h, w, 3), dtype=np.uint8)
    reg_img[..., 0] = gray_warp
    reg_img[..., 1] = old_gray
    reg_img[..., 2] = old_gray

    # Fotograma de referencia representado en RGB
    target_img = np.zeros((h, w, 3), dtype=np.uint8)
    target_img[..., 0] = old_gray
    target_img[..., 1] = old_gray
    target_img[..., 2] = old_gray

    return seq_img, reg_img, target_img


def plot_histogram(
    img,
    *,
    ax=None,
    bins=256,
    title=None,
    thresh=None,
    scale="",
    centroids=None,
    labels=None,
):
    if len(img.shape) > 2:
        raise ValueError(f"Sólo implementado para imágenes en escala de grises.")
    img = img.ravel()
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(img, bins=bins, range=(0, 256))
    if title:
        ax.set_title(title)
    if thresh:
        # Muestra el umbral en umbralizaciones globales
        ax.axvline(thresh, color="red", label="thresh")
        ax.legend(loc="upper right")
    if centroids is not None:
        # Muestra los centroides
        ax.hist(centroids, bins=55, zorder=2, label="centroides")
        ax.legend(loc="upper right")
    if labels is not None:
        rng = np.random.default_rng(21)
        # Colorea las valores de la imagen original según al centroide al que pertenecen
        for value in np.unique(labels):
            color = rng.choice(RANDOM_COLORS, 1)[0]
            x = img[labels == value]
            ax.hist(x, bins=bins, range=(0, bins), color=color)

    ax.grid(axis="y", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    if scale == "log":
        ax.set_yscale("log", base=10)

    return ax
