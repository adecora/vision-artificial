import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon


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
        ax.imshow(img, cmap="binary")
    else:
        ax.imshow(img)

    if axis:
        ax.xaxis.tick_top()
        return ax

    ax.axis("off")
    return ax


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
