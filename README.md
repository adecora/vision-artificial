---
title: Image Playground
emoji: 💻
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Playground para trastear con el preprocesamiento de imágenes
---

# MIIA: Visión Artificial, PEC 1

<!-- Behind de la imagen más clásica de la visión por computador: https://en.wikipedia.org/wiki/Lenna -->
![Computer visión clásico](./data/lena.jpg)

- [x] Eliminación de ruido.
- [x] Transformaciones geométricas.
- [x] Calibración de la cámara.
- [x] Flujo óptico y seguimiento (Tracking).
- [x] Segmentación sin conocimiento del dominio.

## Trabajo

- https://adecora.github.io/vision-artificial/
- https://huggingface.co/spaces/vichel0creg0/image-playground


## Configuración

Algunos de los ficheros de imágenes de [data](./data) y otros binarios están trackeados con [`git-lfs`](https://git-lfs.com/), antes de clonar el repositorio es necesario ejecutar el comando para que los incluya en la descarga.

```
git lfs install
```

Tras clonar el repositorio para instalar el entorno en local, necesitas tener instalado [`uv`](https://docs.astral.sh/uv).

```bash
uv sync
```

## Uso

### Memoria

La memoria se encuentra en versión web en: [https://adecora.github.io/vision-artificial/](https://adecora.github.io/vision-artificial/).

### Procesamiento de imágenes

La aplicación para

La aplicación para el procesamiento de imágenes se encuentra desplegada en [huggingface spaces](https://huggingface.co/spaces/vichel0creg0/image-playground).

### CLI

La forma más fácil de utilizar la el cli es usando `uvx`.

```
uvx "git+https://github.com/adecora/vision-artificial.git"
```

Que da acceso a los siguientes comandos:

```bash
usage: miia-pec1 [-h] {calibrate,capture,meanshift,ui} ...

Calibrar la cámara utilizando imágenes de un tablero de ajedrez.

positional arguments:
  {calibrate,capture,meanshift,ui}
                        Subcomandos disponibles
    calibrate           Calibrar la cámara utilizando imágenes de un tablero de ajedrez.
    capture             Capturar imágenes de calibración utilizando la cámara.
    meanshift           Seguimiento de objetos utilizando el algoritmo meanshift.
    ui                  Lanzar el playground de procesamiento de imágenes.

options:
  -h, --help            show this help message and exit

Ejemplo de uso: calibrate --rows 6 --cols 7 --image_dir /ruta/a/imagenes
```

## Licencia

MIT