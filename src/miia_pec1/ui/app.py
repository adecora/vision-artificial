import copy
from pathlib import Path

import gradio as gr

ASSETS_DIR = Path(__file__).parent / "assets"

from .tranformaciones import (
    BinaryInverseThreshold,
    BinaryThreshold,
    Closing,
    Dilation,
    EqualizeHist,
    Erosion,
    GaussianBlur,
    MedianBlur,
    Opening,
    OtsuThreshold,
    ToZeroInverseThreshold,
    ToZeroThreshold,
    TruncThreshold,
)
from .utils import crop_roi, plot_histogram, roi_bounds

OPERATIONS = {
    "Gaussian Blur": GaussianBlur,
    "Median Blur": MedianBlur,
    "Umbral binario": BinaryThreshold,
    "Umbral binario inverso": BinaryInverseThreshold,
    "Umbral truncado": TruncThreshold,
    "Umbral a cero": ToZeroThreshold,
    "Umbral a cero inverso": ToZeroInverseThreshold,
    "Umbral de Otsu": OtsuThreshold,
    "Dilatación": Dilation,
    "Erosión": Erosion,
    "Cierre": Closing,
    "Apertura": Opening,
    "Ecualización de histograma": EqualizeHist,
}

examples = [
    f for f in ASSETS_DIR.rglob("N_*.png")
    if f.name.startswith(("N_307", "N_328"))
]


def modify_pipeline(action, state, op=None, idx=None):
    """
    Actualiza el estado del pipeline según la acción especificada (add, up, down, delete, clear)

    Parámetros:
    ===========
    action: Acción a realizar ("add", "up", "down", "delete", "clear")
    state: Estado actual del pipeline
    op: Operación a añadir o modificar
    idx: Índice de la operación a modificar o mover

    Retorna:
    ===========
    list: Nuevo estado del pipeline
    """
    new_state = copy.deepcopy(state or [])
    if action == "add" and op in OPERATIONS:
        new_id = max((step["id"] for step in new_state), default=0) + 1
        new_state.append({"id": new_id, "transform": OPERATIONS[op]()})
    elif action == "clear":
        new_state = []
    elif action == "up":
        if idx is not None and 0 < idx < len(new_state):
            new_state[idx - 1], new_state[idx] = new_state[idx], new_state[idx - 1]
    elif action == "down":
        if idx is not None and 0 <= idx < len(new_state) - 1:
            new_state[idx], new_state[idx + 1] = new_state[idx + 1], new_state[idx]
    elif action == "delete":
        if idx is not None and 0 <= idx < len(new_state):
            new_state.pop(idx)

    return new_state


def update_roi_histogram(image, enabled, hist_enabled, xmin, xmax, ymin, ymax):
    if image is None or not enabled or not hist_enabled:
        return None
    bounds = roi_bounds(image, xmin, xmax, ymin, ymax)
    if bounds is None:
        return None
    return plot_histogram(crop_roi(image, xmin, xmax, ymin, ymax))


def store_uploaded_image(image, roi_enabled, hist_enabled, xmin, xmax, ymin, ymax):
    """
    Almacena la imagen subida y genera las vistas previas correspondientes.

    Parámetros:
    ===========
    image: Imagen subida
    roi_enabled: Booleano que indica si la ROI está activada
    hist_enabled: Booleano que indica si el histograma está activado
    xmin: Porcentaje mínimo en X
    xmax: Porcentaje máximo en X
    ymin: Porcentaje mínimo en Y
    ymax: Porcentaje máximo en Y

    Retorna:
    ===========
    tuple: Imagen original, vista previa, histograma de entrada, histograma de ROI
    """
    roi_histogram = update_roi_histogram(
        image, roi_enabled, hist_enabled, xmin, xmax, ymin, ymax
    )
    input_histogram = plot_histogram(image) if hist_enabled else None
    return image, input_histogram, roi_histogram


def clear_uploaded_image():
    """
    Limpia la imagen subida y todos los estados relacionados.
    """
    return None, None, None, None, None, None, None


def update_output_histogram(images, enabled):
    """
    Actualiza el histograma de salida basado en la imagen procesada y check de histograma.

    Parámetros:
    ===========
    images: Lista o tupla de imágenes procesadas
    enabled: Booleano que indica si el histograma está activado

    Retorna:
    ===========
    numpy.ndarray: Histograma de la imagen procesada o None si no está habilitado
    """
    if not enabled or images is None:
        return None
    processed_image = images[1] if isinstance(images, (tuple, list)) else images
    return plot_histogram(processed_image)

def update_display_layout(roi_enabled, hist_enabled):
    """
    Actualiza el UI según el estado de ROI e histograma

    Parámetros:
    ===========
    roi_enabled: Booleano que indica si la ROI está activada
    hist_enabled: Booleano que indica si el histograma está activado
    """
    return (
        gr.update(visible=roi_enabled),
        gr.update(scale=1),
        gr.update(scale=1 if roi_enabled else 2),
        gr.update(visible=hist_enabled, scale=1),
        gr.update(
            visible=roi_enabled and hist_enabled, scale=1 if roi_enabled else 0
        ),
        gr.update(visible=hist_enabled, scale=1 if roi_enabled else 2),
    )


def execute_pipeline(img, pipeline, roi_enabled, hist_enabled, xmin, xmax, ymin, ymax):
    """
    Ejecuta el pipeline de procesamiento de imágenes sobre la imagen proporcionada.

    Parámetros:
    ===========
    img: Imagen de entrada
    pipeline: Lista de pasos de transformación
    roi_enabled: Booleano que indica si la ROI está activada
    hist_enabled: Booleano que indica si el histograma está activado
    xmin: Porcentaje mínimo en X
    xmax: Porcentaje máximo en X
    ymin: Porcentaje mínimo en Y
    ymax: Porcentaje máximo en Y

    Retorna:
    ===========
    tuple: Imagen original, imagen procesada, etapas intermedias, histograma de salida
    """
    original_img = crop_roi(img, xmin, xmax, ymin, ymax) if roi_enabled else img.copy()
    if original_img is None:
        raise gr.Error("La ROI está vacía. Ajusta sus límites.")
    output_stages = []
    output_stages.append(
        (original_img, "00. ROI original" if roi_enabled else "00. Original")
    )
    img = original_img.copy()
    for i, step in enumerate(pipeline, start=1):
        transform = step["transform"]
        img = transform.apply(img)
        output_stages.append((img.copy(), f"{i:02d}. {transform!r}"))

    histogram = plot_histogram(img) if hist_enabled else None
    return (original_img, img), output_stages, histogram, img


with gr.Blocks(title="MIIA-VA") as demo:
    gr.Markdown("""
    # Playground: preprocesamiento de imágenes
    """)

    pipeline = gr.State([])
    source_image = gr.State(None)
    processed_image = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1) as input_column:
            input_image = gr.Image(
                label="Imagen original", type="numpy", image_mode="RGB"
            )
        with gr.Column(scale=2) as output_column:
            with gr.Tab("Imagen procesada"):
                output_image = gr.ImageSlider(interactive=False)
            with gr.Tab("Resultados intermedios"):
                output_stages = gr.Gallery(columns=3, object_fit="contain")
    with gr.Row():
        input_histogram = gr.LinePlot(
            label="Histograma original",
            x="Intensidad",
            y="Frecuencia",
            color="Canal",
            x_lim=[0, 255],
            height=260,
            buttons=["fullscreen"],
            visible=False,
            scale=1,
        )
        roi_histogram = gr.LinePlot(
            label="Histograma ROI",
            x="Intensidad",
            y="Frecuencia",
            color="Canal",
            x_lim=[0, 255],
            height=260,
            buttons=["fullscreen"],
            visible=False,
            scale=0,
        )
        output_histogram = gr.LinePlot(
            label="Histograma procesado",
            x="Intensidad",
            y="Frecuencia",
            color="Canal",
            x_lim=[0, 255],
            height=260,
            buttons=["fullscreen"],
            visible=False,
            scale=2,
        )

    with gr.Row():
        with gr.Column():
            hist_check = gr.Checkbox(label="Ver histograma", value=False)
            roi_check = gr.Checkbox(label="Activar ROI", value=False)
            with gr.Group(visible=False) as roi_controls:
                gr.Markdown("### ROI ")
                with gr.Row():
                    xmin = gr.Slider(
                        minimum=0, maximum=100, value=0, step=1, label="X incial %"
                    )
                    xmax = gr.Slider(
                        minimum=0, maximum=100, value=100, step=1, label="X final %"
                    )
                with gr.Row():
                    ymin = gr.Slider(
                        minimum=0, maximum=100, value=0, step=1, label="Y incial %"
                    )
                    ymax = gr.Slider(
                        minimum=0, maximum=100, value=100, step=1, label="Y final %"
                    )
            with gr.Row():
                operation = gr.Dropdown(
                    choices=list(OPERATIONS),
                    value=next(iter(OPERATIONS)),
                    label="Transformación",
                )
                with gr.Column():
                    add_operation = gr.Button("Añadir transformación")
                    run_pipeline = gr.Button(
                        "Ejecutar pipeline", variant="primary", interactive=False
                    )

        with gr.Column():
            with gr.Accordion("Pipeline", open=True, max_height=500):

                @gr.render(inputs=[pipeline])
                def render_pipeline(state):
                    for idx, step in enumerate(state or []):
                        transform = step["transform"]
                        with gr.Group(key=f"op-{step['id']}"):
                            with gr.Row():
                                gr.Markdown(
                                    f"&nbsp;&nbsp;**{idx + 1}. {transform}**", scale=1
                                )
                                up = gr.Button(
                                    "↑",
                                    size="sm",
                                    interactive=idx > 0,
                                    scale=0,
                                    min_width=50,
                                    key=f"up-{step['id']}",
                                )
                                down = gr.Button(
                                    "↓",
                                    size="sm",
                                    interactive=idx < len(state) - 1,
                                    scale=0,
                                    min_width=50,
                                    key=f"down-{step['id']}",
                                )
                                delete = gr.Button(
                                    "✕",
                                    size="sm",
                                    scale=0,
                                    min_width=50,
                                    key=f"delete-{step['id']}",
                                )

                            controls = transform.gradio_component(key_prefix=step["id"])

                        up.click(
                            lambda state, idx=idx: modify_pipeline(
                                "up", state, idx=idx
                            ),
                            inputs=[pipeline],
                            outputs=[pipeline],
                        )
                        down.click(
                            lambda state, idx=idx: modify_pipeline(
                                "down", state, idx=idx
                            ),
                            inputs=[pipeline],
                            outputs=[pipeline],
                        )
                        delete.click(
                            lambda state, idx=idx: modify_pipeline(
                                "delete", state, idx=idx
                            ),
                            inputs=[pipeline],
                            outputs=[pipeline],
                        )
                        parameter_names = list(controls)
                        parameter_components = list(controls.values())
                        if parameter_components:
                            parameter_triggers = [
                                component.release
                                if "release" in component.EVENTS
                                else component.change
                                for component in parameter_components
                            ]
                            # loquenado, ya no se dispara pipeline.change,
                            # para fumarse los re-render.
                            gr.on(
                                triggers=parameter_triggers,
                                fn=lambda *values, step=step, names=parameter_names: step[
                                    "transform"
                                ].set_parameters(**dict(zip(names, values))),
                                inputs=parameter_components,
                                outputs=[],
                                show_progress="hidden",
                            )

    gr.Examples(
        examples=examples,
        inputs=[input_image],
        cache_examples=False,
    )

    gr.on(
        triggers=[roi_check.change, hist_check.change],
        fn=update_display_layout,
        inputs=[roi_check, hist_check],
        outputs=[
            roi_controls,
            input_column,
            output_column,
            input_histogram,
            roi_histogram,
            output_histogram,
        ],
        show_progress="hidden",
    )

    for trigger in (source_image.change, hist_check.change):
        trigger(
            lambda img, enabled: plot_histogram(img) if enabled else None,
            inputs=[source_image, hist_check],
            outputs=input_histogram,
            show_progress="hidden",
        )

    input_image.change(
        store_uploaded_image,
        inputs=[input_image, roi_check, hist_check, xmin, xmax, ymin, ymax],
        outputs=[source_image, input_histogram, roi_histogram],
        show_progress="hidden",
    )
    input_image.clear(
        clear_uploaded_image,
        outputs=[
            source_image,
            output_image,
            input_histogram,
            roi_histogram,
            output_histogram,
            processed_image,
            output_stages,
        ],
        show_progress="hidden",
    )

    preview_triggers = (
        roi_check.change,
        hist_check.change,
        xmin.input,
        xmax.input,
        ymin.input,
        ymax.input,
    )
    for trigger in preview_triggers:
        trigger(
            update_roi_histogram,
            inputs=[source_image, roi_check, hist_check, xmin, xmax, ymin, ymax],
            outputs=[roi_histogram],
            show_progress="hidden",
        )

    hist_check.change(
        update_output_histogram,
        inputs=[processed_image, hist_check],
        outputs=output_histogram,
        show_progress="hidden",
    )

    add_operation.click(
        lambda state, op: modify_pipeline("add", state, op),
        inputs=[pipeline, operation],
        outputs=[pipeline],
    )

    for trigger in (pipeline.change, source_image.change):
        trigger(
            lambda state, img: gr.update(interactive=bool(state) and img is not None),
            inputs=[pipeline, source_image],
            outputs=[run_pipeline],
            show_progress="hidden",
        )

    run_pipeline.click(
        execute_pipeline,
        inputs=[source_image, pipeline, roi_check, hist_check, xmin, xmax, ymin, ymax],
        outputs=[output_image, output_stages, output_histogram, processed_image],
    )


if __name__ == "__main__":
    demo.launch(favicon_path=ASSETS_DIR / "favicon.ico")
