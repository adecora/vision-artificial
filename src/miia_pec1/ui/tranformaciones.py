from abc import abstractmethod
from typing import ClassVar

import cv2 as cv
import gradio as gr
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    """
    Configuración base para las clases de configuración de ingesta.
    Impide que se pasen parámetros adicionales no definidos en el modelo y habilita la validación por nombre y alias.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Kernel(BaseConfig):
    size: int = Field(default=3, description="Tamaño del kernel")
    min_size: int = Field(default=3, description="Tamaño mínimo del kernel")
    max_size: int = Field(default=21, description="Tamaño máximo del kernel")
    step: int = Field(default=2, description="Incremento del tamaño del kernel")

    def __repr__(self):
        return f"Kernel(size={self.size})"


class ImageSmoothing(BaseConfig):
    """
    Clase base para los métodos de suavizado de imágenes.
    """
    kernel: Kernel = Field(
        default_factory=Kernel, description="Kernel para el suavizado de imágenes"
    )
    sigma_x: float = Field(
        default=0, description="Sigma en X, desviación estándar en la dirección X"
    )
    uses_sigma: ClassVar[bool] = False
    label: ClassVar[str] = "Image Smoothing"

    def __str__(self):
        return self.label

    @abstractmethod
    def apply(self, image):
        raise NotImplementedError("Esta función debe ser implementada por las subclases")

    @property
    def get_parameters(self):
        params = {"kernel_size": self.kernel.size}
        if self.uses_sigma:
            params["sigma_x"] = self.sigma_x
        return params

    def set_parameters(self, **kwargs):
        if "kernel_size" in kwargs:
            self.kernel.size = int(kwargs["kernel_size"])
        if "sigma_x" in kwargs and self.uses_sigma:
            self.sigma_x = float(kwargs["sigma_x"])

    def gradio_component(self, key_prefix=None):
        controls = {
            "kernel_size": gr.Slider(
                minimum=self.kernel.min_size,
                maximum=self.kernel.max_size,
                step=self.kernel.step,
                value=self.kernel.size,
                label="Kernel Size",
                interactive=True,
                key=f"{key_prefix}-kernel_size",
            )
        }
        if self.uses_sigma:
            controls["sigma_x"] = gr.Number(
                value=self.sigma_x,
                label="Sigma X",
                interactive=True,
                key=f"{key_prefix}-sigma_x",
            )
        return controls

class MedianBlur(ImageSmoothing):
    label = "Median Blur"

    def apply(self, image):
        return cv.medianBlur(image, self.kernel.size)


class GaussianBlur(ImageSmoothing):
    label = "Gaussian Blur"
    uses_sigma = True

    def apply(self, image):
        return cv.GaussianBlur(
            image,
            (self.kernel.size, self.kernel.size),
            self.sigma_x,
        )


class Threshold(BaseConfig):
    """
    Clase base para los métodos de umbralización de imágenes.
    """
    threshold: int = Field(default=127, ge=0, le=255)
    max_value: int = Field(default=255, ge=0, le=255)
    method: ClassVar[int] = cv.THRESH_BINARY
    uses_otsu: ClassVar[bool] = False
    uses_max_value: ClassVar[bool] = True
    label: ClassVar[str] = "Threshold"

    def __str__(self):
        return self.label

    def apply(self, image):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if image.ndim == 3 else image
        method = self.method | (cv.THRESH_OTSU if self.uses_otsu else 0)
        return cv.threshold(gray, self.threshold, self.max_value, method)[1]

    @property
    def get_parameters(self):
        params = {}
        if self.uses_max_value:
            params["max_value"] = self.max_value
        if not self.uses_otsu:
            params["threshold"] = self.threshold
        return params

    def set_parameters(self, **kwargs):
        for name in ("threshold", "max_value"):
            if name in kwargs and not (
                (name == "threshold" and self.uses_otsu)
                or (name == "max_value" and not self.uses_max_value)
            ):
                setattr(self, name, int(kwargs[name]))

    def gradio_component(self, key_prefix=None):
        controls = {}
        if self.uses_max_value:
            controls["max_value"] = gr.Slider(
                minimum=0,
                maximum=255,
                step=1,
                value=self.max_value,
                label="Valor máximo",
                interactive=True,
                key=f"{key_prefix}-max_value",
            )
        if not self.uses_otsu:
            controls = {
                "threshold": gr.Slider(
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=self.threshold,
                    label="Umbral",
                    interactive=True,
                    key=f"{key_prefix}-threshold",
                ),
                **controls,
            }
        return controls


class BinaryThreshold(Threshold):
    label = "Umbral binario"


class BinaryInverseThreshold(Threshold):
    method = cv.THRESH_BINARY_INV
    label = "Umbral binario inverso"


class TruncThreshold(Threshold):
    method = cv.THRESH_TRUNC
    uses_max_value = False
    label = "Umbral truncado"


class ToZeroThreshold(Threshold):
    method = cv.THRESH_TOZERO
    uses_max_value = False
    label = "Umbral a cero"


class ToZeroInverseThreshold(Threshold):
    method = cv.THRESH_TOZERO_INV
    uses_max_value = False
    label = "Umbral a cero inverso"


class OtsuThreshold(Threshold):
    uses_otsu = True
    label = "Umbral de Otsu"


class Morphology(BaseConfig):
    """
    Clase base para las operaciones morfológicas en imágenes.
    """

    kernel: Kernel = Field(default_factory=Kernel)
    operation: ClassVar[int] = cv.MORPH_OPEN
    label: ClassVar[str] = "Morfología"

    def __str__(self):
        return self.label

    def apply(self, image):
        size = self.kernel.size
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        if self.operation == cv.MORPH_DILATE:
            return cv.dilate(image, element)
        if self.operation == cv.MORPH_ERODE:
            return cv.erode(image, element)
        return cv.morphologyEx(image, self.operation, element)

    @property
    def get_parameters(self):
        return {"kernel_size": self.kernel.size}

    def set_parameters(self, **kwargs):
        if "kernel_size" in kwargs:
            self.kernel.size = int(kwargs["kernel_size"])

    def gradio_component(self, key_prefix=None):
        return {
            "kernel_size": gr.Slider(
                minimum=self.kernel.min_size,
                maximum=self.kernel.max_size,
                step=self.kernel.step,
                value=self.kernel.size,
                label="Tamaño del kernel",
                interactive=True,
                key=f"{key_prefix}-kernel_size",
            )
        }


class Dilation(Morphology):
    operation = cv.MORPH_DILATE
    label = "Dilatación"


class Erosion(Morphology):
    operation = cv.MORPH_ERODE
    label = "Erosión"


class Closing(Morphology):
    operation = cv.MORPH_CLOSE
    label = "Cierre"


class Opening(Morphology):
    operation = cv.MORPH_OPEN
    label = "Apertura"



class EqualizeHist(BaseConfig):
    label: ClassVar[str] = "Ecualización de histograma"

    def __str__(self):
        return self.label

    def apply(self, image):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if image.ndim == 3 else image
        return cv.equalizeHist(gray)

    def get_parameters(self):
        return {}

    def set_parameters(self, **kwargs):
        pass

    def gradio_component(self, key_prefix=None):
        return {}
