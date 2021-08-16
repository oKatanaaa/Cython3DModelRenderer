from .illumination_drawer import IlluminationDrawer
from ..data_structures import Buffer
import numpy as np


# noinspection PyDefaultArgument
class GuroIllumination(IlluminationDrawer):
    def __init__(self, light_direction=[0, 0, 1]):
        """
        Implements the most primitive Guro illumination.

        Parameters
        ----------
        light_direction : array of shape [3]
            A vector determining the light direction (which direction it falls).
        """
        # Revert the vector to align it with the surface normals in front of camera
        light_direction = -np.asarray(light_direction, dtype='float32')
        self.light_direction = light_direction / np.linalg.norm(light_direction)

    def draw_illumination(self, color_buffer: Buffer, n_buffer: Buffer):
        # 1. Computes a similarity between a normal (of a pixel) and the light direction
        # 2. Multiplies the corresponding pixel color with the computed similarity
        scalar_product = np.sum(n_buffer[...] * self.light_direction, axis=-1, keepdims=True)
        norm = np.linalg.norm(n_buffer[[...]], axis=-1, keepdims=True)
        shadow_coeff = scalar_product / (norm + 1e-6)
        shadow_coeff = np.clip(shadow_coeff, 0, 1)
        color_buffer[...] = (color_buffer[...].astype('float32') * shadow_coeff).astype('uint8')
