from abc import abstractmethod
import numpy as np

from ..data_structures import Buffer


class PixelBufferFiller:
    @abstractmethod
    def compute_triangle_statistics(self, triangle: np.ndarray, colors: np.ndarray, normals: np.ndarray,
                                    color_buffer: Buffer, z_buffer: Buffer, n_buffer: Buffer):
        pass
