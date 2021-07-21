from typing import Tuple

import cv2
import numpy as np


class Buffer:
    """
    Class represents Image entity which basically an decorator above np.ndarray class.
    """

    def __init__(self, height: int, width: int, dim: int = 3, dtype: str = 'float32', init_val=0):
        self._buffer = None

        self._height = height
        self._width = width
        self._dim = dim
        self._dtype = dtype
        self._init_val = init_val

        self.clear()

    def __getitem__(self, val) -> np.ndarray:
        """
        You should use this feature carefully. Default usage example is given below.

        Examples
        --------
        color = image[y, x]

        Parameters
        ----------
        val : slice
            via the parameter you may copy an image, or get access to different axis, so be carefully
        """
        return self._buffer[val]

    def __setitem__(self, key, value) -> None:
        """
        You should use this feature carefully. Default usage example is given below.

        Examples
        --------
        image[y, x] = [255, 255, 255]

        Parameters
        ----------
        key : slice
            via the parameter you may change even whole image
        value : Tuple[int, int, int]
        """
        self._buffer[key] = value

    def write_to_file(self, filename: str) -> None:
        cv2.imwrite(filename, self._buffer[::-1])

    def get_pixel(self, x: int, y: int) -> np.ndarray:
        return self._buffer[y, x]

    def get_size(self) -> Tuple[int, int]:
        return self._height, self._width

    def get_image(self) -> np.ndarray:
        return self._buffer

    def set_pixel(self, x: int, y: int, value: np.ndarray) -> None:
        if x not in range(self._width) or y not in range(self._height):
            return
        self._buffer[y, x] = value

    def clear(self) -> None:
        """
        clear()

        The method resets self.image variable to default condition (fills with zeros)
        """
        self._buffer = np.zeros((self._height, self._width, self._dim), dtype=self._dtype)
        self._buffer[...] = self._init_val
