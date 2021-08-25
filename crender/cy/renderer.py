import numpy as np
from tqdm import tqdm

from .data_structures import Buffer, Model
from .pixel_buffer_filler import PixelBufferFiller
from .illumination import IlluminationDrawer


class Renderer:
    def __init__(
            self, pixel_buffer_filler: PixelBufferFiller, illumination: IlluminationDrawer, triangle_iterator_type: type,
            image_height: int = 512, image_width: int = 512, use_tqdm=True
    ):
        self.pixel_buffer_filler = pixel_buffer_filler
        self.illumination = illumination
        self.triangle_iterator_type = triangle_iterator_type
        self.im_h = image_height
        self.im_w = image_width
        #self.color_buffer = Buffer(image_height, image_width, dim=3, dtype='uint8')
        #self.z_buffer = Buffer(image_height, image_width, dim=1, init_val=1e6, dtype='float32')
        #self.n_buffer = Buffer(image_height, image_width, dim=3, dtype='float32')
        self.use_tqdm = use_tqdm

    def render(self, model: Model, normalize_model=False, random_colors=True):
        """
        Performs rendering of the given model by computing its color, normals and depth and saving it to the
        pixel-level buffers.

        Parameters
        ----------
        model : Model
            Model to crender.
        normalize_model : bool
            If true, model coordinates will be changed in order it to fit the image.
            If false, projective transform will be done.
        random_colors: bool
            In case of no color: if true, triangles will be each in a random color, else they will be white
        Returns
        -------
        Buffer
            The color buffer.
        """
        # Create an image
        if normalize_model:
            image_center = (self.im_h // 2, self.im_w // 2)
            image_span = min(image_center)
            # GENIUS FITTING
            model.scale(image_span / model.get_max_span())
            model.shift(- model.get_mean_vertex() + [image_center[0], image_center[1], -image_span])
        iter_wrap = tqdm if self.use_tqdm else lambda x: x
        for triangle, colors, normals in iter_wrap(self.triangle_iterator_type(model)):
            if colors is None:
                color = np.random.randint(256, size=3) if random_colors else np.array([255, 255, 255])
                colors = np.stack([color] * 3)

            self.pixel_buffer_filler.compute_triangle_statistics(triangle, colors, normals)

        self.illumination.draw_illumination(self.pixel_buffer_filler.get_color_buffer(), self.pixel_buffer_filler.get_normals_buffer())
        return self.pixel_buffer_filler.get_color_buffer()

    def reset_buffers(self):
        self.n_buffer.clear()
        self.z_buffer.clear()
        self.color_buffer.clear()
