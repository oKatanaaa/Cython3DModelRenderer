import numpy as np

from .line_drawer import LineDrawer
from ..pixel_buffer_filler import PixelBufferFiller
from ...data_structures import Buffer


class EdgeOnlyPixelBufferFiller(PixelBufferFiller):
    def __init__(self, line_drawer: LineDrawer, line_color: np.ndarray,
                 draw_edges=True, force_triangle_colors=False):
        self.line_drawer = line_drawer
        self.line_color = line_color
        self.draw_edges = draw_edges
        self.force_triangle_colors = force_triangle_colors

    def compute_triangle_statistics(self, triangle: np.ndarray, colors: np.ndarray, normals: np.ndarray,
                                    color_buffer: Buffer, z_buffer: Buffer, n_buffer: Buffer):

        p0 = [int(triangle[0][0]), int(triangle[0][1])]
        p1 = [int(triangle[1][0]), int(triangle[1][1])]
        p2 = [int(triangle[2][0]), int(triangle[2][1])]

        if self.draw_edges:
            self.line_drawer.draw_line(p0, p1, color_buffer,
                                       (colors[0] if self.force_triangle_colors else self.line_color))
            self.line_drawer.draw_line(p1, p2, color_buffer,
                                       (colors[1] if self.force_triangle_colors else self.line_color))
            self.line_drawer.draw_line(p2, p0, color_buffer,
                                       (colors[2] if self.force_triangle_colors else self.line_color))
        else:
            color_buffer.set_pixel(*p0, (colors[0] if self.force_triangle_colors else self.line_color))
            color_buffer.set_pixel(*p1, (colors[1] if self.force_triangle_colors else self.line_color))
            color_buffer.set_pixel(*p2, (colors[2] if self.force_triangle_colors else self.line_color))
