import numpy as np

from render.pixel_buffer_filler.edge_only.line_drawer.line_drawer import LineDrawer


class LineBresenham(LineDrawer):

    def draw_line(self, p1, p2, image, color):

        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

        if dx < 0:
            dx = -dx
        if dy < 0:
            dy = -dy

        if dx > dy:
            pdx, pdy = sign_x, 0
            es, el = dy, dx
        else:
            pdx, pdy = 0, sign_y
            es, el = dx, dy

        x, y = x1, y1

        error, t = el / 2, 0

        image.set_pixel(x, y, color)

        while t < el:
            error -= es
            if error < 0:
                error += el
                x += sign_x
                y += sign_y
            else:
                x += pdx
                y += pdy
            t += 1
            image.set_pixel(x, y, color)

    # SHIT
    #
    # def draw_line(self, p1, p2, image, color):
    #     if p1 == p2:
    #         image.set_pixel(p1[0], p1[1], color)
    #         return
    #
    #     steep = False
    #     if np.abs(p1[0] - p2[0]) < np.abs(p1[1] - p2[1]):
    #         p1[0], p1[1] = p1[1], p1[0]
    #         p2[0], p2[1] = p2[1], p2[0]
    #         steep = True
    #     if p1[0] > p2[0]:
    #         p1[0], p2[0] = p2[0], p1[0]
    #         p1[1], p2[1] = p2[1], p1[1]
    #
    #     dx = p2[0] - p1[0]
    #     dy = p2[1] - p2[1]
    #     derror = np.abs(dy / dx)
    #     error = 0
    #     y = p1[1]
    #
    #     tmp = [p1[0], p2[0]]
    #     for x in tmp:
    #         if steep:
    #             image.set_pixel(y, x, color)
    #         else:
    #             image.set_pixel(x, y, color)
    #         error += derror
    #         if error > 0.5:
    #             y += (-1, 1)[p2[1] > p1[1]]
    #             error -= 1
    #
