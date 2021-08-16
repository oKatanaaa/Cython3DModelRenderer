from abc import abstractmethod


class LineDrawer:
    @abstractmethod
    def draw_line(self, p1, p2, image, color):
        pass
