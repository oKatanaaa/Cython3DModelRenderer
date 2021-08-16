from abc import abstractmethod
from ..data_structures import Buffer


class IlluminationDrawer:
    @abstractmethod
    def draw_illumination(self, color_buffer: Buffer, n_buffer: Buffer):
        pass


class NoIllumination(IlluminationDrawer):
    def draw_illumination(self, color_buffer: Buffer, n_buffer: Buffer):
        pass
