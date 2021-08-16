from ..triangle_iterator import TriangleIterator
from ...data_structures import Model


class DepthIterator(TriangleIterator):
    def __init__(self, model: Model):
        self._model = model
        self._counter = 0
        self._n_triangles = self._model.n_triangles()
        self._triangles = [self._model.get_triangle(counter) for counter in range(self._n_triangles)]
        self._triangles.sort(key=lambda triangle_data: min(triangle_data[0][:, 2]))

    def __len__(self):
        return self._n_triangles

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter >= self._n_triangles:
            raise StopIteration('There are no triangles left in the model.')

        triangle_data = self._triangles[self._counter]
        self._counter += 1

        return triangle_data
