from ..triangle_iterator import TriangleIterator
from ...data_structures import Model


class SimpleIterator(TriangleIterator):
    def __init__(self, model: Model):
        self._model = model
        self._counter = 0
        self._n_triangles = model.n_triangles()

    def __len__(self):
        return self._model.n_triangles()

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter >= self._n_triangles:
            raise StopIteration('There are no triangles left in the model.')

        triangle_data = self._model.get_triangle(self._counter)
        self._counter += 1

        return triangle_data
