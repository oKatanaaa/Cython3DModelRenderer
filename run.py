import cv2
import crender.py as py

from crender.cy import Renderer
from crender.cy.data_structures import Model
from crender.cy.pixel_buffer_filler import AdvancedPixelBufferFiller
from crender.cy.triangle_iterator import SimpleIterator
from crender.cy.illumination import GuroIllumination


def py_renderer(model):
    filler = py.pixel_buffer_filler.AdvancedPixelBufferFiller(1024, 1024, fov=45)
    illumination = py.illumination.GuroIllumination([0, 0, 1])
    # By default it will crender 512x512 images
    renderer = py.Renderer(filler, illumination, SimpleIterator, *filler.get_size())
    image = renderer.render(model)
    image.write_to_file('output/T-Rex.png')


def cy_renderer(model):
    filler = AdvancedPixelBufferFiller(1024, 1024, fov=45, n_threads=8)
    illumination = GuroIllumination([0, 0, 1])
    # By default it will crender 512x512 images
    renderer = Renderer(filler, illumination, SimpleIterator, *filler.get_size())
    image = renderer.render(model)
    cv2.imwrite('output/T-Rex.png', image[::-1].astype('uint8'))


if __name__ == '__main__':
    def fit_model(m):
        m.shift(-m.get_mean_vertex())
        m.scale(1 / m.get_max_span())
        m.shift(shift=[0, 0, 1])

    model = Model.read_model('objects/T-Rex.obj')

    model.rotate([-90, 180, 0])
    model.rotate([10, -80, 0])
    fit_model(model)

    cy_renderer(model)


