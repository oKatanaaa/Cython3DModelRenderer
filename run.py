import numpy as np

from crender import Renderer
from crender.data_structures import Model
from crender.pixel_buffer_filler import EdgeOnlyPixelBufferFiller
from crender.pixel_buffer_filler.edge_only.line_drawer import LineBresenham
from crender.pixel_buffer_filler import AdvancedPixelBufferFiller
from crender.triangle_iterator import SimpleIterator
from crender.illumination import GuroIllumination, NoIllumination


if __name__ == '__main__':

    # line_color = np.array([255, 255, 255])
    # eo_td = EdgeOnlyPixelBufferFiller(LineBresenham(), line_color, force_triangle_colors=True)
    filler = AdvancedPixelBufferFiller(1024, 1024, fov=45)
    illumination = GuroIllumination([0, 0, 1])
    # illumination = NoIllumination()
    # By default it will render 512x512 images
    renderer = Renderer(filler, illumination, SimpleIterator, *filler.get_size())

    def fit_model(m):
        m.shift(-m.get_mean_vertex())
        m.scale(1 / m.get_max_span())
        m.shift(shift=[0, 0, 4])

    model = Model.read_model('objects/Cube2.obj')
    model.rotate([45, 45, 45])
    fit_model(model)
    image = renderer.render(model)
    image.write_to_file('output/cube2.jpg')
    renderer.reset_buffers()

    model = Model.read_model('objects/T-Rex.obj')

    model.rotate([-90, 180, 0])
    model.rotate([10, -80, 0])
    fit_model(model)
    image = renderer.render(model)
    image.write_to_file('output/T-Rex.png')
    renderer.reset_buffers()

    model = Model.read_model('objects/bunny.obj')
    fit_model(model)
    image = renderer.render(model)
    image.write_to_file('output/bunny.jpg')
    renderer.reset_buffers()

    model = Model.read_model('objects/igor.obj')
    # fit_model(model)
    # image = renderer.render(model)
    # image.write_to_file('output/igor.png')
    # renderer.reset_buffers()

    model.rotate([0, 180, 0])
    fit_model(model)
    image = renderer.render(model)
    image.write_to_file('output/igor_y180.png')



