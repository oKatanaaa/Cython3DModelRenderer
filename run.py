from crender import Renderer
from crender.py.data_structures import Model
from crender.py.pixel_buffer_filler import AdvancedPixelBufferFiller
from crender.py.triangle_iterator import SimpleIterator
from crender.py.illumination import GuroIllumination

if __name__ == '__main__':

    # line_color = np.array([255, 255, 255])
    # eo_td = EdgeOnlyPixelBufferFiller(LineBresenham(), line_color, force_triangle_colors=True)
    filler = AdvancedPixelBufferFiller(1024, 1024, fov=45)
    illumination = GuroIllumination([0, 0, 1])
    # illumination = NoIllumination()
    # By default it will crender 512x512 images
    renderer = Renderer(filler, illumination, SimpleIterator, *filler.get_size())

    def fit_model(m):
        m.shift(-m.get_mean_vertex())
        m.scale(1 / m.get_max_span())
        m.shift(shift=[0, 0, 4])

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



