# cython: profile=True

cimport numpy as cnp
cimport cython
from libc.math cimport ceil
from libc.stdlib cimport free

from .math_utils cimport compute_bar_coords, reduce_min, reduce_max, clip, matmul_3x4, matmul
from .array_utils cimport select_values_float, select_values_int

import numpy as np

from crender.py.pixel_buffer_filler.pixel_buffer_filler import PixelBufferFiller
from crender.py.data_structures import Buffer



def im_ind(xy_coords):
    # a small utils for fast indexing in the image tensor
    return xy_coords[:, 1], xy_coords[:, 0]


cdef class AdvancedPixelBufferFiller:
    cdef:
        int _h
        int _w
        float _fov
        float _f
        float _z_near
        float _z_far
        float _a
        float[:, ::1] _ones4
        object _proj_mat
        int[:, :, ::1] _xy_grid
        float[:, ::1] _projected_tri_buffer

    def __cinit__(self, h, w, fov=90.0, z_near=0.1, z_far=1000.0):
        self._h = <int>h
        self._w = <int>w
        # Prepare the pixel coords buffer beforehand to make slices of it later instead
        # of reallocating memory each time
        # You have to specify the buffer type explicitly because otherwise
        # in another environment you may encounter an issue when numpy allocates an
        # arroy of dtype long which leads to runtime exception.
        # For example, on my laptop with Windows the code without specified dtype runs just fine.
        # But if I try to run it on Linux, it crashes telling me "dtype mismatch expected int but got long" or
        # something like that. It may be related to numpy version, but I didn't check.
        # The point is ALWAYS SPECIFY THE DTYPE!
        x_coords = np.arange(0, w, dtype='int32')
        y_coords = np.arange(0, h, dtype='int32')
        x, y = np.meshgrid(x_coords, y_coords)
        self._xy_grid = np.stack([x, y], axis=-1)
        self._fov = <float>fov
        self._f = 1 / np.tan(self._fov / 2 / 180 * np.pi)
        self._z_near = <float>z_near
        self._z_far = <float>z_far
        # Aspect ratio
        self._a = h / w
        self._projected_tri_buffer = np.empty(shape=[4, 4], dtype='float32')
        self._init_projection_matrix()

    def get_size(self):
        return self._h, self._w

    def _init_projection_matrix(self):
        q = self._z_far / (self._z_far - self._z_near)
        self._proj_mat = np.array([
            [self._f / self._a,     0,                  0, 0],
            [0,               self._f,                  0, 0],
            [0,                     0,                  q, 1],
            [0,                     0,  -self._z_near * q, 0]
        ], dtype='float32')

        self._ones4 = np.ones((3, 4), dtype='float32')

    def compute_triangle_statistics(self, triangle: cnp.ndarray, colors: cnp.ndarray, normals: cnp.ndarray,
                                    color_buffer: Buffer, z_buffer: Buffer, n_buffer: Buffer):
        # 1. Determine the area of interest
        # 2. Compute barycentric coordinates
        # 3. Fill in z-buffer and determine which pixels are visible
        # 3. Fill in other buffers
        assert color_buffer.get_size() == z_buffer.get_size() == n_buffer.get_size() == (self._h, self._w), \
            "Buffers' spatial dimensions must be the same, but received: " \
            f"color_buffer - {color_buffer.get_size()}, " \
            f"z_buffer - {z_buffer.get_size()}, " \
            f"n_buffer - {n_buffer.get_size()}."

        # (Task #13)
        if np.dot([0, 0, 1], np.mean(normals, axis=0)) >= 0:
            # The triangle faces away from the camera, so don't need to draw it
            return

        cdef cnp.ndarray[float, ndim=2] projected_tri = self._project_on_screen(triangle)

        cdef cnp.ndarray[int, ndim=2] pixel_coords
        pixel_coords = np.asarray(self._compute_pixel_coords(projected_tri))

        if pixel_coords.shape[0] == 0:
            # The triangle may be invisible, no need for further processing
            return

        # Returns coords only for those pixels that lie within the triangle
        cdef cnp.ndarray[float, ndim=2] pixel_bar_coords
        pixel_bar_coords, pixel_coords = self._compute_barycentric_coords(projected_tri, pixel_coords)

        if pixel_bar_coords.shape[0] == 0:
            # No pixels are visible
            return
        # Returns coords only for visible pixels (that are not behind something)
        cdef:
            float[:, :] pixel_bar_coords_
            int[:, :] pixel_coords_

        pixel_bar_coords_, pixel_coords_ = self._fill_z_buffer(projected_tri, pixel_coords, pixel_bar_coords, z_buffer)
        pixel_bar_coords = np.asarray(pixel_bar_coords_)
        pixel_coords = np.asarray(pixel_coords_)

        if pixel_bar_coords.shape[0] == 0:
            # The triangle may be invisible, no need to call buffers filling
            return

        self._fill_buffer(colors, pixel_coords, pixel_bar_coords, color_buffer)
        self._fill_buffer(normals, pixel_coords, pixel_bar_coords, n_buffer)

    cdef cnp.ndarray[float, ndim=2] _project_on_screen(self, float[:,:] tri):
        self._ones4[:3, :3] = tri
        # --- Perspective projection
        # Projects vertices onto the screen plane and makes them to be in
        # range [-1, 1]. Vertices outside of that range are invisible.
        cdef float[:, ::1] projected_tri = self._projected_tri_buffer
        # Clear the buffer
        projected_tri[...] = 0.0
        matmul_3x4(self._ones4, self._proj_mat, projected_tri)
        # --- Note
        # Before there was the following line:
        # projected_tri = np.asarray(self._projected_tri_buffer)
        # It doesn't change the code functionality (I just forgot to delete it), but it turns out
        # that execution of that line takes up to 32% of the whole time in this method!
        # Removing this line boosted performance of that method from 0.019s to 0.013s.
        # (measurements are taken using cProfile from tottime column).
        # --- Perspective divide: (x, y) / z.
        # It makes farther objects to appear smaller on the screen.
        # Z coordinate is made to be in range [0, 1].
        # 0 means the object is very close to the camera (lies on the screen).
        # 1 means the object is on the border of visibility (the farthest visible point).
        # Any points outside of that range are ether behind the camera or too far away to be visible.
        cdef size_t i
        for i in range(3):
            projected_tri[i, 0] /= projected_tri[i, -1]
            projected_tri[i, 1] /= projected_tri[i, -1]
            projected_tri[i, 2] /= projected_tri[i, -1]

        # --- Stretching the objects along the screen.
        # The same as:
        # 1. multiply by w/2, h/2
        # 2. shift by w/2, h/2 into the center of the screen
        cdef float shift = 1.0, x_scale = self._w / 2.0, y_scale = self._h / 2.0
        projected_tri[0, 0] += shift; projected_tri[0, 1] += shift
        projected_tri[1, 0] += shift; projected_tri[1, 1] += shift
        projected_tri[2, 0] += shift; projected_tri[2, 1] += shift

        projected_tri[0, 0] *= x_scale; projected_tri[0, 1] *= y_scale
        projected_tri[1, 0] *= x_scale; projected_tri[1, 1] *= y_scale
        projected_tri[2, 0] *= x_scale; projected_tri[2, 1] *= y_scale
        return np.asarray(projected_tri[:3, :3])

    cdef int[:, :] _compute_pixel_coords(self, float[:, :] tri):
        """
        Returns a grid map with (x, y) coordinates of pixels that lie within a rectangle which
        encases the `triangle`.

        Parameters
        ----------
        tri : np.ndarray of shape [3, 3]
            Triangle which pixels' coordinates to compute.
        buffer_size: tuple
            Contains height and width of the buffer.
        Returns
        -------
        np.ndarray of shape [n_pixels, 2, 1]
        """
        cdef:
            int h = self._h, w = self._w
            float[:] x = tri[:, 0], y = tri[:, 1]
            float _x_left = reduce_min(tri[:, 0]), _x_right = reduce_max(tri[:, 0])
            float _y_top = reduce_max(tri[:, 1]), _y_bot = reduce_min(tri[:, 2])
            int x_left = <int>ceil(_x_left), x_right = <int>ceil(_x_right)
            int y_top = <int>ceil(_y_top), y_bot = <int>ceil(_y_bot)

        x_left = clip(x_left, 0, w)
        x_right = clip(x_right, 0, w)

        y_top = clip(y_top, 0, h)
        y_bot = clip(y_bot, 0, h)

        cdef int[:, :, :] xy_slice = self._xy_grid[y_bot: y_top, x_left: x_right, :]
        return np.asarray(xy_slice).reshape(-1, 2)

    @cython.wraparound(False)
    cdef _compute_barycentric_coords(self, float[:, ::] tri, int[:, :] pixel_coords):
        """
        Computes barycentric coordinates for the given `pixel_coords` within the `triangle`
        Parameters
        ----------
        tri : np.ndarray of shape [3, 3]

        pixel_coords : np.ndarray of shape [n, 2]

        Returns
        -------
        np.ndarray  of shape [n, 3]
            Computed barycentric coords.
        """
        # Triangles coords
        cdef:
            # Pixels' coords
            int[:] x = pixel_coords[:, 0]
            int[:] y = pixel_coords[:, 1]
            float[:, :] bar = np.asarray(compute_bar_coords(tri, x, y), dtype='float32')
            int[:] select = np.empty(bar.shape[0], dtype='int32')
            size_t i, new_size = 0

        # Determine which pixels are within the triangle
        for i in range(bar.shape[0]):
            if bar[i, 0] > 0.0 and bar[i, 1] > 0.0 and bar[i, 2] > 0.0:
                select[i] = 1
                new_size += 1
                continue
            select[i] = 0

        # Select the necessary values
        cdef:
            float[:, :] bar_ = select_values_float(bar, select, new_size)
            int[:, :] pixel_coords_ = select_values_int(pixel_coords, select, new_size)
        # Select coords only for the encased pixels
        return np.asarray(bar_), np.asarray(pixel_coords_)

    def _fill_buffer(self, values: cnp.ndarray, pixel_coords: cnp.ndarray, bar_coords: cnp.ndarray, buffer: Buffer):
        """
        Fills in the the given `buffer` with the interpolated values of `values`.

        Parameters
        ----------
        values : np.ndarray of shape [3, d]
            Values that represent some d-dimensional characteristics of the triangle vertices.
        pixel_coords : np.ndarray of shape [n, 2]
            (x, y) coordinates if the pixels to fill in.
        bar_coords : np.ndarray of shape [n, 3]
            Corresponding barycentric coordinates of the pixel that will be used for interpolation of the `values`.
        """
        # Interpolation is done via weighting the values by their barycentric coordinates:
        # val' = l0 * val0 + l1 * val1 * l2 * val2
        # [n, 3] * [3, d] = [n, d]
        interpolated_values = np.dot(bar_coords, values)
        buffer[im_ind(pixel_coords)] = interpolated_values

    @cython.wraparound(False)
    @cython.overflowcheck(False)
    cdef _fill_z_buffer(self, float[:, :] tri, int[:, :] pixel_coords, float[:, :] bar_coords, z_buffer: Buffer):
        """
        Fills in the z-buffer by taking into account already existing values.

        Parameters
        ----------
        tri : np.ndarray of shape [3, 3]
            Triangle's vertices.
        pixel_coords : np.ndarray of shape [n, 2, 1]
            (x, y) coordinates of the pixels candidates to fill in.
        bar_coords : np.ndarray of shape [n, 3]
            Corresponding barycentric coordinates of the pixel that will be used for interpolation of z-coordinate.
        z_buffer : Buffer
            The z-buffer to fill in.
        """
        cdef:
            # If you dont initialize new_size with 0, it turns out that it equals 3. Surprise!
            size_t i, new_size = 0
            int x, y
            int[:] select = np.empty(shape=bar_coords.shape[0], dtype='int32')
            float z

        for i in range(bar_coords.shape[0]):
            # --- Depth interpolation
            # Save the results into bar_coords buffer to save memory
            z = bar_coords[i, 0] * tri[0, 2] + bar_coords[i, 1] * tri[1, 2] + bar_coords[i, 2] * tri[2, 2]
            if not (-1. <= z and z <= 1.):
                select[i] = 0
                continue

            x = pixel_coords[i, 0]
            y = pixel_coords[i, 1]
            # Check if new z is closer to the camera than the previous z
            if z_buffer[y, x] > z:
                z_buffer[y, x] = z
                select[i] = 1
                new_size += 1
            else:
                select[i] = 0

        # Separate variables are used because otherwise a memory leak happens.
        # I have no idea why is that so...
        cdef int[:, :] pixel_coords_ = select_values_int(pixel_coords, select, new_size)
        cdef float[:, :] bar_coords_ = select_values_float(bar_coords, select, new_size)
        # --- Return only 'visible' pixel coords
        return bar_coords_, pixel_coords_


if __name__ == '__main__':
    filler = AdvancedPixelBufferFiller()
    image_height = 10
    image_width = 10
    color_buffer = Buffer(image_height, image_width, dim=3, dtype='uint8')
    z_buffer = Buffer(image_height, image_width, dim=1, dtype='float32', init_val=100000)
    n_buffer = Buffer(image_height, image_width, dim=3, dtype='float32')

    import numpy as np

    tri = np.array([[0, 10, 1], [10, 0, 1], [0, 0, 1]], 'float32')
    colors = np.array([[0, 255, 1], [255, 0, 1], [0, 0, 1]], 'float32')
    normals = np.random.randn(3, 3).astype('float32')

    pixel_coords = filler._compute_pixel_coords(tri, color_buffer.get_size())
    print('shape', pixel_coords.shape)
    print('max\min', pixel_coords.max(), pixel_coords.min())

    pixel_bar_coords, pixel_coords = filler._compute_barycentric_coords(tri, pixel_coords)
    print('shape', pixel_coords.shape)
    print('shape', pixel_bar_coords.shape)
    print('xy max\min', pixel_coords.max(), pixel_coords.min())
    print('bar max\min', pixel_bar_coords.max(), pixel_bar_coords.min())
    pixel_bar_coords, pixel_coords = filler._fill_z_buffer(tri, pixel_coords, pixel_bar_coords, z_buffer)
    print('shape', pixel_coords.shape)
    print('shape', pixel_bar_coords.shape)
    print('xy max\min', pixel_coords.max(), pixel_coords.min())
    print('bar max\min', pixel_bar_coords.max(), pixel_bar_coords.min())

    import matplotlib.pyplot as plt

    z_buffer.clear()
    filler.compute_triangle_statistics(tri, colors, normals, color_buffer, z_buffer, n_buffer)
    plt.imshow(color_buffer.get_image())
    plt.show()
