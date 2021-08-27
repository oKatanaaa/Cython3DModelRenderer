# cython: profile=True

cimport numpy as cnp
cimport cython
from libc.math cimport ceil

from .math_utils cimport compute_bar_coords, reduce_min, reduce_max, clip, matmul_3x4

import numpy as np
from crender.py.data_structures import Buffer



cdef:
    int NO_VISIBLE_PIXELS = -1
    int HAS_VISIBLE_PIXELS = 1


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
        float[:, ::1] _proj_mat
        int[:, :, ::1] _xy_grid
        float[:, ::1] _projected_tri_buffer

        float[:, :, ::1] normals_buffer
        float[:, :, ::1] color_buffer
        float[:, ::1] z_buffer

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
        # Initialize buffers
        self.normals_buffer = np.zeros(shape=[h, w, 3], dtype='float32')
        self.color_buffer = np.zeros(shape=[h, w, 3], dtype='float32')
        self.z_buffer = np.ones(shape=[h, w], dtype='float32') * 1e6

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

    # --- Performance note
    # If you type all the arguments here as a typed memoryview (float[:, :]),
    # the overall performance will drop a bit. That's because a lot of memoryview initializations
    # are happening. It is okay when a memoryview is being initialized once somewhere,
    # but in this case it happens A LOT and takes up a considerable amount of time.
    def compute_triangle_statistics(self,
                                    cnp.ndarray[float, ndim=2] triangle,
                                    cnp.ndarray[float, ndim=2] colors,
                                    cnp.ndarray[float, ndim=2] normals):
        # 1. Determine the area of interest
        # 2. Compute barycentric coordinates
        # 3. Fill in z-buffer and determine which pixels are visible
        # 3. Fill in other buffers
        # (Task #13)
        cdef float mean_z = (normals[0, 2] + normals[1, 2] + normals[2, 2]) / 3.0
        if mean_z >= 0.0:
            # The triangle faces away from the camera, so don't need to draw it
            return

        cdef:
            float[:, :] projected_tri = self._project_on_screen(triangle)
            int[:, :] pixel_coords = self._compute_pixel_coords(projected_tri)
            float[:, ::1] bar_coords
            int[::1] select

        if pixel_coords.shape[0] == 0:
            # The triangle may be invisible, no need for further processing
            return

        # Returns coords only for those pixels that lie within the triangle
        bar_coords, select = self._compute_barycentric_coords(projected_tri, pixel_coords)

        if bar_coords.shape[0] == 0:
            # No pixels are visible
            return

        # Returns coords only for visible pixels (that are not behind something)
        cdef int return_code = self._fill_z_buffer(projected_tri, pixel_coords, bar_coords, select)

        if return_code == NO_VISIBLE_PIXELS:
            return

        self._fill_buffer_3d(colors, pixel_coords, bar_coords, select, self.color_buffer)
        self._fill_buffer_3d(normals, pixel_coords, bar_coords, select, self.normals_buffer)

    cdef float[:, :] _project_on_screen(self, float[:,:] tri):
        # TODO
        # Optimize the way the triangle is being stored. Avoid slicing at the end.
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
        return projected_tri[:3, :3]

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
            # --- Performance note
            # Those memoryviews (x and y) were created specifically to avoid multiple slicing
            # as I suspected it would take up some time.
            # But even though I made those local memoryview, I forgot to use them and the code
            # looked something like this:
            # float _x_left = reduce_min(tri[:, 0]), _x_right = reduce_max(tri[:, 0])
            # float _y_top = reduce_max(tri[:, 1]), _y_bot = reduce_min(tri[:, 1])
            # When I fixed it, the performance improved from 0.275s to 0.199s. That's freaking a lot!
            # I keep getting amazed by how much of the devil in the details there is when writing
            # a highly performant cython code.
            float _x_left = reduce_min(x), _x_right = reduce_max(x)
            float _y_top = reduce_max(y), _y_bot = reduce_min(y)
            int x_left = <int>ceil(_x_left), x_right = <int>ceil(_x_right)
            int y_top = <int>ceil(_y_top), y_bot = <int>ceil(_y_bot)

        x_left = clip(x_left, 0, w)
        x_right = clip(x_right, 0, w)

        y_top = clip(y_top, 0, h)
        y_bot = clip(y_bot, 0, h)

        cdef int[:, :, :] xy_slice = self._xy_grid[y_bot: y_top, x_left: x_right, :]
        return np.asarray(xy_slice).reshape(-1, 2)

    @cython.wraparound(False)
    cdef _compute_barycentric_coords(self, float[:, :] tri, int[:, :] pixel_coords):
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
            float[:, ::1] bar = compute_bar_coords(tri, pixel_coords)
            int[::1] select = np.empty(bar.shape[0], dtype='int32')
            size_t i

        # Determine which pixels are within the triangle
        for i in range(bar.shape[0]):
            if bar[i, 0] > 0.0 and bar[i, 1] > 0.0 and bar[i, 2] > 0.0:
                select[i] = 1
                continue
            select[i] = 0

        return bar, select

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.overflowcheck(False)
    cdef void _fill_buffer_3d(
            self,
            cnp.ndarray[float, ndim=2] values,
            int[:, :] pixel_coords,
            float[:, ::1] bar_coords,
            int[::1] select,
            float[:, :, ::1] buffer
    ):
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
        cdef:
            size_t i
            int x, y
            float l1, l2, l3

        for i in range(pixel_coords.shape[0]):
            if select[i] == 0:
                continue

            x = pixel_coords[i, 0]; y = pixel_coords[i, 1]
            l1 = bar_coords[i, 0]; l2 = bar_coords[i, 1]; l3 = bar_coords[i, 2]
            buffer[y, x, 0] = values[0, 0] * l1 + values[1, 0] * l2 + values[2, 0] * l3
            buffer[y, x, 1] = values[0, 1] * l1 + values[1, 1] * l2 + values[2, 1] * l3
            buffer[y, x, 2] = values[0, 2] * l1 + values[1, 2] * l2 + values[2, 2] * l3

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.overflowcheck(False)
    cdef int _fill_z_buffer(self, float[:, :] tri, int[:, :] pixel_coords, float[:, ::1] bar_coords, int[::1] select):
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
            int x, y, return_code = NO_VISIBLE_PIXELS
            float z

        for i in range(bar_coords.shape[0]):
            if select[i] == 0:
                continue
            # --- Depth interpolation
            # Save the results into bar_coords buffer to save memory
            z = bar_coords[i, 0] * tri[0, 2] + bar_coords[i, 1] * tri[1, 2] + bar_coords[i, 2] * tri[2, 2]
            if not (-1. <= z <= 1.):
                select[i] = 0
                continue

            x = pixel_coords[i, 0]
            y = pixel_coords[i, 1]
            # Check if new z is closer to the camera than the previous z
            if self.z_buffer[y, x] > z:
                self.z_buffer[y, x] = z
                return_code = HAS_VISIBLE_PIXELS
            else:
                select[i] = 0
        return return_code

    def get_normals_buffer(self):
        return np.asarray(self.normals_buffer)

    def get_color_buffer(self):
        return np.asarray(self.color_buffer)

    def get_z_buffer(self):
        return np.asarray(self.z_buffer)
