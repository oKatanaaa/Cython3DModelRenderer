# cython: profile=True
# distutils: extra_compile_args = /openmp
# distutils: extra_link_args = /openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp
cimport cython
from cython.parallel cimport prange, parallel
from libc.math cimport ceil
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
cimport openmp as omp

from .math_utils cimport clip, compute_bar_coords_single_pixel, Vec3

import numpy as np
from crender.py.data_structures import Model


cdef class AdvancedPixelBufferFiller:
    cdef:
        int h
        int w
        float fov
        float f
        float z_near
        float z_far
        float a
        int n_threads
        float[:, ::1] ones4
        float[:, ::1] proj_mat
        int[:, :, ::1] xy_grid
        float[:, ::1] projected_tri_buffer

        float[:, :, ::1] normals_buffer
        float[:, :, ::1] color_buffer
        float[:, ::1] z_buffer

        omp.omp_lock_t lock
        omp.omp_lock_t *lock_grid

    def __cinit__(self, h, w, fov=90.0, z_near=0.1, z_far=1000.0, n_threads=1):
        self.h = <int>h
        self.w = <int>w
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

        self.fov = <float>fov
        self.f = 1 / np.tan(self.fov / 2 / 180 * np.pi)
        self.z_near = <float>z_near
        self.z_far = <float>z_far
        # Aspect ratio
        self.a = h / w
        self._init_projection_matrix()

        self.n_threads = n_threads

        # Initialize buffers
        self.normals_buffer = np.zeros(shape=[h, w, 3], dtype='float32')
        self.color_buffer = np.zeros(shape=[h, w, 3], dtype='float32')
        self.z_buffer = np.ones(shape=[h, w], dtype='float32') * 1e6

        # Initialize the lock grid.
        # Using a single lock dampens multithreading performance drastically.
        # So instead an entire grid of locks is created so that each
        # pixel has its own lock. This way threads won't be blocking each other.
        self.lock_grid = <omp.omp_lock_t*>malloc(self.h * self.w * sizeof(omp.omp_lock_t))

        cdef int i
        for i in range(self.h * self.w):
            omp.omp_init_lock(&self.lock_grid[i])


    def get_size(self):
        return self.h, self.w

    cdef _init_projection_matrix(self):
        q = self.z_far / (self.z_far - self.z_near)
        self.proj_mat = np.array([
            [self.f / self.a,   0,                 0, 0],
            [0,                 self.f,            0, 0],
            [0,                 0,                 q, 1],
            [0,                 0,  -self.z_near * q, 0]
        ], dtype='float32')

    def render_model(self, model: Model):
        cdef:
            float[:, :, :] triangles = model._vertices_by_triangles.copy()
            float[:, :, :] colors = model._colors_by_triangles.copy()
            float[:, :, :] normals = model._normals_by_triangles.copy()
            int i

        self.project_on_screen_multithread(triangles, triangles)
        self.compute_triangle_statistics_multithread(
            triangles,
            colors,
            normals
        )

    cdef void project_on_screen_multithread(self, float[:, :, :] tris, float[:, :, :] buff):
        cdef:
            int ii, i, j, k
            float z, shift = 1.0, x_scale = self.w / 2.0, y_scale = self.h / 2.0

        with nogil, parallel(num_threads=self.n_threads):
            printf("_project_on_screen_multithread num_threads=%d thread_id=%d\n", omp.omp_get_num_threads(), omp.omp_get_thread_num())

            for ii in prange(tris.shape[0], schedule='static'):
                # Single triangle projection loop
                for i in range(3):
                    z = tris[ii, i, 2]
                    for j in range(3):
                        buff[ii, i, j] = tris[ii, i, 0] * self.proj_mat[0, j] + tris[ii, i, 1] * self.proj_mat[1, j] + \
                                         tris[ii, i, 2] * self.proj_mat[2, j] + self.proj_mat[3, j]
                    # Do perspective divide (normalize z value)
                    buff[ii, i, 0] = buff[ii, i, 0] / z
                    buff[ii, i, 1] = buff[ii, i, 1] / z
                    buff[ii, i, 2] = buff[ii, i, 2] / z

                    buff[ii, i, 0] = buff[ii, i, 0] + shift
                    buff[ii, i, 1] = buff[ii, i, 1] + shift

                    buff[ii, i, 0] = buff[ii, i, 0] * x_scale
                    buff[ii, i, 1] = buff[ii, i, 1] * y_scale

    cdef void _compute_pixel_coords_c(self, float *tri, int *out) nogil:
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
            int i
            int h = self.h, w = self.w
            float _x_left = self.w, _x_right = 0
            float _y_top = self.h, _y_bot = 0
            float x, y

        # Find the borders of the triangle
        for i in range(3):
            x = tri[i*3]
            y = tri[i*3 + 1]
            if x < _x_left:
                _x_left = x
            if x > _x_right:
                _x_right = x
            if y < _y_top:
                _y_top = y
            if y > _y_bot:
                _y_bot = y

        cdef:
            int x_left = <int>ceil(_x_left), x_right = <int>ceil(_x_right)
            int y_top = <int>ceil(_y_top), y_bot = <int>ceil(_y_bot)

        out[0] = clip(x_left, 0, w)
        out[1] = clip(x_right, 0, w)

        out[2] = clip(y_top, 0, h)
        out[3] = clip(y_bot, 0, h)

    cdef void compute_triangle_statistics_multithread(self,
                                        float[:, :, :] triangles,
                                        float[:, :, :] colors,
                                        float[:, :, :] normals) nogil:
        cdef:
            int x_left, x_right, y_top, y_bot
            # Loop variables
            int ii, x, y
            float mean_z
            # Normal values
            float n0, n1, n2
            # Color values
            float c0, c1, c2
            # Barycentric coordinates
            Vec3 bar
            float new_z
            omp.omp_lock_t *lock_grid = self.lock_grid
            int *borders_buffer

        with nogil, parallel(num_threads=self.n_threads):
            printf("num_threads=%d thread_id=%d\n", omp.omp_get_num_threads(), omp.omp_get_thread_num())
            borders_buffer = <int*>malloc(sizeof(int) * 4)

            for ii in prange(triangles.shape[0], schedule='dynamic'):

                if (normals[ii, 0, 2] + normals[ii, 1, 2] + normals[ii, 2, 2]) / 3 >= 0.0:
                    # Triangle is faced away from the camera
                    continue

                self._compute_pixel_coords_c(&triangles[ii, 0, 0], borders_buffer)
                x_left = borders_buffer[0]; x_right = borders_buffer[1]
                y_top = borders_buffer[2]; y_bot = borders_buffer[3]
                if x_left - x_right == 0 or y_top - y_bot == 0:
                    # The triangle is invisible, no need for further processing
                    continue

                for x in range(x_left, x_right):
                    for y in range(y_top, y_bot):
                        bar = compute_bar_coords_single_pixel(&triangles[ii, 0, 0], x, y)
                        if bar.x1 < 0.0 or bar.x2 < 0.0 or bar.x3 < 0.0:
                            continue
                        # --- Filling z-buffer
                        new_z = triangles[ii, 0, 2] * bar.x1 + triangles[ii, 1, 2] * bar.x2 + triangles[ii, 2, 2] * bar.x3
                        if not (-1.0 <= new_z or new_z <= 1.0):
                            continue

                        if new_z > self.z_buffer[y, x]:
                            continue

                        n0 = normals[ii, 0, 0] * bar.x1 + normals[ii, 1, 0] * bar.x2 + normals[ii, 2, 0] * bar.x3
                        n1 = normals[ii, 0, 1] * bar.x1 + normals[ii, 1, 1] * bar.x2 + normals[ii, 2, 1] * bar.x3
                        n2 = normals[ii, 0, 2] * bar.x1 + normals[ii, 1, 2] * bar.x2 + normals[ii, 2, 2] * bar.x3
                        c0 = colors[ii, 0, 0] * bar.x1 + colors[ii, 1, 0] * bar.x2 + colors[ii, 2, 0] * bar.x3
                        c1 = colors[ii, 0, 1] * bar.x1 + colors[ii, 1, 1] * bar.x2 + colors[ii, 2, 1] * bar.x3
                        c2 = colors[ii, 0, 2] * bar.x1 + colors[ii, 1, 2] * bar.x2 + colors[ii, 2, 2] * bar.x3
                        # Critical section: overriding z-buffer / color-buffer / normals buffer
                        omp.omp_set_lock(&lock_grid[y * self.w + x])
                        self.z_buffer[y, x] = new_z
                        self.color_buffer[y, x, 0] = c0
                        self.color_buffer[y, x, 1] = c1
                        self.color_buffer[y, x, 2] = c2

                        self.normals_buffer[y, x, 0] = n0
                        self.normals_buffer[y, x, 1] = n1
                        self.normals_buffer[y, x, 2] = n2
                        omp.omp_unset_lock(&lock_grid[y * self.w + x])

            free(<void*>borders_buffer)

    def get_normals_buffer(self):
        return np.asarray(self.normals_buffer)

    def get_color_buffer(self):
        return np.asarray(self.color_buffer)

    def get_z_buffer(self):
        return np.asarray(self.z_buffer)
