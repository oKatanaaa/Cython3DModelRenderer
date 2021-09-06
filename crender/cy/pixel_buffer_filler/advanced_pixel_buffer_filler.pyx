# cython: profile=True
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

cimport numpy as cnp
cimport cython
from cython.parallel cimport prange, parallel
from libc.math cimport ceil
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
cimport openmp as omp

from .math_utils cimport compute_bar_coords, reduce_min, reduce_max, clip, project_triangle, \
    compute_bar_coords_single_pixel, Vec3
from .array_utils cimport allocate_float_mat, allocate_int_vec

import numpy as np
from crender.py.data_structures import Buffer, Model



cdef:
    int NO_VISIBLE_PIXELS = -1
    int HAS_VISIBLE_PIXELS = 1


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
        x, y = np.meshgrid(x_coords, y_coords)
        self.xy_grid = np.stack([x, y], axis=-1)
        self.fov = <float>fov
        self.f = 1 / np.tan(self.fov / 2 / 180 * np.pi)
        self.z_near = <float>z_near
        self.z_far = <float>z_far
        # Aspect ratio
        self.a = h / w
        self.projected_tri_buffer = np.empty(shape=[4, 4], dtype='float32')
        self._init_projection_matrix()

        self.n_threads = n_threads

        # Initialize buffers
        self.normals_buffer = np.zeros(shape=[h, w, 3], dtype='float32')
        self.color_buffer = np.zeros(shape=[h, w, 3], dtype='float32')
        self.z_buffer = np.ones(shape=[h, w], dtype='float32') * 1e6

        omp.omp_init_lock(&self.lock)

    def get_size(self):
        return self.h, self.w

    cdef _init_projection_matrix(self):
        q = self.z_far / (self.z_far - self.z_near)
        self.proj_mat = np.array([
            [self.f / self.a, 0, 0, 0],
            [0, self.f, 0, 0],
            [0,                     0,                  q, 1],
            [0, 0, -self.z_near * q, 0]
        ], dtype='float32')

        self.ones4 = np.ones((3, 4), dtype='float32')

    def render_model(self, model: Model):
        cdef:
            float[:, :, :] triangles = model._vertices_by_triangles
            float[:, :, :] colors = model._colors_by_triangles
            float[:, :, :] normals = model._normals_by_triangles
            size_t i

        cdef:
            float[:, :] projected_tri_buff
            int[:] borders_buff
        print('nthreads', self.n_threads)
        with nogil, parallel():
            omp.omp_set_dynamic(self.n_threads)
            self.work(triangles, colors, normals)

    @cython.boundscheck(False)
    cdef void work(self, float[:, :, :] triangles, float[:, :, :] colors, float[:, :, :] normals) nogil:
        cdef:
            float[:, :] projected_tri_buff
            int[:] borders_buff
            size_t i
        projected_tri_buff = allocate_float_mat(3, 4)
        borders_buff = allocate_int_vec(4)
        printf("%d", omp.omp_get_num_threads())
        for i in prange(triangles.shape[0], schedule='static'):
            self._project_on_screen(triangles[i], projected_tri_buff)
            self.compute_triangle_statistics(
                projected_tri_buff,
                colors[i],
                normals[i],
                borders_buff
            )

        free(<void *>&projected_tri_buff[0,0])
        free(<void *>&borders_buff[0])

    cdef void compute_triangle_statistics(self,
                                    float[:, :] triangle,
                                    float[:, :] colors,
                                    float[:, :] normals,
                                    int[:] borders_buffer) nogil:
        # 1. Determine the area of interest
        # 2. Compute barycentric coordinates
        # 3. Fill in z-buffer and determine which pixels are visible
        # 3. Fill in other buffers
        # (Task #13)
        cdef float mean_z = (normals[0, 2] + normals[1, 2] + normals[2, 2]) / 3.0
        if mean_z >= 0.0:
            # The triangle faces away from the camera, so don't need to draw it
            return

        self._compute_pixel_coords(triangle, borders_buffer)
        cdef:
            size_t x_left = borders_buffer[0], x_right = borders_buffer[1]
            size_t y_top = borders_buffer[2], y_bot = borders_buffer[3]
            size_t x, y
            Vec3 bar
            float new_z
            omp.omp_lock_t lock = self.lock
            # Normal values
            float n0, n1, n2
            # Color values
            float c0, c1, c2

        # TODO
        # This is an incorrect check, fix it
        if x_left - x_right == 0 or y_top - y_bot == 0:
            # The triangle is invisible, no need for further processing
            printf('%d %d %d %d \n', x_left, x_right, y_top, y_bot)
            return

        for x in range(x_left, x_right):
            for y in range(y_top, y_bot):
                bar = compute_bar_coords_single_pixel(triangle, x, y)
                if bar.x1 < 0.0 or bar.x2 < 0.0 or bar.x3 < 0.0:
                    printf('Bar coords are negative')
                    continue
                # --- Filling z-buffer
                new_z = triangle[0, 2] * bar.x1 + triangle[1, 2] * bar.x2 + triangle[2, 2] * bar.x3
                if not (-1.0 < new_z or new_z < 1.0):
                    printf('New z is not within the interval')
                    continue

                if new_z < self.z_buffer[y, x]:
                    # Critical section: overriding z-buffer
                    omp.omp_set_lock(&lock)
                    self.z_buffer[y, x] = new_z
                    omp.omp_unset_lock(&lock)
                    printf('Set!')
                else:
                    printf('New z is larger')
                    continue

                # --- Filling normals buffer
                n0 = normals[0, 0] * bar.x1 + normals[1, 0] * bar.x2 + normals[2, 0] * bar.x3
                n1 = normals[0, 1] * bar.x1 + normals[1, 1] * bar.x2 + normals[2, 1] * bar.x3
                n2 = normals[0, 2] * bar.x1 + normals[1, 2] * bar.x2 + normals[2, 2] * bar.x3
                # Critical section: overriding z-buffer
                omp.omp_set_lock(&lock)
                self.normals_buffer[y, x, 0] = n0
                self.normals_buffer[y, x, 1] = n1
                self.normals_buffer[y, x, 2] = n2
                omp.omp_unset_lock(&lock)

                # --- Filling color buffer
                c0 = colors[0, 0] * bar.x1 + colors[1, 0] * bar.x2 + colors[2, 0] * bar.x3
                c1 = colors[0, 1] * bar.x1 + colors[1, 1] * bar.x2 + colors[2, 1] * bar.x3
                c2 = colors[0, 2] * bar.x1 + colors[1, 2] * bar.x2 + colors[2, 2] * bar.x3
                # Critical section: overriding z-buffer
                omp.omp_set_lock(&lock)
                self.color_buffer[y, x, 0] = c0
                self.color_buffer[y, x, 1] = c1
                self.color_buffer[y, x, 2] = c2
                omp.omp_unset_lock(&lock)
                printf('Set!!!')

    cdef void _project_on_screen(self, float[:,:] tri, float[:, :] buffer) nogil:
        # TODO
        # Optimize the way the triangle is being stored. Avoid slicing at the end.
        self.ones4[:3, :3] = tri
        # --- Perspective projection
        # Projects vertices onto the screen plane and makes them to be in
        # range [-1, 1]. Vertices outside of that range are invisible.
        project_triangle(tri, self.proj_mat, buffer)

        # --- Stretching the objects along the screen.
        # The same as:
        # 1. multiply by w/2, h/2
        # 2. shift by w/2, h/2 into the center of the screen
        cdef float shift = 1.0, x_scale = self.w / 2.0, y_scale = self.h / 2.0
        buffer[0, 0] += shift; buffer[0, 1] += shift
        buffer[1, 0] += shift; buffer[1, 1] += shift
        buffer[2, 0] += shift; buffer[2, 1] += shift

        buffer[0, 0] *= x_scale; buffer[0, 1] *= y_scale
        buffer[1, 0] *= x_scale; buffer[1, 1] *= y_scale
        buffer[2, 0] *= x_scale; buffer[2, 1] *= y_scale

    cdef void _compute_pixel_coords(self, float[:, :] tri, int[:] out) nogil:
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
            int h = self.h, w = self.w
            float[:] x = tri[:, 0], y = tri[:, 1]
            float _x_left = reduce_min(x), _x_right = reduce_max(x)
            float _y_top = reduce_max(y), _y_bot = reduce_min(y)
            int x_left = <int>ceil(_x_left), x_right = <int>ceil(_x_right)
            int y_top = <int>ceil(_y_top), y_bot = <int>ceil(_y_bot)

        out[0] = clip(x_left, 0, w)
        out[1] = clip(x_right, 0, w)

        out[2] = clip(y_top, 0, h)
        out[3] = clip(y_bot, 0, h)

    def get_normals_buffer(self):
        return np.asarray(self.normals_buffer)

    def get_color_buffer(self):
        return np.asarray(self.color_buffer)

    def get_z_buffer(self):
        return np.asarray(self.z_buffer)
