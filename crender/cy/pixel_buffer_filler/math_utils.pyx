# cython: profile=False
from cython cimport cdivision, wraparound, boundscheck, overflowcheck
cimport cython
import numpy as np


@boundscheck(False)
@cdivision(True)
@overflowcheck(False)
cdef Vec3 compute_bar_coords_single_pixel(float *tri, int x, int y) nogil:
    cdef:
        float x0 = tri[0], y0 = tri[1]
        float x1 = tri[3], y1 = tri[4]
        float x2 = tri[6], y2 = tri[7]
        # Precompute constants that take place in the barycentric coords formula
        float l01 = x1 - x2, l02 = y1 - y2
        float l03 = l01 * (y0 - y2) - l02 * (x0 - x2)

        float l11 = x2 - x0, l12 = y2 - y0
        float l13 = l11 * (y1 - y0) - l12 * (x1 - x0)

        float l21 = x0 - x1, l22 = y0 - y1
        float l23 = l21 * (y2 - y1) - l22 * (x2 - x1)
        Vec3 coords

    coords.x1 = bar_compute_single_coord(l01, l02, l03, y2, x2, <float>x, <float>y)
    coords.x2 = bar_compute_single_coord(l11, l12, l13, y0, x0, <float>x, <float>y)
    coords.x3 = bar_compute_single_coord(l21, l22, l23, y1, x1, <float>x, <float>y)
    return coords


@boundscheck(False)
@cdivision(True)
@overflowcheck(False)
cdef inline float bar_compute_single_coord(float l1, float l2, float l3, float a, float b, float x, float y) nogil:
    return (l1 * (y - a) - l2 * (x - b)) / l3


@boundscheck(False)
@wraparound(False)
cdef float[:, ::1] compute_bar_coords(float[:,:] tri, int[:, :] xy):
    cdef:
        float[:, ::1] bar = np.empty(shape=[xy.shape[0], 3], dtype='float32')

        float x0 = tri[0, 0], y0 = tri[0, 1]
        float x1 = tri[1, 0], y1 = tri[1, 1]
        float x2 = tri[2, 0], y2 = tri[2, 1]
        # Precompute constants that take place in the barycentric coords formula
        float l01 = x1 - x2, l02 = y1 - y2
        float l03 = l01 * (y0 - y2) - l02 * (x0 - x2)

        float l11 = x2 - x0, l12 = y2 - y0
        float l13 = l11 * (y1 - y0) - l12 * (x1 - x0)

        float l21 = x0 - x1, l22 = y0 - y1
        float l23 = l21 * (y2 - y1) - l22 * (x2 - x1)

        int i = 0

        float x_val, y_val

    # This does not have much of an effect on the overall performance.
    # I just wanted to try it out.
    #for i in prange(bar.shape[0], nogil=True, schedule='static'):
    for i in range(bar.shape[0]):
        x_val = <float>xy[i, 0]
        y_val = <float>xy[i, 1]
        bar[i, 0] = bar_compute_single_coord(l01, l02, l03, y2, x2, x_val, y_val)
        bar[i, 1] = bar_compute_single_coord(l11, l12, l13, y0, x0, x_val, y_val)
        bar[i, 2] = bar_compute_single_coord(l21, l22, l23, y1, x1, x_val, y_val)
    return bar


cdef inline float c_min(float a, float b) nogil:
    if a < b:
        return a
    else:
        return b


@wraparound(False)
cdef float reduce_min(float[:] arr) nogil:
    cdef:
        size_t i = 1
        float min_val = arr[0]

    for i in range(arr.shape[0]):
        min_val = c_min(arr[i], min_val)

    return min_val


cdef inline float c_max(float a, float b) nogil:
    if a > b:
        return a
    else:
        return b


@wraparound(False)
cdef float reduce_max(float[:] arr) nogil:
    cdef:
        size_t i = 1
        float max_val = arr[0]

    for i in range(arr.shape[0]):
        max_val = c_max(arr[i], max_val)

    return max_val




@wraparound(False)
cdef void matmul(float[:,::1] a, float[:, ::1] b, float[:, ::1] out):
    cdef:
        size_t i, j, k
        size_t N = a.shape[0], M = b.shape[1], D = a.shape[1]

    for i in range(N):
        for j in range(M):
            for k in range(D):
                out[i, j] += a[i, k] * b[k, j]


@boundscheck(False)
@wraparound(False)
@cdivision(True)
@cython.overflowcheck(False)
cdef void project_triangle(float[:, :] tri, float[:, :] projection_mat, float[:, :] out) nogil:
    # a - [3, 3]
    # b - [4, 4]
    # out - [3, 3]
    cdef:
        size_t i, j
        float z
    for i in range(3):
        for j in range(3):
            out[i, j] = tri[i, 0] * projection_mat[0, j] + tri[i, 1] * projection_mat[1, j] + tri[i, 2] * projection_mat[2, j] + projection_mat[3, j]
        # Do perspective divide (normalize z value)
        z = tri[i, 2] + 1e-6
        out[i, 0] /= z
        out[i, 1] /= z
        out[i, 2] /= z
