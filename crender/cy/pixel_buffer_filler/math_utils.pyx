# cython: profile=False
from cython cimport cdivision, wraparound
import cython
import numpy as np


@cdivision(True)
cdef inline float bar_compute_single_coord(float l1, float l2, float l3, float a, float b, float x, float y):
    return (l1 * (y - a) - l2 * (x - b)) / l3


@wraparound(False)
cdef float[:, ::1] compute_bar_coords(float[:,:] tri, int[:] x, int[:] y):
    cdef:
        float[:, ::1] bar = np.empty(shape=[x.shape[0], 3], dtype='float32')

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

    for i in range(bar.shape[0]):
        x_val = <float>x[i]
        y_val = <float>y[i]
        bar[i, 0] = bar_compute_single_coord(l01, l02, l03, y2, x2, x_val, y_val)
        bar[i, 1] = bar_compute_single_coord(l11, l12, l13, y0, x0, x_val, y_val)
        bar[i, 2] = bar_compute_single_coord(l21, l22, l23, y1, x1, x_val, y_val)
    return bar


cdef inline float c_min(float a, float b):
    if a < b:
        return a
    else:
        return b


@wraparound(False)
cdef float reduce_min(float[:] arr):
    cdef:
        size_t i = 1
        float min_val = arr[0]

    for i in range(arr.shape[0]):
        min_val = c_min(arr[i], min_val)

    return min_val


cdef inline float c_max(float a, float b):
    if a > b:
        return a
    else:
        return b


@wraparound(False)
cdef float reduce_max(float[:] arr):
    cdef:
        size_t i = 1
        float max_val = arr[0]

    for i in range(arr.shape[0]):
        max_val = c_max(arr[i], max_val)

    return max_val


cdef int clip(int a, int min_val, int max_val):
    if a < min_val:
        return min_val
    if a > max_val:
        return max_val
    return a

@wraparound(False)
cdef void matmul(float[:,::1] a, float[:, ::1] b, float[:, ::1] out):
    cdef:
        size_t i, j, k
        size_t N = a.shape[0], M = b.shape[1], D = a.shape[1]

    for i in range(N):
        for j in range(M):
            for k in range(D):
                out[i, j] += a[i, k] * b[k, j]


@wraparound(False)
cdef void matmul_3x4(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out):
    # a - [3, 4]
    # b - [4, 4]
    # out - [3, 4]
    cdef:
        size_t i, j
    for i in range(3):
        for j in range(4):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j] + a[i, 3] * b[3, j]
