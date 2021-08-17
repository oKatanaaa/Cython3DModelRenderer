# cython: profile=True

from libc.stdlib cimport malloc
from cython cimport cdivision
import numpy as np

cdef float* allocate_float_buffer(size_t n):
    cdef float *buffer = <float*>malloc(n * sizeof(float))
    return buffer


@cdivision(True)
cdef inline float bar_compute_single_coord(float l1, float l2, float l3, float a, float b, float x, float y):
    return (l1 * (y - a) - l2 * (x - b)) / l3

cdef float[:, :] compute_bar_coords(float[:,:] tri, int[:] x, int[:] y):
    cdef:
        float[:, :] bar = np.empty(shape=[x.shape[0], 3], dtype='float32')

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

    for i in range(bar.shape[0]):
        bar[i, 0] = bar_compute_single_coord(l01, l02, l03, y2, x2, <float>x[i], <float>y[i])
        bar[i, 1] = bar_compute_single_coord(l11, l12, l13, y0, x0, <float>x[i], <float>y[i])
        bar[i, 2] = bar_compute_single_coord(l21, l22, l23, y1, x1, <float>x[i], <float>y[i])
    return bar
