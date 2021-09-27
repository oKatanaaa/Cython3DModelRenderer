# cython: profile=False
from cython cimport cdivision, boundscheck, overflowcheck


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

