cdef struct Vec3:
    float x1
    float x2
    float x3

cdef Vec3 compute_bar_coords_single_pixel(float *tri, int x, int y) nogil

cdef inline int clip(int a, int min_val, int max_val) nogil:
    if a < min_val:
        return min_val
    if a > max_val:
        return max_val
    return a
