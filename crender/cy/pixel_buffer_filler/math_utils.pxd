cdef float[:, ::1] compute_bar_coords(float[:, :] tri, int[:] x, int[:] y)

cdef float reduce_min(float[:] arr)
cdef float reduce_max(float[:] arr)

cdef inline int clip(int a, int min_val, int max_val):
    if a < min_val:
        return min_val
    if a > max_val:
        return max_val
    return a

cdef void matmul(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out)

cdef void matmul_3x4(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out)
