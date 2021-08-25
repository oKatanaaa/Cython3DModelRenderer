cdef float[:, ::1] compute_bar_coords(float[:, :] tri, int[:] x, int[:] y)

cdef float reduce_min(float[:] arr)
cdef float reduce_max(float[:] arr)

cdef int clip(int a, int min_val, int max_val)

cdef void matmul(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out)

cdef void matmul_3x4(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out) nogil