cdef float[:, ::1] compute_bar_coords(float[:, :] tri, int[:, :] xy)

cdef float reduce_min(float[:] arr)
cdef float reduce_max(float[:] arr)

cdef int clip(int a, int min_val, int max_val)

cdef void matmul(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out)

cdef void project_triangle(float[:, ::1] tri, float[:, ::1] projection_mat, float[:, ::1] out) nogil