cdef struct Vec3:
    float x1
    float x2
    float x3

cdef Vec3 compute_bar_coords_single_pixel(float[:, :] tri, int x, int y) nogil
cdef float[:, ::1] compute_bar_coords(float[:, :] tri, int[:, :] xy)

cdef float reduce_min(float[:] arr) nogil
cdef float reduce_max(float[:] arr) nogil

cdef int clip(int a, int min_val, int max_val) nogil

cdef void matmul(float[:, ::1] a, float[:, ::1] b, float[:, ::1] out)

cdef void project_triangle(float[:, :] tri, float[:, :] projection_mat, float[:, :] out) nogil