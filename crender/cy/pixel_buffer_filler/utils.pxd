cdef float[:, :] compute_bar_coords(float[:,:] tri, int[:] x, int[:] y)

cdef float reduce_min(float[:] arr)
cdef float reduce_max(float[:] arr)

cdef int clip(int a, int min_val, int max_val)
