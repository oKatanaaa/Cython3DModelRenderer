cdef float[:, ::1] select_values_float(float[:, :] arr, int[::1] select, size_t new_size)
cdef int[:, ::1] select_values_int(int[:, :] arr, int[::1] select, size_t new_size)