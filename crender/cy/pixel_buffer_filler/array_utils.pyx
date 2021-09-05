# cython: profile=False
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from cython cimport wraparound
import numpy as np


cdef float* allocate_float_buffer(size_t n):
    cdef float *buffer = <float*>malloc(n * sizeof(float))
    return buffer


cdef float[:, :] allocate_float_mat(size_t n, size_t m):
    return <float[:n, :m]>allocate_float_buffer(n * m)


@wraparound(False)
cdef float[:, ::1] select_values_float(float[:, :] arr, int[::1] select, size_t new_size):
    cdef:
        size_t i, j = 0
        float[:, ::1] new_arr = np.empty(shape=(new_size, arr.shape[1]), dtype='float32')
    for i in range(arr.shape[0]):
        if select[i] == 1:
            new_arr[j, ...] = arr[i, ...]
            j += 1
    return new_arr


@wraparound(False)
cdef int[:, ::1] select_values_int(int[:, :] arr, int[::1] select, size_t new_size):
    cdef:
        size_t i, j = 0
        int[:, ::1] new_arr = np.empty(shape=[new_size, arr.shape[1]], dtype='int32')
    for i in range(arr.shape[0]):
        if select[i] == 1:
            new_arr[j, ...] = arr[i, ...]
            j += 1
    return new_arr
