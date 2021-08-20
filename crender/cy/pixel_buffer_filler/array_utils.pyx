cimport numpy as cnp
from cython cimport wraparound
import numpy as np


@wraparound(False)
cdef float[:, :] select_values_float(float[:, :] arr, int[:] select, size_t new_size):
    cdef:
        size_t i, j = 0
        float[:, :] new_arr = np.empty(shape=[new_size, arr.shape[1]], dtype='float32')
    for i in range(arr.shape[0]):
        if select[i] == 1:
            new_arr[j, ...] = arr[i, ...]
            j += 1
    return new_arr


@wraparound(False)
cdef int[:, :] select_values_int(int[:, :] arr, int[:] select, size_t new_size):
    cdef:
        size_t i, j = 0
        int[:, :] new_arr = np.empty(shape=[new_size, arr.shape[1]], dtype='int32')
    for i in range(arr.shape[0]):
        if select[i] == 1:
            new_arr[j, ...] = arr[i, ...]
            j += 1
    return new_arr