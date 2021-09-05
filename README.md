# Cython3DModelRenderer


This repo contains a Cython implementation of the following repository (which is written purely in Python):
- *to be revealed*

The project was done solely for education purposes: 3d graphics, Cython, unit testing. 
All the code can be used as a reference to help you understand the math and algorithms.

The repository contains several versions of the same renderer that vary in their implementation and, therefore, 
performance:
- Version A. This version mainly preserves the way all the computation is done in the Python-only implementation. It has
a 9x performance boost compared to Python-only implementation. The version is memory bounded (there is a lot of 
memory allocation going on) and it is very hard to optimize it further without refactoring the memory management.
So one of the conclusion we can draw regarding why some well-optimized numpy code may run not as fast as it could, is
that the computation involves a lot of memory allocation. Since allocating memory is a costly operation, it dampens
the speed drastically. You have to use preallocated buffers for storing intermediate results. This way the code will
run much, much faster.

## Performance

To be measured...
