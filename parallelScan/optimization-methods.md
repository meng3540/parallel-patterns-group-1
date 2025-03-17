# Optimization
In this section, we explore two methods that prove to be rather effective in optimizing the provided Kogge-Stone parallel scan algorithm.
### Increasing Parallelism
In CUDA programming, parallelism is critical for maximizing GPU efficiency, and increasing parallelism means using more threads and blocks. The current implementation of the Kogge-Stone scan launches the CUDA kernel as follows:
```c=
Kogge_Stone_scan_kernel<<<(arraySize + SECTION_SIZE - 1) / SECTION_SIZE, SECTION_SIZE>>>(dev_x, dev_y, arraySize);
```
It is determined that:
* The total number of blocks is calculated as `(arraySize + SECTION_SIZE - 1) / SECTION_SIZE`.
* Each block contains `SECTION_SIZE` threads.
* Each thread processes one element of the array.

The limitations of this approach lie in (1) the under-utilization of streaming processors (SMs) and (2) the inefficient handling of large arrays. A single block processes a maximum of `SECTION_SIZE` elements. If `SECTION_SIZE=32` and `arraySize=1,000,000`, then only 31,250 blocks are launched. Since modern GPUs have thousands of CUDA cores across many SMs, a small number of blocks means some SMs remain idle. 

### Optimizing Memory Access
