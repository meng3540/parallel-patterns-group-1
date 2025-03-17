# Optimization
In this section, we explore two methods that prove to be rather effective in optimizing the provided Kogge-Stone parallel scan algorithm.
### Increasing Parallelism
In CUDA programming, parallel is critical for maximizing GPU efficiency, and increasing parallelism means using more threads and blocks. The current implementation of the Kogge-Stone scan launches the CUDA kernel as follows:
```c=
Kogge_Stone_scan_kernel<<<(arraySize + SECTION_SIZE - 1) / SECTION_SIZE, SECTION_SIZE>>>(dev_x, dev_y, arraySize);
```
It is determined that
* The total number of blocks is calculated as `(arraySize + SECTION_SIZE - 1) / SECTION_SIZE`
### Optimizing Memory Access
