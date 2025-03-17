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

The main limitations of this approach are:

1. Under-utilization of Streaming Multiprocessors (SMs): Modern GPUs have hundreds or even thousands of cores spread across multiple SMs. If we only launch a limited number of blocks, some SMs may remain idle or underused. For example, with `SECTION_SIZE = 32` and `arraySize = 1,000,000`, we get only 31,250 blocks, which may not fully use all available SMs.

2. Inefficient Handling of Large Arrays: Since each block processes only `SECTION_SIZE` elements, handling very large datasets requires launching a huge number of blocks. However, because these blocks operate independently, they do not share information beyond their own shared memory. This means an extra computation step is needed to merge the partial results from different blocks.

### Optimizing Memory Access
