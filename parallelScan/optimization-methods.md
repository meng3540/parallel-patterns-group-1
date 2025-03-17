# Optimization
In this section, we explore two methods that prove to be rather effective in optimizing the provided Kogge-Stone parallel scan algorithm.
### Increasing Parallelism
In CUDA programming, parallelism is critical for maximizing GPU efficiency, and increasing parallelism means using more threads and blocks. The current implementation of the Kogge-Stone scan launches the CUDA kernel as follows:
```c=
Kogge_Stone_scan_kernel<<<(arraySize + SECTION_SIZE - 1) / SECTION_SIZE, SECTION_SIZE>>>(dev_x, dev_y, arraySize);
```
This means:
* The total number of blocks is calculated as `(arraySize + SECTION_SIZE - 1) / SECTION_SIZE`.
* Each block contains `SECTION_SIZE` threads.
* Each thread processes one element of the array.

The main limitations of this approach are:

1. **Under-utilization of Streaming Multiprocessors (SMs)**: Modern GPUs have hundreds or even thousands of cores spread across multiple SMs. If we only launch a limited number of blocks, some SMs may remain idle or underused. For example, with `SECTION_SIZE = 32` and `arraySize = 1,000,000`, we get only 31,250 blocks, which may not fully use all available SMs.

2. **Inefficient Handling of Large Arrays**: Since each block processes only `SECTION_SIZE` elements, handling very large datasets requires launching a huge number of blocks. However, because these blocks operate independently, they do not share information beyond their own shared memory. This means an extra computation step is needed to merge the partial results from different blocks.

Our goal is to develop a work-efficient scan algorithm that takes advantage of the sequential algorithm's efficiency and the GPU's parallelism. We can achieve this by using an algorithmic pattern called **Binary Balanced Sum**. The input array elements are arranged into a binary tree structure, each being a leaf in this tree. A binary tree with $n$ leaves has $d=log_2n$ levels, and each level $d$ has $2^d$ nodes. At each level, we perform addition operations to compute partial sums. This tree-like approach allows multiple additions to happen simultaneously.

The algorithm includes two phases:
1. Reduce Phase (Up-Sweep): We start at the bottom (leaves) and move up to the top (root). At each step, we sum pairs of elements and store the sums at the next level up. This continues until we reach the top, which contains the total sum of the array.
![image](https://github.com/user-attachments/assets/2e373fed-afc3-4bc8-bd7a-6c46636262f9)
 
  
3. Down-Sweep:
   
