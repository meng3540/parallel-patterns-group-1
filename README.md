# Introduction
>1) What are parallel patterns in computing?
>2) What is their significance?- Explain in the context of typical applications they are used in.
>3) How is heterogeneous GPU-CPU computing useful in solving a parallel pattern?

In computing, patterns are a way templates of the best practices used in software engineering. They represent recurring themes and techniques that can be reused to solve different problems. The same concept applies to parallel programming where parallel patterns refer to recurring combinations of task distribution and data access that are used to solve parallel programming problems.  

Parallel patterns are significant because they offer proven techniques for solving parallel programming tasks and making them more manageable. They're used for a variety of applications such as image processing, data analytics, and AI training, where parallelizing the task speeds up processing.

Heterogeneous GPU-CPU computing is useful in solving parallel patterns because it leverages the strengths of both systems. It does this by assigning computationally heavy tasks to the GPU, as it can process multiple threads simultaneously, while delegating sequential operations to the CPU to ensure better resource utilization and more efficient processing.

## Parallel Scan / Prefix Sum Pattern

**Overview & Applications:**  
The Parallel Scan, also known as the Prefix Sum pattern, is a fundamental building block in parallel computing. It takes an input array and produces an output array where each element is the cumulative result (often a sum) of all preceding elements in the array. This pattern isn’t just limited to addition—it can be applied to any associative operation (like multiplication, maximum, or even custom operations). Its versatility makes it useful in a variety of applications, including:

- **Data Processing:** Cumulative sums, running totals, and histogram building.
- **Algorithmic Building Blocks:** Used in stream compaction, sorting algorithms (e.g., radix sort), and computing recurrence relations.
- **Graphics & Simulation:** Parallel computations in graphics processing, physics simulations, and more.

**Basic Algorithm Description:**  
The parallel prefix sum algorithm typically operates in two main phases:

1. **Up-Sweep (Reduce) Phase:**  
   - **Process:** The algorithm organizes the data into a balanced binary tree structure. Starting from the leaves (the input array), each pair of elements is combined (e.g., summed) and stored in their parent node.
   - **Goal:** This phase computes partial sums, propagating information upward until the root holds the total sum of the entire array.
   - **Complexity:** This phase requires O(log n) steps, where n is the number of elements.

2. **Down-Sweep Phase:**  
   - **Process:** With the partial sums computed, the algorithm then traverses back down the tree. It uses the previously computed values to determine the final prefix sums for each element.
   - **Result:** Depending on the variant, the algorithm can produce an inclusive scan (where each output includes the current element) or an exclusive scan (where each output excludes the current element).
   - **Efficiency:** Like the up-sweep, this phase also completes in O(log n) steps.

**Rationale for Parallel Processing & Hardware Implementation:**  
- **Intrinsic Parallelism:**  
  The key advantage of the prefix sum pattern is that its operations can be performed concurrently. During both phases, many independent operations (like summing pairs) can be executed simultaneously, taking full advantage of multi-core processors and SIMD (Single Instruction, Multiple Data) architectures.

- **Performance Benefits:**  
  While a sequential prefix sum algorithm runs in O(n) time, the parallel version can reduce the time complexity to O(log n) with sufficient hardware resources. This logarithmic speedup is critical for large-scale computations.

- **Hardware Utilization:**  
  Modern GPUs and multi-core CPUs are specifically designed to handle many small, simultaneous operations. Implementing the prefix sum algorithm on such hardware not only leverages their parallel processing capabilities but also lays the groundwork for more complex parallel algorithms that rely on efficient data aggregation.

In summary, the Parallel Scan/Prefix Sum pattern is an essential, highly efficient technique in parallel computing. It transforms sequential accumulation into a concurrent process, enabling significant performance improvements for a wide range of applications, from simple data processing to complex algorithmic tasks.

# Optimization
One effective method to optimize the provided Kogge-Stone scan algorithm is reducing the number of global memory accesses by improving shared memory usage.

In the basic code, each thread reads from the global memory into shared memory (XY[threadIdx.x] = X[i];), performs the scan, and writes back the result (Y[i] = XY[threadIdx.x];). While shared memory is used for intermediate results in the scan, the final result is written back to global memory one thread at a time.

**Coalesced Memory Writes and Improved Global Memory Access Patterns**
Currently, the write pattern to global memory (Y[i] = XY[threadIdx.x]) is done by each thread individually, which could be inefficient if threads in the same warp are accessing non-adjacent memory locations. This can lead to uncoalesced memory accesses, slowing down the overall performance.

We can optimize the memory access pattern by writing the results back to global memory in a coalesced manner, ensuring that threads in a warp access consecutive memory locations. This is especially important for memory-bound operations like prefix sums. The optimized idea is to increase coalescing by having multiple threads write to a single memory location at once in some cases.

**Steps for Optimization:**
- Merge Global Memory Writes: We can reduce the number of global memory writes by allowing multiple threads to store their results together, minimizing the number of accesses. This approach is efficient when the number of threads is large enough to ensure that threads in the same warp write to consecutive locations in global memory.

- Use Warp-Level Synchronization: Since the Kogge-Stone scan is a parallel prefix sum, we can use warp-level synchronization and atomic operations to handle the final results efficiently. By minimizing the amount of global memory interaction, we can exploit the high-throughput of shared memory and reduce the bottleneck caused by global memory accesses.
