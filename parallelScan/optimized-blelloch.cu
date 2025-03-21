#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SECTION_SIZE 1024 /* Define the size of each block section */
#define N 65536 /* Define the size of the array */

cudaError_t launch_Blelloch_scan_kernel(float* x, float* y, unsigned int arraySize);

/*
    CUDA kernel implementing the Blelloch scan algorithm with single-pass multi-block execution.
    This performs an exclusive scan (prefix sum) across the input array in parallel.
*/
__global__ void Blelloch_scan_kernel(float* X, float* Y, unsigned int arraySize, float* scan_value, int* flags, int* blockCounter) {
    extern __shared__ float temp[];  // Shared memory for storing per-block scan data

    __shared__ int bid_s; // Shared variable to store the dynamically assigned block index
    __shared__ float previous_sum;  // Shared memory for accumulated sum of previous blocks

    // Assign a unique block ID dynamically using atomic addition
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads(); // Ensure all threads see the correct block ID

    int bid = bid_s;  // Local copy of block index
    int thid = blockDim.x * bid + threadIdx.x;  // Global thread index
    int idx = threadIdx.x;  // Local thread index within the block
    int offset = 1;  // Offset variable used in tree-based operations

    int ai = 2 * idx;
    int bi = 2 * idx + 1;

    // Load input into shared memory with boundary checking to prevent out-of-bounds access
    if (2 * thid < arraySize) temp[ai] = X[2 * thid];
    else temp[ai] = 0.0f;

    if (2 * thid + 1 < arraySize) temp[bi] = X[2 * thid + 1];
    else temp[bi] = 0.0f;

    // **Up-Sweep (Reduce Phase)** - Builds a summation tree
    for (int d = SECTION_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            temp[bi] += temp[ai]; // Reduce sum to build a binary tree
        }
        offset *= 2;
    }

    // Wait for previous block's sum and propagate it
    if (threadIdx.x == 0) {
        while (atomicAdd(&flags[bid], 0) == 0) {} // Wait for previous flag to ensure sequential sum propagation
        previous_sum = scan_value[bid];  // Read the sum of all previous blocks
        scan_value[bid + 1] = previous_sum + temp[SECTION_SIZE - 1]; // Store cumulative sum for the next block
        __threadfence(); // Ensure memory consistency across blocks
        atomicAdd(&flags[bid + 1], 1); // Signal the next block that its previous sum is ready
    }
    __syncthreads();  // Ensure all threads see the correct previous_sum

    // **Down-Sweep Phase** - Converts the sum tree into an exclusive prefix sum
    if (idx == 0) temp[SECTION_SIZE - 1] = 0; // Root of the sum tree is set to 0
    for (int d = 1; d < SECTION_SIZE; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; // Convert to exclusive prefix sum
        }
    }
    __syncthreads();

    // Write Results Back to Global Memory
    if (2 * thid < arraySize) Y[2 * thid] = temp[ai] + previous_sum; // Apply previous block sum
    if (2 * thid + 1 < arraySize) Y[2 * thid + 1] = temp[bi] + previous_sum;
}

int main() {
    const int arraySize = N;
    float* x = (float*)malloc(arraySize * sizeof(float));
    float* y = (float*)malloc(arraySize * sizeof(float));


    // Initialize input array
    for (int i = 0; i < arraySize; i++) x[i] = i + 1;

    // Launch the CUDA kernel to perform the scan
    cudaError_t cudaStatus = launch_Blelloch_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Blelloch_scan_kernel failed!\n");
        return 1;
    }

    // Print computed prefix sum
    printf("Y = ");
    for (int i = 0; i < arraySize; i++) {
        printf("%0.2f ", y[i]);
    }
    printf("\n");

    cudaDeviceReset(); // Reset device before exiting
    return 0;
}

/*
    Host function that allocates memory, copies data, and launches the CUDA kernel.
    This function manages GPU memory allocation, data transfer, kernel execution,
    and retrieval of results back to the host.
*/
cudaError_t launch_Blelloch_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x, * dev_y, * dev_scan_value;
    int* dev_flags, * dev_blockCounter;
    int numThreads = SECTION_SIZE / 2;
    int numBlocks = (arraySize + SECTION_SIZE - 1) / SECTION_SIZE;

    // Allocate memory on the host for partial sums and synchronization flags
    float* h_scan_value = (float*)malloc((numBlocks + 1) * sizeof(float));
    int* h_flags = (int*)malloc((numBlocks + 1) * sizeof(int));
    int h_blockCounter = 0;

    // Initialize scan values and flags
    h_scan_value[0] = 0.0f;
    h_flags[0] = 1;
    for (int i = 1; i <= numBlocks; i++) h_flags[i] = 0;

    // Allocate memory on the GPU
    cudaMalloc(&dev_x, arraySize * sizeof(float));
    cudaMalloc(&dev_y, arraySize * sizeof(float));
    cudaMalloc(&dev_scan_value, (numBlocks + 1) * sizeof(float));
    cudaMalloc(&dev_flags, (numBlocks + 1) * sizeof(int));
    cudaMalloc(&dev_blockCounter, sizeof(int));

    // Copy input data and initialization values to the GPU
    cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_scan_value, h_scan_value, (numBlocks + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_flags, h_flags, (numBlocks + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_blockCounter, &h_blockCounter, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with dynamic shared memory
    Blelloch_scan_kernel << <numBlocks, numThreads, SECTION_SIZE * sizeof(float) >> > (
        dev_x, dev_y, arraySize, dev_scan_value, dev_flags, dev_blockCounter);
    cudaDeviceSynchronize();

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel launch
    Blelloch_scan_kernel << <numBlocks, numThreads, SECTION_SIZE * sizeof(float) >> > (
        dev_x, dev_y, arraySize, dev_scan_value, dev_flags, dev_blockCounter);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms. \n", milliseconds);

        // Bandwidth calculation
        long totalDataTransferred = arraySize * sizeof(float) * 2;
    float effectiveBandwidth = (totalDataTransferred / (milliseconds / 1000.0f)) / 1e9;
    printf("Effective bandwidth (GB/s): %.6f GB/s. \n\n", effectiveBandwidth);

        // Retrieve results from GPU memory
        cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated GPU memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_scan_value);
    cudaFree(dev_flags);
    cudaFree(dev_blockCounter);
    free(h_scan_value);
    free(h_flags);

    return cudaSuccess;
}
