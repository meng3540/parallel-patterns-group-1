#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SECTION_SIZE 128 /* Define the size of each block section */
#define N 4000
cudaError_t launch_Blelloch_scan_kernel(float* x, float* y, unsigned int arraySize);

__global__ void Blelloch_scan_kernel(
    float* X, float* Y, unsigned int N,
    float* scan_value, int* flags, int* blockCounter) {

    extern __shared__ float temp[];

    __shared__ int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();

    int bid = bid_s;
    int thid = blockDim.x * bid + threadIdx.x;
    int idx = threadIdx.x;
    int offset = 1;

    int ai = 2 * idx;
    int bi = 2 * idx + 1;

    if (ai < N) temp[ai] = X[2 * thid];
    else temp[ai] = 0.0f;

    if (bi < N) temp[bi] = X[2 * thid + 1];
    else temp[bi] = 0.0f;

    for (int d = N >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadIdx.x == 0) {
        scan_value[bid + 1] = temp[N - 1];
        __threadfence();
        atomicAdd(&flags[bid + 1], 1);
    }

    if (thid == 0) temp[N - 1] = 0;
    for (int d = 1; d < N; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    __shared__ float block_offset;
    if (threadIdx.x == 0) {
        while (flags[bid] == 0);
        block_offset = scan_value[bid];
    }
    __syncthreads();

    if (ai < N) Y[ai] = temp[ai] + block_offset;
    if (bi < N) Y[bi] = temp[bi] + block_offset;
}

int main() {
    const int arraySize = N;
    float x[arraySize], y[arraySize];
    for (int i = 0; i < arraySize; i++) x[i] = i + 1;

    cudaError_t cudaStatus = launch_Blelloch_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Blelloch_scan_kernel failed!\n");
        return 1;
    }

    printf("Y = ");
    for (int i = 0; i < arraySize; i++) {
        printf("%0.2f ", y[i]);
    }
    printf("\n");

    cudaDeviceReset();
    return 0;
}

cudaError_t launch_Blelloch_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x, * dev_y, * dev_scan_value;
    int* dev_flags, * dev_blockCounter;
    int numThreads = SECTION_SIZE/2;
    int numBlocks = (arraySize + SECTION_SIZE - 1) / SECTION_SIZE;

    float* h_scan_value = (float*)malloc((numBlocks + 1) * sizeof(float));
    int* h_flags = (int*)malloc((numBlocks + 1) * sizeof(int));
    int h_blockCounter = 0;

    h_scan_value[0] = 0.0f;
    h_flags[0] = 1;
    for (int i = 1; i <= numBlocks; i++) h_flags[i] = 0;

    cudaMalloc(&dev_x, arraySize * sizeof(float));
    cudaMalloc(&dev_y, arraySize * sizeof(float));
    cudaMalloc(&dev_scan_value, (numBlocks + 1) * sizeof(float));
    cudaMalloc(&dev_flags, (numBlocks + 1) * sizeof(int));
    cudaMalloc(&dev_blockCounter, sizeof(int));

    cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_scan_value, h_scan_value, (numBlocks + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_flags, h_flags, (numBlocks + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_blockCounter, &h_blockCounter, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    Blelloch_scan_kernel << <numBlocks, numThreads, SECTION_SIZE * sizeof(float) >> > (
        dev_x, dev_y, arraySize,
        dev_scan_value, dev_flags, dev_blockCounter);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms.\n", milliseconds);

    long totalDataTransferred = arraySize * sizeof(float) * 2;
    float effectiveBandwidth = (totalDataTransferred / (milliseconds / 1000.0f)) / 1e9;
    printf("Effective bandwidth (GB/s): %.6f GB/s.\n", effectiveBandwidth);

    cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_scan_value);
    cudaFree(dev_flags);
    cudaFree(dev_blockCounter);
    return cudaSuccess;
}
