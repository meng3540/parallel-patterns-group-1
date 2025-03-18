#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SECTION_SIZE 32  // Number of threads per block

cudaError_t launch_Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N);

// Optimized Kogge-Stone Scan Kernel Using Warp Shuffles
__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;  // Lane index within a warp

    // Load input from global memory
    float val = (i < N) ? X[i] : 0.0f;

    // Perform warp-level scan using __shfl_up_sync
    for (int stride = 1; stride < warpSize; stride *= 2) {
        float prev = __shfl_up_sync(0xFFFFFFFF, val, stride, warpSize);
        if (lane >= stride) {
            val += prev;
        }
    }

    // Write results back to global memory efficiently
    if (i < N) {
        Y[i] = val;
    }
}

int main() {
    const int arraySize = 5;
    float x[arraySize] = { 1, 2, 3, 4, 5 };
    float y[arraySize];

    cudaError_t cudaStatus = launch_Kogge_Stone_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kogge_Stone_scan_kernel failed!\n");
        return 1;
    }

    printf("{1,2,3,4,5} => {%f, %f, %f, %f, %f}\n", y[0], y[1], y[2], y[3], y[4]);
    
    cudaDeviceReset();
    return 0;
}

cudaError_t launch_Kogge_Stone_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x = nullptr;
    float* dev_y = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x, arraySize * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_y, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    // **Memory transfer: Copy data from host to device**
    cudaStatus = cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = SECTION_SIZE;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    
    Kogge_Stone_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_x, dev_y, arraySize);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaDeviceSynchronize();

    // **Coalesced memory transfer: Copy results back from device to host**
    cudaStatus = cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms.\n", milliseconds);

    long totalDataTransferred = (arraySize * sizeof(float) * 2);
    float effectiveBandwidth = (totalDataTransferred / (milliseconds / 1000.0f)) / 1e9;
    printf("Effective bandwidth (GB/s): %.6f GB/s.\n", effectiveBandwidth);

Error:
    cudaFree(dev_x);
    cudaFree(dev_y);
    return cudaStatus;
}
