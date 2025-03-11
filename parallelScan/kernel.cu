#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Define shared memory section size as a constant
#define SECTION_SIZE 32

cudaError_t launch_Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N);

__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        XY[threadIdx.x] = X[i];
    }
    else {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }

    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

int main() {
    const int arraySize = 5;
    float x[arraySize] = { 1, 2, 3, 4, 5 };
    float y[arraySize] = { 0 };

    // Launch CUDA kernel
    cudaError_t cudaStatus = launch_Kogge_Stone_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kogge_Stone_scan_kernel failed!");
        return 1;
    }

    printf("{1,2,3,4,5} => {%f, %f, %f, %f, %f}\n",
        y[0], y[1], y[2], y[3], y[4]);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function to allocate memory and launch the kernel
cudaError_t launch_Kogge_Stone_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x = 0;
    float* dev_y = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_y, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch CUDA kernel
    Kogge_Stone_scan_kernel << <(arraySize + SECTION_SIZE - 1) / SECTION_SIZE, SECTION_SIZE >> > (dev_x, dev_y, arraySize);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_x);
    cudaFree(dev_y);
    return cudaStatus;
}
