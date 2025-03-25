#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SECTION_SIZE 32  // Number of threads per block

cudaError_t launch_Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N);

// Optimized Kogge-Stone Scan Kernel with Debugging
__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, float* S, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;  // Lane index within a warp

    // **Load Input from Global to Shared Memory**
    float val = (i < N) ? X[i] : 0.0f;  // Coalesced memory access

    // **Debugging: Print input values before processing**
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("First 5 input values: %f %f %f %f %f\n", X[0], X[1], X[2], X[3], X[4]);
    }

    // **Perform Warp-Level Scan Using __shfl_up_sync()**
    for (int stride = 1; stride < warpSize; stride *= 2) {
        float prev = __shfl_up_sync(0xFFFFFFFF, val, stride, warpSize);
        if (lane >= stride) {
            val += prev;
        }
    }

    // **Store Results in Shared Memory**
    XY[threadIdx.x] = val;
    __syncthreads();

    // **Block-Wide Kogge-Stone Scan**
    for (unsigned int stride = warpSize; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        XY[threadIdx.x] = temp;
        __syncthreads();
    }

    // **Write Final Result to Global Memory**
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }

    // **Store Last Element of Each Block in S**
    if (threadIdx.x == blockDim.x - 1) {
        S[blockIdx.x] = XY[threadIdx.x];
    }
}

// Kernel for Scanning Partial Sums (Hierarchical Scan)
__global__ void S_scan_kernel(float* S, unsigned int nBlocks) {
    extern __shared__ float temp_out[];

    // Load data into shared memory
    if (threadIdx.x < nBlocks) {
        temp_out[threadIdx.x] = S[threadIdx.x];
    } else {
        temp_out[threadIdx.x] = 0.0f;
    }

    // **Perform Parallel Scan**
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = 0;
        if (threadIdx.x >= stride) {
            temp = temp_out[threadIdx.x] + temp_out[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            temp_out[threadIdx.x] = temp;
        }
    }
    __syncthreads();

    // **Write Results Back to Global Memory**
    if (threadIdx.x < nBlocks) {
        S[threadIdx.x] = temp_out[threadIdx.x];
    }
}

// Kernel for Adding Block-Wide Prefix Sums
__global__ void addS_kernel(float* Y, float* S, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && i < N) {
        Y[i] += S[blockIdx.x - 1];
    }
}

// **Main Function**
int main() {
    const int arraySize = 64;
    float x[arraySize];
    float y[arraySize];

    for (int i = 0; i < arraySize; i++) {
        x[i] = i + 1;
    }

    cudaError_t cudaStatus = launch_Kogge_Stone_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kogge_Stone_scan_kernel failed!\n");
        return 1;
    }

    printf("\nY = ");
    for (int i = 0; i < arraySize; i++) {
        printf("%0.2f ", y[i]);
    }
    printf("\n");

    cudaDeviceReset();
    return 0;
}

// **CUDA Function to Launch Kernel**
cudaError_t launch_Kogge_Stone_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x = nullptr;
    float* dev_y = nullptr;
    float* dev_S = nullptr;

    int numBlocks = (arraySize + SECTION_SIZE - 1) / SECTION_SIZE;
    cudaError_t cudaStatus;

    // **Set CUDA Device**
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        goto Error;
    }

    // **Allocate Memory**
    cudaStatus = cudaMalloc((void**)&dev_x, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_y, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_S, numBlocks * sizeof(float));
    if (cudaStatus != cudaSuccess) goto Error;

    // **Copy Data from Host to Device**
    cudaStatus = cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // **Launch Kogge-Stone Kernel**
    Kogge_Stone_scan_kernel<<<numBlocks, SECTION_SIZE>>>(dev_x, dev_y, dev_S, arraySize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Kernel error: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaDeviceSynchronize();

    // **Perform Hierarchical Scan if Needed**
    if (numBlocks > 1) {
        S_scan_kernel<<<1, numBlocks, numBlocks * sizeof(float)>>>(dev_S, numBlocks);
        cudaDeviceSynchronize();

        addS_kernel<<<numBlocks, SECTION_SIZE>>>(dev_y, dev_S, arraySize);
        cudaDeviceSynchronize();
    }

    // **Copy Data Back to Host**
    cudaStatus = cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) goto Error;

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
    cudaFree(dev_S);
    return cudaStatus;
}
