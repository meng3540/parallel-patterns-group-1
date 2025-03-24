#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define SECTION_SIZE 32  /* Number of threads per block */

cudaError_t launch_Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N);

/* Optimized Kogge-Stone Scan Kernel with Warp Shuffle and Memory Coalescing */
__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, float* S, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;  /* Lane index within a warp */

    /* **Coalesced Memory Access: Load Input from Global to Shared Memory** */
    float val = (i < N) ? X[i] : 0.0f;  /* Each thread loads a contiguous element */

    /* **Perform Warp-Level Scan Using __shfl_up_sync()** */
    for (int stride = 1; stride < warpSize; stride *= 2) {
        float prev = __shfl_up_sync(0xFFFFFFFF, val, stride, warpSize);
        if (lane >= stride) {
            val += prev;
        }
    }

    /* **Store Results in Shared Memory for Inter-Block Computation** */
    XY[threadIdx.x] = val;
    

    /* **Block-Wide Kogge-Stone Scan for Large Arrays** */
    for (unsigned int stride = warpSize; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        XY[threadIdx.x] = temp;
        __syncthreads();
    }

    /* **Write Final Result to Global Memory** */
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }

    /* **Store Last Element of Each Block in S (for Hierarchical Scan)** */
    if (threadIdx.x == blockDim.x - 1) {
        S[blockIdx.x] = XY[threadIdx.x];
    }
}

/* Kernel for Scanning Partial Sums (Hierarchical Scan) */
__global__ void S_scan_kernel(float* S, unsigned int nBlocks) {
    unsigned int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float temp_out[];
    int lane2 = threadIdx.x % warpSize;  /* Lane index within a warp */

    /* **Coalesced Memory Access: Load Input from Global to Shared Memory** */
    float val2 = (i2 < nBlocks) ? S[i2] : 0.0f;
    /* Load data into shared memory with coalesced access */
    if (threadIdx.x < nBlocks) {
        temp_out[threadIdx.x] = S[threadIdx.x];
    }
    else {
        temp_out[threadIdx.x] = 0.0f;
    }

    for (int stride = 1; stride < warpSize; stride *= 2) {
        float prev2 = __shfl_up_sync(0xFFFFFFFF, val2, stride, warpSize);
        if (lane2 >= stride) {
            val2 += prev2;
        }
    }


    /* Perform parallel scan on the partial sums */
    for (unsigned int stride = warpSize; stride < blockDim.x; stride *= 2) {
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

    /* Store back to global memory */
    if (threadIdx.x < nBlocks) {
        S[threadIdx.x] = temp_out[threadIdx.x];
    }
}

/* Kernel for Adding Block-Wide Prefix Sums */
__global__ void addS_kernel(float* Y, float* S, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* **Each block adds the prefix sum of previous blocks** */
    if (blockIdx.x > 0 && i < N) {
        Y[i] += S[blockIdx.x - 1];
    }
}

int main() {
    const int arraySize = 32678;
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

    printf("Y = ");
    for (int i = 0; i < arraySize; i++) {
        printf("%0.2f ", y[i]);
    }

    cudaDeviceReset();
    return 0;
}

cudaError_t launch_Kogge_Stone_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x, * dev_y, * dev_S;

    int numBlocks = (arraySize + SECTION_SIZE - 1) / SECTION_SIZE;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMalloc((void**)&dev_x, arraySize * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_y, arraySize * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_S, numBlocks * sizeof(float));
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* **Launch Kogge-Stone Kernel** */
    Kogge_Stone_scan_kernel << < numBlocks, SECTION_SIZE >> > (dev_x, dev_y, dev_S, arraySize);
    cudaDeviceSynchronize();

    /* **Perform Hierarchical Scan if Needed** */
    if (numBlocks > 1) {
        S_scan_kernel << < 1, numBlocks, numBlocks * sizeof(float) >> > (dev_S, numBlocks);
        cudaDeviceSynchronize();

        addS_kernel << < numBlocks, SECTION_SIZE >> > (dev_y, dev_S, arraySize);
        cudaDeviceSynchronize();
    }

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
    return cudaStatus;
}
