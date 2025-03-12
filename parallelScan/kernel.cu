#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/* Define shared memory section size */
#define SECTION_SIZE 32

cudaError_t launch_Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N);

/* CUDA kernel implementing the Kogge-Stone scan algorithm */
__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, unsigned int N) {
    /* Shared memory allocation for storing sums */
    __shared__ float XY[SECTION_SIZE];
    /* Compute global index for each thread */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load input data into shared memory */
    if (i < N) {
        XY[threadIdx.x] = X[i];
    }
    else {
        XY[threadIdx.x] = 0.0f; /* Set out-of-bounds threads to zero */
    }

    /* Perform Kogge-Stone parallel prefix sum */
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads(); /* Synchronize threads before reading */
        float temp;
        if (threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads(); /* Synchronize threads before writing */
        if (threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }

    /* Write results back to global memory */
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

int main() {
    const int arraySize = 5;
    float x[arraySize] = { 1, 2, 3, 4, 5 }; /* Input array */
    float y[arraySize] = { 0 }; /* Output array */

    cudaError_t cudaStatus = launch_Kogge_Stone_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kogge_Stone_scan_kernel failed!");
        return 1;
    }

    /* Print the computed prefix sum */
    printf("{1,2,3,4,5} => {%f, %f, %f, %f, %f}\n",
        y[0], y[1], y[2], y[3], y[4]);

    /* Reset the CUDA device */
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

/* Helper function to allocate memory and launch the kernel */
cudaError_t launch_Kogge_Stone_scan_kernel(float* x, float* y, unsigned int arraySize) {
    float* dev_x = 0; /* Device memory for input array */
    float* dev_y = 0; /* Device memory for output array */
    cudaError_t cudaStatus;

    /* Set the CUDA device */
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    /* Allocate device memory for input array */
    cudaStatus = cudaMalloc((void**)&dev_x, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    /* Allocate device memory for output array */
    cudaStatus = cudaMalloc((void**)&dev_y, arraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    /* CUDA events for timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Copy input data from host to device */
    cudaStatus = cudaMemcpy(dev_x, x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /* Record start time */
    cudaEventRecord(start);
 
    /* Launch the CUDA kernel */
    cudaDeviceSynchronize();
    Kogge_Stone_scan_kernel << <(arraySize + SECTION_SIZE - 1) / SECTION_SIZE, SECTION_SIZE >> > (dev_x, dev_y, arraySize);

    /* Record stop time */
    cudaEventRecord(stop);

    /* Check for kernel launch errors */
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

    /* Copy results from device to host */
    cudaStatus = cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /* Print the kernel execution time */
    printf("Kernel execution time: %.3f ms.\n", milliseconds);

    /* Calculate the total amount of data transferred in bytes */
    long totalDataTransferred = (arraySize * sizeof(float) * 2);

    /* Calculate the effective bandwidth in GB/s */
    float effectiveBandwidth = (totalDataTransferred / (milliseconds / 1000.0f)) / 1e9;

    /* Print the effective bandwidth in GB/s */
    printf("Effective bandwidth (GB/s): %.6f GB/s.\n", effectiveBandwidth);

Error:
    /* Free allocated device memory */
    cudaFree(dev_x);
    cudaFree(dev_y);
    return cudaStatus;
}
