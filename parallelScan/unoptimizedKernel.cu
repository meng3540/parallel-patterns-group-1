#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/* Define size of chunks that input array will be split into*/
#define SECTION_SIZE 32
cudaError_t launch_Kogge_Stone_scan_kernel(float* x, float* y, unsigned int arraySize);

/* CUDA kernel implementing the Kogge-Stone scan algorithm */
__global__ void Kogge_Stone_scan_kernel(float* X, float* Y, float* S, unsigned int N) {
    /* Shared memory allocation for storing sums */
    __shared__ float XY[SECTION_SIZE];
    
    /* Compute global index for each thread */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load input data into shared memory */
    if (i < N) {
		XY[threadIdx.x] = X[i]; /* Assign input data to shared memory */
    }
    else {
        XY[threadIdx.x] = 0.0f; /* Set out-of-bounds threads to zero */
    }

    /* Perform Kogge-Stone parallel prefix sum */
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads(); /* Synchronize threads before reading */
        float temp;
        if (threadIdx.x >= stride)
            /* Store the sum in a temporary variable to avoid overwriting XY while other threads are using it*/
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride]; 
        __syncthreads(); /* Synchronize threads before writing */
        if (threadIdx.x >= stride)
            /* Write the sum back to shared memory */
			XY[threadIdx.x] = temp; 
    }
    __syncthreads();
    /* Write results back to global memory */
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
	/* Store the sum of each section in S */
    if (threadIdx.x == blockDim.x - 1) {
        S[blockIdx.x] = XY[threadIdx.x];
    }
}

/* CUDA kernel to perform a parallel prefix sum on the partial sums in S */
__global__ void S_scan_kernel(float* S, unsigned int nBlocks) {
    /* Shared memory allocation for storing sums */
    extern __shared__ float temp_out[];

    /* Load input data into shared memory */
    if (threadIdx.x < nBlocks) {
        temp_out[threadIdx.x] = S[threadIdx.x]; /* Assign input data to shared memory */
    }
    else {
        temp_out[threadIdx.x] = 0.0f; /* Set out-of-bounds threads to zero */
    }

    /* Perform parallel prefix sum */
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads(); /* Synchronize threads before reading */
        float temp;
        if (threadIdx.x >= stride)
            /* Store the sum in a temporary variable to avoid overwriting S while other threads are accessing it*/
            temp = temp_out[threadIdx.x] + temp_out[threadIdx.x - stride];
        __syncthreads(); /* Synchronize threads before writing */
        if (threadIdx.x >= stride)
            /* Write the sum back to shared memory */
            temp_out[threadIdx.x] = temp;
    }
    __syncthreads();
    /* Write results back to global memory */
    if (threadIdx.x < nBlocks) {
        S[threadIdx.x] = temp_out[threadIdx.x];
    }
}
/* CUDA kernel to add the sum of each section to the one ahead of it */
__global__ void addS_kernel(float* Y, float* S, unsigned int N) {
	/* Compute global index for each thread */
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	/* Add the sum of the previous section to the current section */
	if (blockIdx.x > 0 && i < N) {
		Y[i] += S[blockIdx.x - 1];
	}
}

/* The code below is derived from the default kernel code given in Visual Studio*/

int main() {
    /* Declare/initialize input and output arrays */
	const int arraySize = 1000;
	float x[arraySize];
	float y[arraySize];
	for (int i = 0; i < arraySize; i++) {
		x[i] = i + 1;
	}

	/* Execute helper function which launches all 3 kernels */
    cudaError_t cudaStatus = launch_Kogge_Stone_scan_kernel(x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kogge_Stone_scan_kernel failed!");
        return 1;
    }
    
    /* Print the computed prefix sum */
    printf("Y = ");
    for (int i = 0;i<arraySize;i++){
		printf("%0.2f ", y[i]);
    }

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
    float* dev_x; /* Device memory for input array */
    float* dev_y; /* Device memory for output array */
	float* dev_S; /* Device memory for storing sums */

    /* Calculate number of blocks needed for array */
	int numBlocks = (arraySize + SECTION_SIZE - 1) / SECTION_SIZE;

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
    
	/* Allocate device memory for partial sums */
    cudaStatus = cudaMalloc((void**)&dev_S, numBlocks * sizeof(float));
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
    Kogge_Stone_scan_kernel << <numBlocks, SECTION_SIZE >> > (dev_x, dev_y, dev_S, arraySize);

    /* Check for kernel launch errors */
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	/* Synchronize the device */
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
        goto Error;
    }

	/* Perform a parallel prefix sum on the partial sums in S if multiple blocks were used*/
    if (numBlocks > 1) {
        
        /* Launch prefix sum for S array with a block size of 1, with one thread for each block, and passing the size of the temp variable */
        S_scan_kernel << <1, numBlocks, numBlocks * sizeof(float) >> > (dev_S, numBlocks);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) goto Error;
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) goto Error;

		/* Add the sum of each section to the one ahead of it */
        addS_kernel << <numBlocks, SECTION_SIZE >> > (dev_y, dev_S, arraySize);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) 
            goto Error;
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) 
            goto Error;

    }
    /* Copy results from device to host */
    cudaStatus = cudaMemcpy(y, dev_y, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /* Record stop time */
    cudaEventRecord(stop);

    /* Wait for the stop event to complete */
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
