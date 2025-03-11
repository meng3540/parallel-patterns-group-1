# Unoptimized Kogge-Stone Algorithm in CUDA

The Kogge-Stone algorithm is a parallel implementation of the prefix sum operation which takes advantage of reduction trees to take the computational complexity from O(n) in the sequential case to O(n*log(n)).

We implemented the algorithm in CUDA and the kernel is defined in the following code snippet:

```c=
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE]; 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }

    if(i < N) {
        Y[i] = XY[threadIdx.x];
    }
}
```
