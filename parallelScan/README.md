# Unoptimized Kogge-Stone Algorithm in CUDA

The Kogge-Stone algorithm is a parallel implementation of the prefix sum operation which takes advantage of reduction trees to take the computational complexity from O(n) in the sequential case to O(n*log(n)).

We implemented the algorithm in CUDA and the basic kernel is defined in the following code snippet:

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
The algorithm takes X as an input array, Y as an output array, and N as the number of elements in the input array.

To start, the algorithm assigns a shared variable *XY* so that all the threads can share the data from the input array. Each thread initializes the element of XY at its relative id (XY[threadIdx.x]) with the element from the input array X that is at its global position (X[i]). That is, each block will compute the partial sum of a SECTION_SIZE of the input array.

The part of the code that does the computation is:
```
   for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            XY[threadIdx.x] = temp;
    }
```
This loop iterates through the array XY, calculating the sum of all the preceding elements, in this case including itself (there are variations where it doesn't include itself). 
