# Unoptimized Kogge-Stone Algorithm in CUDA

The Kogge-Stone algorithm is a parallel implementation of the prefix sum operation which takes advantage of reduction trees to improve execution time with the trade-off of taking the computational complexity from O(n) in the sequential case to O(n*log(n)).

We implemented the algorithm in CUDA and the basic kernel is defined in the following code snippet:

```
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

To start, the algorithm assigns a shared variable ```XY``` which is an array with a length of ```SECTION_SIZE``` so that all the threads **within the block** can share the data from the input array. Additionally, the global thread id is calculated and stored in the variable ```i```. Then, each thread initializes ```XY[threadIdx.x]```&mdash;the element at the thread's relative position within the block&mdash;with ```X[i]```, which is the element at the threads global position. This has the effect of splitting the input array into chunks of size ```SECTION_SIZE``` for each block to compute a partial sum.

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
Putting the synchronization aspects of the loop aside for a moment, the main function of this loop is that in each iteration, the array element ```XY[threadIdx.x]``` is added with the array element ```XY[threadIdx.x - stride]```&mdash;the element ```stride``` steps away. The result then replaces the previous ```XY[threadIdx.x]```, repeating until each thread has computed the sum of itself and all the preceding elements.

Once the computation is done, the result is put into the output array Y at the index corresponding to the global id of the thread.

### Thread Synchronization

It's worth noting that since ```XY``` is in shared memory, each thread is changing the array simultaneously. That's where we come back to the synchronization aspect. The first ```__syncthreads();``` ensures that all the threads have finished writing the initial values into ```XY```, otherwise threads that are ahead may be working with the wrong data. Next, ```if(threadIdx.x >= stride)``` ensures that the program does not try to access memory that is out of bounds or a negative array index, which would throw an error. If that condition is met, the computation is stored in a temporary variable. The reason it must be stored in a temporary variable at first is to avoid modifying ```XY``` before all threads have finished their calculation. Othwerwise a race condition could occur in which one thread overwrites data that another thread is trying to read.

### Input size limit
In this novice implementation, the algorithm doesn't give you the final result for an arbitrary size input since each block calculates a partial sum of the overall input. This is limitation is partially controlled by section size, indeed you could increase the section size in order to allow for larger inputs, but that would only work for inputs that have less elements than the maximum threads per block and can fit in the shared memory of a single block. In order to resolve this issue, you can use one extra array ```S```, which would store the result of each block, and two more kernels. The first kernel would find the prefix sum of ```S``` and the second kernel would add each element of ```S``` to each block of ```Y```, effectively adding to each block the sum of the previous blocks.

