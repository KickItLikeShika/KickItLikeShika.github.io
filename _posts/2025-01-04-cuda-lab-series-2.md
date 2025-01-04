# CUDA Lab Series 2 - Getting Started with CUDA: A Practical Guide


## Table of Contents
1. [Introduction](#introduction)

## Introduction
CUDA is an extension of C/C++ that enables parallel programming on NVIDIA GPUs. While it might seem daunting at first, CUDA builds upon familiar C concepts while adding parallel computing capabilities. This guide will walk you through the essential differences and practical basics.

---

## CUDA vs C: Key Differences
While CUDA is an extension of C, it introduces several key concepts and features that differentiate it from standard C programming:

1. Compiler Differences:
    - The main difference is the compiler (nvcc instead of gcc) and the file extension (`.cu` instead of `.c`). NVCC (NVIDIA CUDA Compiler) is actually a compiler driver that splits your code into two parts:
        - Host code (runs on CPU) → Compiled by regular C/C++ compiler
        - Device code (runs on GPU) → Compiled by NVIDIA compiler

2. Parallelism:
    - Standard C is designed for sequential or limited parallel execution using threads.
    - CUDA provides explicit support for massive parallelism by executing thousands of threads on a GPU.

---

## CUDA Execution Model

### Understanding Threads, Blocks, and Grids
CUDA's execution model is hierarchical, organized into three levels:

1. Threads (lowest level) 
2. Blocks (groups of threads)
3. Grid (collection of blocks)

#### Threads
- The basic unit of parallel execution in CUDA
- Each thread executes the same kernel function
- Threads have unique IDs within their block (threadIdx)
- Can access their ID using built-in variables:
    - threadIdx.x: Index in x dimension
    - threadIdx.y: Index in y dimension (if using 2D blocks)
    - threadIdx.z: Index in z dimension (if using 3D blocks)



#### Blocks
- Groups of threads that can cooperate
- Can be 1D, 2D, or 3D
- All blocks in a grid must have the same dimensions
- Threads within a block can:
    - Synchronize using __syncthreads()
    - Share memory
- Block dimensions accessed via blockDim.x, blockDim.y, blockDim.z
- Block index accessed via blockIdx.x, blockIdx.y, blockIdx.z
- Limited number of threads per block (typically 1024)

#### Grid
- Collection of thread blocks
- Can be 1D, 2D, or 3D
- Grid dimensions specified when launching kernel
- Grid dimensions accessed via gridDim.x, gridDim.y, gridDim.z


### Thread Organization and Indexing
Here's how threads are organized in a typical 1D example:
```
Grid
|
|---> Block 0 ---> [Thread 0][Thread 1][Thread 2]...[Thread 255]
|---> Block 1 ---> [Thread 0][Thread 1][Thread 2]...[Thread 255]
|---> Block 2 ---> [Thread 0][Thread 1][Thread 2]...[Thread 255]
...
```

To calculate the global thread index in 1D:
```cu
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D grids and blocks:
```cu
int global_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
int global_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
```

Launching Kernels: 
Kernel launch syntax uses the triple angle bracket notation:
```cu
// for 1D grid and block for now
myKernel<<<numBlocks, threadsPerBlock>>>(args);
```

### Choosing Block and Grid Dimensions
Several factors influence the choice of block and grid dimensions:
1. Hardware Limits:
    - Maximum threads per block (typically 1024)
    - Maximum dimensions of block (typically 1024×1024×64)
    - Available shared memory per block

2. Performance Considerations:
    - Warps (groups of 32 threads) are the actual execution units
    - Block size should be a multiple of warp size (32)
    - Common block sizes: 128, 256, 512 threads

3. Problem Size:
    - Need enough total threads to cover your data
    - Formula for 1D: gridSize = ceil(n / blockSize)

---

## Essential CUDA Keywords

### Function Qualifiers
CUDA introduces three main function qualifiers:
1. `__global__`:
    - Called from CPU, executes on GPU
    - Must return void
    - Launches a kernel (kernels are functions in CUDA)
    ```cu
        __global__ void addVectors(float* a, float* b, float* c, int n) {
            ...
        }
    ```

2. `__device__`:
    - Called from GPU, executes on GPU
    - Helper functions for your kernels
    ```cu
        __device__ float multiply(float a, float b) {
            ...
        }
    ```

3. `__host__`: 
    - Called from CPU, executes on CPU (default for regular functions)
    - Can be combined with device for functions that run on both
    ```cu
        __host__ __device__ float add(float a, float b) {
            ...
        }
    ```

### Memory Management
CUDA has its own memory management functions that parallel C's standard memory functions:

Let's assume we have an array of 4 elements on host
```cu
int n = 4;
float a[4] = {1, 2, 3, 4};
```

1. `cudaMalloc()`: Allocate memory 
    ```cu
    // CUDA allocation
    float* device_array;
    
    // The number of bytes to allocate, typically calculated using sizeof() * number_of_elements
    int size = n * sizeof(float)
    
    cudaMalloc((void **) &device_array, size);
    ```
    
    For the last line, CUDA requires a `void **` for the first argument of `cudaMalloc`, The cast `(void **)` tells the compiler to treat the address of `device_array` (a `float **`) as a `void **`. This casting is necessary because `cudaMalloc` is a generic function that works with all types of pointers. It expects a `void **` to accommodate any pointer type.

2. `cudaMemcpy()`: Copy memory
    ```cu
    // move vectors from cpu/host to gpu/device
    cudaMemcpy(device_array, a, size, cudaMemcpyHostToDevice);
    ```
    
    The code above means moving `device_array` from Host (CPU) to Device (GPU), where it exist in the place we have allocated in previous step.
    `cudaMemcpyHostToDevice` is a symbolic constant predefined in CUDA, and also there is `cudaMemcpyDeviceToHost`, to move data from Device to Host.

3. `cudaFree()`: Free device memory
    ```cu
    cudaFree(device_array);
    ```

---

## Practical Example: Vector Addition

Let's put it all together with a complete example for adding 2 vectors:

```cu
#include <stdio.h>

__global__
void vecAddKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // initiated grid will have blocks of same thread size, but threads in last block might not be used as vector size might be smaller,
    // so that's why we have this if conidtion
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void addVectors(float* a_h, float* b_h, float* c_h, int n) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;

    // allocate memory on gpu/device for the new vectors
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);

    // move vectors from cpu/host to gpu/device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // launch the grid, ceil(n/256) blocks of 256 threads each
    // and execute on device
    vecAddKernel<<<ceil(n/256.0), 256>>>(a_d, b_d, c_d, n);

    // move vector from cpu gpu to cpu
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // free gpu/device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main() {

    int n = 4;
    float a[4] = {1, 2, 3, 4};
    float b[4] = {1, 2, 3, 4};
    float c[4];

    addVectors(a, b, c, n);
    for (int i = 0; i < n; i++) {
        printf("%f\n", c[i]);
    }

    return 0;
}
```

Let's break down our vector addition example to understand each component:

1. 
    ```cu
    __global__
    void vecAddKernel(float* a, float* b, float* c, int n) {
        // Calculate global thread index
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if this thread should process an element
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
    ```
    - Uses __global__ to indicate it's a CUDA Kernel that runs on the GPU
    - Calculates unique index for each thread
    - Includes bounds check for last block

2. 
    ```cu
        void addVectors(float* a_h, float* b_h, float* c_h, int n) {
            int size = n * sizeof(float);
            float *a_d, *b_d, *c_d;

            // Allocate GPU memory
            cudaMalloc((void **) &a_d, size);
            cudaMalloc((void **) &b_d, size);
            cudaMalloc((void **) &c_d, size);

            // Copy input data to GPU
            cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
            cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

            // Calculate grid dimensions
            int threadsPerBlock = 256;
            int blocksPerGrid = ceil(n / (float)threadsPerBlock);

            // Launch kernel
            vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, n);

            // Copy result back to CPU
            cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

            // Free GPU memory
            cudaFree(a_d);
            cudaFree(b_d);
            cudaFree(c_d);
        }
    ```
    - This is a host function
    - Allocates memory on GPU using cudaMalloc
    - Copies input data using cudaMemcpy
    - Calculates appropriate grid dimensions
    - Launches kernel with <<<>>> syntax
    - Retrieves results and cleans up

3. 
    ```cu
        int main() {
            int n = 4;
            float a[4] = {1, 2, 3, 4};
            float b[4] = {1, 2, 3, 4};
            float c[4];

            addVectors(a, b, c, n);
            
            // Print results
            for (int i = 0; i < n; i++) {
                printf("%f\n", c[i]);
            }

            return 0;
        }
    ```
    - Main function to run the program

To compile and run:
```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

---

## Conclusion
By understanding CUDA's thread hierarchy, memory management, and kernel launches, you can start leveraging the power of parallel computing. This example lays the groundwork for exploring more advanced concepts and optimizations in CUDA programming.
