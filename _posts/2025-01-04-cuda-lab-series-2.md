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

## Complete Example: Vector Addition
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

To compile and run:
```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

