# CUDA Lab Series 1: Parallel Computing and CUDA


## Table of Contents
1. [Introduction](#introduction)
2. [Sequential vs Parallel Computing](#sequential-vs-parallel-computing)
    1. [Sequential Computing](#sequential-computing)
    2. [Parallel Computing](#parallel-computing)
3. [CPU vs GPU: Architectural Differences](#cpu-vs-gpu-architectural-differences)
    1. [CPU Architecture](#cpu-architecture)
    1. [GPU Architecture](#gpu-architecture)
4. [The Birth of CUDA](#the-birth-of-cuda)
5. [How Does CUDA Work?](#how-does-cuda-work?)
6. [Conclusion](#conclusion)


## Introduction

As a Machine Learning Engineer that relentlessly pursuites more computational power, I realized how fundamental it is to learn more about Parallel Computing.
And at the forefront of this paradigm shift stands NVIDIA's CUDA (Compute Unified Device Architecture), a revolutionary platform that has transformed how we approach high-performance computing.

---

## Sequential vs Parallel Computing

### Sequential Computing

Traditional computing follows a sequential model, executing instructions one after another. 
Imagine a single chef in a kitchen, methodically completing one task before moving to the next. This approach, while straightforward, has inherent limitations:
- Tasks must wait their turn
- Linear scaling of performance with processor speed
- Bottlenecked by the speed of a single processor core.

### Parallel Computing

Parallel computing divides a problem into smaller subproblems that can be solved concurrently. This involves using multiple processors or threads to perform computations simultaneously.
Imagine having multiple chefs in the kitchen, each handling different tasks simultaneously. This Model:
- Distributes workload across multiple processing units
- Executes multiple instructions simultaneously
- Scales performance with the number of processing units
- Reduces overall execution time for suitable problems

---

## CPU vs GPU: Architectural Differences

### CPU Architecture

Modern CPUs are designed as generalist processors, optimized for sequential processing with:
- Few cores (typically 4-16)
- Large cache memory
- Complex control logic
- Advanced branch prediction
- High clock speeds

These characteristics make CPUs excellent for tasks requiring complex decision-making and sequential operations.

### GPU Architecture

GPUs evolved from specialized graphics processors into general-purpose computing powerhouses with:

- Thousands of cores
- Simplified control logic
- High memory bandwidth
- Optimized for parallel operations
- Lower clock speeds but higher throughput

--- 

## The Birth of CUDA
As data volumes surged and computational requirements grew, traditional CPU-centric architectures struggled to keep pace. NVIDIA's CUDA was introduced in 2007 as a solution to empower developers to tap into the parallel processing prowess of GPUs.

CUDA revolutionized this by providing:
- Direct access to GPU's parallel computing elements
- A software layer that enabled general-purpose computing
- A familiar programming model based on C
- Tools for debugging and optimization

---

## How Does CUDA Work?
CUDA operates by launching kernels, which are functions executed in parallel on a grid of threads. Each thread is uniquely identified and executes a portion of the task.

Key Concepts:

- Threads and Blocks:
- A thread is the smallest unit of execution.
- Threads are grouped into blocks, and blocks form a grid.

Memory Hierarchy:
- Global Memory: Accessible by all threads but slow.
- Shared Memory: Shared among threads within a block, offering faster access.
- Local Memory: Private to each thread, stored in registers.

Execution Model:
- Threads are executed in warps (32 threads per warp).
- Warps are scheduled for execution by the device (GPU).

Example Workflow:
- Allocate memory on the device (GPU) using `cudaMalloc`.
- Transfer data from the host (CPU) to the device (GPU) using `cudaMemcpy`.
- Launch a kernel function with a specified grid and block size.
- Retrieve results from the device (GPU) to the host (CPU).
- Free allocated device (GPU) memory using `cudaFree`.

---

## Conclusion
CUDA represents more than just a programming platform; it's a fundamental shift in how we approach computational problems. As we push the boundaries of what's computationally possible, parallel computing through platforms like CUDA becomes not just an option, but a necessity.
The future of computing lies not in faster sequential processing, but in more efficient parallel computation. CUDA has played a crucial role in democratizing access to parallel computing resources, enabling breakthroughs across multiple fields, from scientific research to artificial intelligence.
Understanding CUDA and parallel computing principles is becoming increasingly essential for developers and researchers working on computationally intensive problems. As we move forward, the principles and practices established by CUDA will continue to influence how we design and implement high-performance computing solutions.
