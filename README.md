# CUDA Programming for Dummies --- in ML perspective!

CUDA examples and exercises focused on performance optimization, parallel algorithms, and their application to fundamental Deep Learning components.

## Project Goals

This repository serves as an **interactive learning environment** to master key parallel computing concepts:

1. CUDA concepts: High level CUDA concepts including **threads, synchronisation, shared memory and tiling**.
2.  **Thrust Proficiency:** Use NVIDIA's Thrust library for highly-optimized parallel patterns (e.g., sort, reduce, transform).
3.  **Application:** Apply CUDA to **Matrix Multiplication (GEMM)** and basic **Neural Network** architectures.

---


## Key Exercises

| File/Area | Concept Learned | Primary Task |
| :--- | :--- | :--- |
| `optimized_max_displacement.cu` | **Fused Operations** | Analyze the memory access pattern of the zip iterator. |
| `performance_comparison.cu` | **Benchmarking** | Benchmark naive vs. optimized code across varying data sizes. |
| `matmul/` | **Tiled Kernels** | Implement and test a **tiled GEMM kernel** for cache reuse. |
| `neural_nets/` | **Element-wise Transforms** | Use `thrust::transform` to implement custom **ReLU/Sigmoid** activation functions. |

---

## Requirements

* **CUDA Toolkit 11.0 or higher**
* A CUDA-capable **NVIDIA GPU**
* A **C++14 compatible compiler** (e.g., `nvcc`)

## Acknowledgments 

[An even easier introduction to CUDA
](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

[CUDA programming guide
](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)

[Fundamentals of Accelerated Computing with Modern CUDA C++
](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-AC-01+V2)
