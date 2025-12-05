# 🚀 CUDA Thrust & High-Performance Computing Examples

A hands-on collection of CUDA examples and exercises focused on performance optimization, parallel algorithms, and their application to fundamental Deep Learning components.

## 🌟 Project Goals

This repository serves as an **interactive learning environment** to master key parallel computing concepts:

1.  **Thrust Proficiency:** Use NVIDIA's Thrust library for highly-optimized parallel patterns (e.g., sort, reduce, transform).
2.  **Performance Tuning:** Understand and implement critical GPU optimization techniques like **Memory Coalescing** and **Fused Operations**. 
3.  **Application:** Apply CUDA to **Matrix Multiplication (GEMM)** and basic **Neural Network** architectures.

---

## 📂 Repository Structure

The project is divided into distinct learning paths:

| Folder | Focus | Description |
| :--- | :--- | :--- |
| `examples_exercises` | **Thrust/Optimization** | Core examples demonstrating performance tuning, iterators (`zip_iterator`), and high-level Thrust algorithms for general parallel problems (e.g., particle displacement). |
| `matmul` | **Linear Algebra/Low-Level CUDA** | Examples focused on **raw CUDA kernels** for efficient Matrix Multiplication (GEMM), a prerequisite for neural networks. |
| `neural_nets` | **Deep Learning Fundamentals** | Implementation of a basic feed-forward network utilizing the optimized operations from `matmul` and Thrust for element-wise tasks. |

---

## 📚 Key Exercises

| File/Area | Concept Learned | Primary Task |
| :--- | :--- | :--- |
| `optimized_max_displacement.cu` | **Fused Operations** | Analyze the memory access pattern of the zip iterator. |
| `performance_comparison.cu` | **Benchmarking** | Benchmark naive vs. optimized code across varying data sizes. |
| `matmul/` | **Tiled Kernels** | Implement and test a **tiled GEMM kernel** for cache reuse. |
| `neural_nets/` | **Element-wise Transforms** | Use `thrust::transform` to implement custom **ReLU/Sigmoid** activation functions. |

---

## ✅ Requirements

* **CUDA Toolkit 11.0 or higher**
* A CUDA-capable **NVIDIA GPU**
* A **C++14 compatible compiler** (e.g., `nvcc`)

## Acknowledgments 

Nvidia Developer blogs
