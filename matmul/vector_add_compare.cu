#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int N = 1 << 20;  // 1,048,576 elements

__global__ void vectorAdd(const int *A, const int *B, int *C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) C[i] = A[i] + B[i];
}

int main() {
    const int BYTES = N * sizeof(int);

    int *h_A = (int*)malloc(BYTES);
    int *h_B = (int*)malloc(BYTES);
    int *h_C = (int*)malloc(BYTES);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // ---------------- CPU TIMING ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU time: " << cpu_ms << " ms" << std::endl;

    // ---------------- GPU -----------------------
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, BYTES);
    cudaMalloc((void**)&d_B, BYTES);
    cudaMalloc((void**)&d_C, BYTES);

    auto gpu_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "GPU time (HtoD + kernel + DtoH): " << gpu_ms << " ms" << std::endl;
    std::cout << "Speedup (CPU/GPU): " << cpu_ms / gpu_ms << "x" << std::endl;

    // Validate last result
    if (h_C[N-1] == 3*(N-1)) std::cout << "Result OK\n";
    else std::cout << "Wrong result!\n";

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
