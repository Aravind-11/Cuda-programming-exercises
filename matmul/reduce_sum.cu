#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

const int N = 1 << 20;  // 1,048,576
const int THREADS = 256;

// ----------------------------------------------------------
// KERNEL 1: Reduce each block's chunk into one partial value
// ----------------------------------------------------------
__global__ void reduceSum(int *g_input, int *g_output, int size) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    // load to shared
    s_data[tid] = (i < size ? g_input[i] : 0);
    __syncthreads();

    // reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // write block result
    if (tid == 0) {
        g_output[blockIdx.x] = s_data[0];
    }
}

int main() {
    const int bytes = N * sizeof(int);

    // CPU memory
    int *h_A = (int*)malloc(bytes);
    int expected = 0;

    for (int i = 0; i < N; i++) {
        h_A[i] = 1;
        expected += 1;
    }

    std::cout << "Expected sum = " << expected << std::endl;

    // GPU memory
    int *d_A, *d_partial, *d_final;
    cudaMalloc(&d_A, bytes);

    int blocks_pass1 = (N + THREADS - 1) / THREADS;
    cudaMalloc(&d_partial, blocks_pass1 * sizeof(int));
    cudaMalloc(&d_final, sizeof(int));

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // PASS 1
    reduceSum<<<blocks_pass1, THREADS, THREADS * sizeof(int)>>>(d_A, d_partial, N);
    cudaDeviceSynchronize();

    // PASS 2 (second reduction: reduces block sums)
    int blocks_pass2 = (blocks_pass1 + THREADS - 1) / THREADS;
    reduceSum<<<blocks_pass2, THREADS, THREADS * sizeof(int)>>>(d_partial, d_final, blocks_pass1);
    cudaDeviceSynchronize();

    // Copy back
    int h_final = 0;
    cudaMemcpy(&h_final, d_final, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_final == expected)
        std::cout << "SUCCESS! Final sum = " << h_final << std::endl;
    else
        std::cout << "FAIL! Got " << h_final << " expected " << expected << std::endl;

    // cleanup
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_partial);
    cudaFree(d_final);
}
