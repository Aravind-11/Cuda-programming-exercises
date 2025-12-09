#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Global constants for the network size (needed for memory allocation and loop bounds)
const int M = 128;   // Batch Size
const int DIN = 64;  // Input Dimension
const int DHID = 32; // Hidden Dimension
const int C = 10;    // Number of Classes
const float LR = 0.01f; // Learning Rate

const int THREADS_PER_BLOCK = 256;

// Helper macros
#define CUDA_CHECK(call)                                                          \
{                                                                                 \
    cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",               \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call);         \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
}
#define DIV_UP(a, b) ((a + b - 1) / b)

// --- BACKWARD PASS KERNELS ---

/**
 * @brief Calculates the initial gradient dL/dZ2 (pre-softmax/logit).
 * dL/dZ2 = Y_hat - Y_true
 * * @param Y_hat Softmax output (M x C)
 * @param Y_true True labels (M x 1)
 * @param d_dZ2 Output gradient (M x C)
 */
__global__ void lossBackwardKernel(const float *Y_hat, const int *Y_true, float *d_dZ2, 
                                   int M, int C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < C) {
        float gradient = Y_hat[row * C + col]; // Start with P_i

        // If this column is the true class index, subtract 1
        if (col == Y_true[row]) {
            gradient -= 1.0f;
        }

        d_dZ2[row * C + col] = gradient / (float)M; // Normalize by Batch Size (M)
    }
}

/**
 * @brief Performs the backward pass for Matrix Multiplication: dL/dX = dL/dZ * W^T
 * Calculates the gradient passed back to the previous layer.
 */
__global__ void gemmBackwardInputKernel(const float *d_dZ, const float *W, float *d_dX, 
                                        int M, int K, int N) {
    // dL/dX is (M x K)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        // Dot product of dZ[row, :] and W^T[:, col]
        for (int n = 0; n < N; ++n) {
            // W is (K x N), W^T access is W[k*N + n] -> W[col*N + n]
            sum += d_dZ[row * N + n] * W[col * N + n];
        }
        d_dX[row * K + col] = sum;
    }
}

/**
 * @brief Performs the backward pass for Matrix Multiplication: dL/dW = X^T * dL/dZ
 * Calculates the gradient of the weights.
 */
__global__ void gemmBackwardWeightKernel(const float *X, const float *d_dZ, float *d_dW, 
                                         int M, int K, int N) {
    // dL/dW is (K x N)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // K rows
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N cols

    if (row < K && col < N) {
        float sum = 0.0f;
        // Dot product of X^T[row, :] and dZ[:, col]
        for (int m = 0; m < M; ++m) {
            // X[m * K + row] * d_dZ[m * N + col]
            sum += X[m * K + row] * d_dZ[m * N + col];
        }
        d_dW[row * N + col] = sum;
    }
}

/**
 * @brief Performs the backward pass for Bias: dL/dB = sum(dL/dZ, axis=0)
 */
__global__ void biasBackwardKernel(const float *d_dZ, float *d_dB, int M, int N) {
    // One thread per column (bias element)
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        float sum = 0.0f;
        // Sum down the column
        for (int m = 0; m < M; ++m) {
            sum += d_dZ[m * N + col];
        }
        d_dB[col] = sum;
    }
}

/**
 * @brief Performs element-wise ReLU backward: dL/dZ = dL/dA * ReLU'(Z)
 * ReLU'(Z) = 1 if Z > 0, 0 otherwise.
 */
__global__ void reluBackwardKernel(float *d_dZ, const float *Z, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Apply the mask: gradient is zeroed out where the pre-activation was <= 0
        if (Z[i] <= 0.0f) {
            d_dZ[i] = 0.0f;
        }
    }
}

/**
 * @brief Performs Stochastic Gradient Descent (SGD) update: W = W - LR * dW
 */
__global__ void updateWeightsKernel(float *W, const float *dW, int size, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        W[i] -= lr * dW[i];
    }
}

// --- MAIN FUNCTION (Simulating the Backward Pass Data Flow) ---

int main() {
    std::cout << "--- Backward Pass Kernel Demonstration ---" << std::endl;
    std::cout << "Simulating data flow for M=" << M << ", D_hid=" << DHID << ", C=" << C << std::endl;

    // --- 1. Memory Allocation and Dummy Data (Simulate Forward Pass Outputs) ---
    float *d_W2, *d_B2, *d_A1, *d_Z1_pre, *d_dZ2_grad;
    int *d_Y_true;

    // Layer 2 Tensors (Required for L2 Backward)
    CUDA_CHECK(cudaMalloc((void**)&d_W2, DHID * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B2, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dZ2_grad, M * C * sizeof(float))); // Input gradient

    // Layer 1 Tensors (Required for L1 Backward)
    CUDA_CHECK(cudaMalloc((void**)&d_A1, M * DHID * sizeof(float))); // Input to L2 GEMM
    CUDA_CHECK(cudaMalloc((void**)&d_Z1_pre, M * DHID * sizeof(float))); // L1 Pre-ReLU

    // Gradients to be calculated
    float *d_dW2, *d_dB2, *d_dA1_grad, *d_dW1, *d_dB1;
    CUDA_CHECK(cudaMalloc((void**)&d_dW2, DHID * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dB2, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dA1_grad, M * DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dW1, DIN * DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dB1, DHID * sizeof(float)));

    // Dummy Initialization (Fill with arbitrary values to simulate forward pass)
    // In a real scenario, these would be outputs of the forward pass
    std::vector<int> h_Y_true(M);
    std::vector<float> h_Y_hat(M * C);
    for (int i = 0; i < M * C; ++i) h_Y_hat[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < M; ++i) h_Y_true[i] = rand() % C;
    CUDA_CHECK(cudaMalloc((void**)&d_Y_true, M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_Y_true, h_Y_true.data(), M * sizeof(int), cudaMemcpyHostToDevice));
    // Y_hat is copied into d_dZ2_grad to simulate the first step
    CUDA_CHECK(cudaMemcpy(d_dZ2_grad, h_Y_hat.data(), M * C * sizeof(float), cudaMemcpyHostToDevice));

    // --- 2. BACKWARD PASS CHAIN EXECUTION (Simplified) ---

    // Grid/Block Setup
    dim3 grid_mat(DIV_UP(C, 16), DIV_UP(M, 16));
    dim3 block_mat(16, 16);
    int threads_bias = THREADS_PER_BLOCK;
    int blocks_bias_c = DIV_UP(C, THREADS_PER_BLOCK);
    int blocks_bias_hid = DIV_UP(DHID, THREADS_PER_BLOCK);

    std::cout << "\n[L2] 1. Initial Gradient (dL/dZ2 = Y_hat - Y_true)..." << std::endl;
    // d_dZ2_grad is updated in-place
    lossBackwardKernel<<<grid_mat, block_mat>>>(d_dZ2_grad, d_Y_true, d_dZ2_grad, M, C);

    std::cout << "[L2] 2. Bias Gradient (dL/dB2)..." << std::endl;
    biasBackwardKernel<<<blocks_bias_c, threads_bias>>>(d_dZ2_grad, d_dB2, M, C);

    std::cout << "[L2] 3. Weight Gradient (dL/dW2)..." << std::endl;
    dim3 grid_w2(DIV_UP(C, 16), DIV_UP(DHID, 16));
    gemmBackwardWeightKernel<<<grid_w2, block_mat>>>(d_A1, d_dZ2_grad, d_dW2, M, DHID, C);

    std::cout << "[L2] 4. Input Gradient (dL/dA1) to be passed to L1..." << std::endl;
    dim3 grid_a1(DIV_UP(DHID, 16), DIV_UP(M, 16));
    gemmBackwardInputKernel<<<grid_a1, block_mat>>>(d_dZ2_grad, d_W2, d_dA1_grad, M, DHID, C);

    // Continue to L1 (just showing the start of the chain)

    std::cout << "[L1] 5. ReLU Backward (dL/dZ1 = dL/dA1 * ReLU'(Z1_pre))..." << std::endl;
    int size1 = M * DHID;
    reluBackwardKernel<<<DIV_UP(size1, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_dA1_grad, d_Z1_pre, size1);

    std::cout << "\n[Optimizer] Update kernels (W = W - LR * dW) are ready." << std::endl;

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "\n--- Backward Pass Kernels executed successfully. ---" << std::endl;

    // Cleanup
    cudaFree(d_W2); cudaFree(d_B2); cudaFree(d_A1); cudaFree(d_Z1_pre); cudaFree(d_dZ2_grad);
    cudaFree(d_dW2); cudaFree(d_dB2); cudaFree(d_dA1_grad); cudaFree(d_dW1); cudaFree(d_dB1);
    cudaFree(d_Y_true);

    return 0;
}
