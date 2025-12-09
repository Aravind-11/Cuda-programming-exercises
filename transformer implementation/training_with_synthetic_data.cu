
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <time.h>
#include <iomanip>

// --- HYPERPARAMETERS AND NETWORK CONFIGURATION ---
const int M = 128;   // Batch Size (Rows)
const int DIN = 64;  // Input Dimension
const int DHID = 32; // Hidden Dimension Size
const int C = 10;    // Number of Classes (Output Dimension)
const float LR = 0.01f; // Learning Rate

const int NUM_EPOCHS = 5;
const int BATCHES_PER_EPOCH = 20; // Simulated Batches per Epoch
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

// --- KERNELS (All kernels for Forward, Backward, and Update) ---

__global__ void initWeightsKernel(float *d_data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        unsigned int seed = i;
        float rand_val = (float)seed / (float)0xFFFFFFFF; 
        d_data[i] = (rand_val - 0.5f); 
    }
}

__global__ void reduceSumKernel(float *d_in, float *d_out, int size) {
    __shared__ float sdata[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? d_in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(d_out, sdata[0]); }
}

__global__ void simpleGemmBiasKernel(const float *X, const float *W, const float *B, float *Z, 
                                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += X[row * K + k] * W[k * N + col];
        }
        sum += B[col];
        Z[row * N + col] = sum;
    }
}

__global__ void reluKernel(float *A, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (A[i] < 0.0f) { A[i] = 0.0f; }
    }
}

__global__ void softmaxKernel(float *Z, int M, int C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = -1e20f;
        float sum_exp = 0.0f;
        int row_start = row * C;
        for (int j = 0; j < C; ++j) { max_val = fmaxf(max_val, Z[row_start + j]); }
        for (int j = 0; j < C; ++j) {
            Z[row_start + j] = expf(Z[row_start + j] - max_val);
            sum_exp += Z[row_start + j];
        }
        for (int j = 0; j < C; ++j) { Z[row_start + j] /= sum_exp; }
    }
}

__global__ void crossEntropyKernel(const float *Y_hat, const int *Y_true, float *d_loss_elements, 
                                   int M, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        int true_class = Y_true[i];
        float predicted_prob = Y_hat[i * C + true_class];
        float loss = -logf(fmaxf(predicted_prob, 1e-9f)); 
        d_loss_elements[i] = loss;
    }
}

__global__ void lossBackwardKernel(const float *Y_hat, const int *Y_true, float *d_dZ_out, 
                                   int M, int C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < C) {
        float gradient = Y_hat[row * C + col]; 
        if (col == Y_true[row]) { gradient -= 1.0f; }
        d_dZ_out[row * C + col] = gradient / (float)M; 
    }
}

__global__ void gemmBackwardInputKernel(const float *d_dZ, const float *W, float *d_dX, 
                                         int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += d_dZ[row * N + n] * W[col * N + n];
        }
        d_dX[row * K + col] = sum;
    }
}

__global__ void gemmBackwardWeightKernel(const float *X, const float *d_dZ, float *d_dW, 
                                         int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            sum += X[m * K + row] * d_dZ[m * N + col];
        }
        d_dW[row * N + col] = sum;
    }
}

__global__ void biasBackwardKernel(const float *d_dZ, float *d_dB, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            sum += d_dZ[m * N + col];
        }
        d_dB[col] = sum;
    }
}

__global__ void reluBackwardKernel(float *d_dZ, const float *Z, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (Z[i] <= 0.0f) { d_dZ[i] = 0.0f; }
    }
}

__global__ void updateWeightsKernel(float *W, const float *dW, int size, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        W[i] -= lr * dW[i];
    }
}

// --- HOST UTILITY FUNCTION (Generates clustered synthetic data) ---

/**
 * @brief Generates synthetic classification data where inputs are clustered around the class ID.
 */
void generate_synthetic_batch(std::vector<float>& h_X, std::vector<int>& h_Y_true, int M, int DIN, int C) {
    for (int i = 0; i < M; ++i) {
        // Assign a random class label
        int true_class = rand() % C;
        h_Y_true[i] = true_class;

        // Generate input vector X, centered around the class ID
        float center_value = (float)true_class / (float)C;
        for (int j = 0; j < DIN; ++j) {
            // Random noise between -0.1 and 0.1, centered at the class value
            float noise = (float)rand() / RAND_MAX * 0.2f - 0.1f;
            h_X[i * DIN + j] = center_value + noise;
        }
    }
}

// --- MAIN FUNCTION: Full Training Loop ---

int main() {
    std::cout << "--- Starting " << NUM_EPOCHS << "-Epoch Training on GPU ---" << std::endl;
    std::cout << "Configuration: Batch=" << M << ", D_in=" << DIN << ", D_hid=" << DHID << ", Classes=" << C << ", LR=" << LR << std::endl;
    std::cout << "Simulating " << BATCHES_PER_EPOCH << " batches per epoch using structured synthetic data." << std::endl;

    // Seed for host random data
    srand(time(NULL)); 

    // --- 1. Memory Setup and Allocation (Device) ---
    float *d_W1, *d_B1, *d_Z1_pre, *d_A1; 
    float *d_W2, *d_B2, *d_Z2_pre, *d_A2; 
    float *d_W3, *d_B3, *d_Z3; 
    float *d_X;
    int *d_Y_true;
    float *d_dW1, *d_dB1, *d_dA1_grad; 
    float *d_dW2, *d_dB2, *d_dA2_grad; 
    float *d_dW3, *d_dB3, *d_dZ3_grad;
    float *d_loss_elements, *d_final_loss_sum; 

    // Host Tensors (for transferring each batch)
    std::vector<int> h_Y_true(M);
    std::vector<float> h_X(M * DIN);
    float zero = 0.0f;
    int size_hid = M * DHID;

    // Allocate all device memory
    CUDA_CHECK(cudaMalloc((void**)&d_X, M * DIN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y_true, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_loss_elements, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_final_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_W1, DIN * DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B1, DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z1_pre, size_hid * sizeof(float))); 
    CUDA_CHECK(cudaMalloc((void**)&d_A1, size_hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dW1, DIN * DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dB1, DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dA1_grad, size_hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_W2, DHID * DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B2, DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z2_pre, size_hid * sizeof(float))); 
    CUDA_CHECK(cudaMalloc((void**)&d_A2, size_hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dW2, DHID * DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dB2, DHID * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dA2_grad, size_hid * sizeof(float))); 
    CUDA_CHECK(cudaMalloc((void**)&d_W3, DHID * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B3, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z3, M * C * sizeof(float))); 
    CUDA_CHECK(cudaMalloc((void**)&d_dW3, DHID * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dB3, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dZ3_grad, M * C * sizeof(float)));

    // Initialize Weights on GPU
    initWeightsKernel<<<DIV_UP(DIN * DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_W1, DIN * DHID);
    initWeightsKernel<<<DIV_UP(DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_B1, DHID);
    initWeightsKernel<<<DIV_UP(DHID * DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_W2, DHID * DHID);
    initWeightsKernel<<<DIV_UP(DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_B2, DHID);
    initWeightsKernel<<<DIV_UP(DHID * C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_W3, DHID * C);
    initWeightsKernel<<<DIV_UP(C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_B3, C);

    // --- 2. TRAINING LOOP ---
    dim3 block_mat(16, 16);
    dim3 grid_mat1(DIV_UP(DHID, 16), DIV_UP(M, 16));
    dim3 grid_mat3(DIV_UP(C, 16), DIV_UP(M, 16));
    dim3 grid_w3(DIV_UP(C, 16), DIV_UP(DHID, 16));
    dim3 grid_a2(DIV_UP(DHID, 16), DIV_UP(M, 16));
    dim3 grid_w2(DIV_UP(DHID, 16), DIV_UP(DHID, 16));
    dim3 grid_a1(DIV_UP(DHID, 16), DIV_UP(M, 16));
    dim3 grid_w1(DIV_UP(DHID, 16), DIV_UP(DIN, 16));

    float final_loss_sum_h;

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        float epoch_total_loss = 0.0f;

        for (int batch = 0; batch < BATCHES_PER_EPOCH; ++batch) {

            // --- A. LOAD BATCH (Simulated Data Loading) ---
            generate_synthetic_batch(h_X, h_Y_true, M, DIN, C);
            CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), M * DIN * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Y_true, h_Y_true.data(), M * sizeof(int), cudaMemcpyHostToDevice));

            // Reset loss accumulator
            CUDA_CHECK(cudaMemcpy(d_final_loss_sum, &zero, sizeof(float), cudaMemcpyHostToDevice));

            // --- B. FORWARD PASS ---

            // L1: X * W1 + B1 -> Z1_pre -> ReLU -> A1
            // A1 is also used as X for L2 and for the L2 backward pass
            simpleGemmBiasKernel<<<grid_mat1, block_mat>>>(d_X, d_W1, d_B1, d_Z1_pre, M, DIN, DHID);
            CUDA_CHECK(cudaMemcpy(d_A1, d_Z1_pre, size_hid * sizeof(float), cudaMemcpyDeviceToDevice));
            reluKernel<<<DIV_UP(size_hid, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_A1, size_hid);

            // L2: A1 * W2 + B2 -> Z2_pre -> ReLU -> A2
            simpleGemmBiasKernel<<<grid_mat1, block_mat>>>(d_A1, d_W2, d_B2, d_Z2_pre, M, DHID, DHID);
            CUDA_CHECK(cudaMemcpy(d_A2, d_Z2_pre, size_hid * sizeof(float), cudaMemcpyDeviceToDevice));
            reluKernel<<<DIV_UP(size_hid, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_A2, size_hid);

            // L3: A2 * W3 + B3 -> Z3 (logits) -> Softmax -> Y_hat (d_Z3)
            simpleGemmBiasKernel<<<grid_mat3, block_mat>>>(d_A2, d_W3, d_B3, d_Z3, M, DHID, C);
            softmaxKernel<<<DIV_UP(M, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_Z3, M, C); 

            // LOSS CALCULATION
            crossEntropyKernel<<<DIV_UP(M, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_Z3, d_Y_true, d_loss_elements, M, C);
            reduceSumKernel<<<DIV_UP(M, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_loss_elements, d_final_loss_sum, M);

            // --- C. BACKWARD PASS ---

            // L3 Backward (Softmax + Cross-Entropy)
            lossBackwardKernel<<<grid_mat3, block_mat>>>(d_Z3, d_Y_true, d_dZ3_grad, M, C);
            biasBackwardKernel<<<DIV_UP(C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_dZ3_grad, d_dB3, M, C);
            gemmBackwardWeightKernel<<<grid_w3, block_mat>>>(d_A2, d_dZ3_grad, d_dW3, M, DHID, C); 
            gemmBackwardInputKernel<<<grid_a2, block_mat>>>(d_dZ3_grad, d_W3, d_dA2_grad, M, DHID, C); 

            // L2 Backward (ReLU and Linear)
            reluBackwardKernel<<<DIV_UP(size_hid, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_dA2_grad, d_Z2_pre, size_hid); // d_dA2_grad becomes dL/dZ2
            biasBackwardKernel<<<DIV_UP(DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_dA2_grad, d_dB2, M, DHID);
            gemmBackwardWeightKernel<<<grid_w2, block_mat>>>(d_A1, d_dA2_grad, d_dW2, M, DHID, DHID); 
            gemmBackwardInputKernel<<<grid_a1, block_mat>>>(d_dA2_grad, d_W2, d_dA1_grad, M, DHID, DHID); // d_dA1_grad is dL/dA1

            // L1 Backward (ReLU and Linear)
            reluBackwardKernel<<<DIV_UP(size_hid, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_dA1_grad, d_Z1_pre, size_hid); // d_dA1_grad becomes dL/dZ1
            biasBackwardKernel<<<DIV_UP(DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_dA1_grad, d_dB1, M, DHID);
            gemmBackwardWeightKernel<<<grid_w1, block_mat>>>(d_X, d_dA1_grad, d_dW1, M, DIN, DHID); 

            // --- D. OPTIMIZER STEP (SGD) ---
            updateWeightsKernel<<<DIV_UP(DHID * C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_W3, d_dW3, DHID * C, LR);
            updateWeightsKernel<<<DIV_UP(C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_B3, d_dB3, C, LR);
            updateWeightsKernel<<<DIV_UP(DHID * DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_W2, d_dW2, DHID * DHID, LR);
            updateWeightsKernel<<<DIV_UP(DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_B2, d_dB2, DHID, LR);
            updateWeightsKernel<<<DIV_UP(DIN * DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_W1, d_dW1, DIN * DHID, LR);
            updateWeightsKernel<<<DIV_UP(DHID, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_B1, d_dB1, DHID, LR);

            // Retrieve loss and accumulate
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&final_loss_sum_h, d_final_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            epoch_total_loss += final_loss_sum_h;
        }

        // --- 3. EPOCH END REPORT ---
        float avg_loss = epoch_total_loss / (float)(BATCHES_PER_EPOCH * M);
        std::cout << "Epoch " << std::setw(2) << epoch + 1 << ": Average Loss = " << std::fixed << std::setprecision(6) << avg_loss << std::endl;
    }

    // --- 4. Cleanup ---
    std::cout << "\n--- Training Complete. Cleaning up memory. ---" << std::endl;
    cudaFree(d_X); cudaFree(d_Y_true); cudaFree(d_loss_elements); cudaFree(d_final_loss_sum);
    cudaFree(d_W1); cudaFree(d_B1); cudaFree(d_Z1_pre); cudaFree(d_A1);
    cudaFree(d_W2); cudaFree(d_B2); cudaFree(d_Z2_pre); cudaFree(d_A2);
    cudaFree(d_W3); cudaFree(d_B3); cudaFree(d_Z3); 
    cudaFree(d_dW1); cudaFree(d_dB1); cudaFree(d_dA1_grad); 
    cudaFree(d_dW2); cudaFree(d_dB2); cudaFree(d_dA2_grad); 
    cudaFree(d_dW3); cudaFree(d_dB3); cudaFree(d_dZ3_grad);

    return 0;
}
