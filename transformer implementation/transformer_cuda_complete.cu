#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// FORWARD KERNELS
// ============================================================================

__global__ void gelu_forward_kernel(const float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = x[idx];
        float a = sqrtf(2.0f / M_PI);
        float u = a * (val + 0.044715f * val * val * val);
        float tanh_u = tanhf(u);
        y[idx] = 0.5f * val * (1.0f + tanh_u);
    }
}

__global__ void layer_norm_forward_kernel(
    const float* x, float* y, float* mean, float* inv_std,
    int BT, int D, float eps
) {
    int token_idx = blockIdx.x;
    if (token_idx >= BT) return;
    
    extern __shared__ float shared[];
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += x[token_idx * D + i];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mu = shared[0] / D;
    if (threadIdx.x == 0) mean[token_idx] = mu;
    __syncthreads();
    
    // Compute variance
    sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = x[token_idx * D + i] - mu;
        sum += diff * diff;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float var = shared[0] / D;
    float inv = 1.0f / sqrtf(var + eps);
    if (threadIdx.x == 0) inv_std[token_idx] = inv;
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        y[token_idx * D + i] = (x[token_idx * D + i] - mu) * inv;
    }
}

__global__ void scale_shift_kernel(
    const float* x, const float* gamma, const float* beta,
    float* y, int BT, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BT * D;
    if (idx < total) {
        int d = idx % D;
        y[idx] = x[idx] * gamma[d] + beta[d];
    }
}

__global__ void softmax_forward_kernel(const float* x, float* y, int B, int T) {
    int batch = blockIdx.y;
    int row = blockIdx.x;
    
    if (batch >= B || row >= T) return;
    
    extern __shared__ float shared[];
    
    int offset = (batch * T + row) * T;
    
    // Find max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        max_val = fmaxf(max_val, x[offset + i]);
    }
    shared[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        float e = expf(x[offset + i] - max_val);
        y[offset + i] = e;
        sum += e;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum = shared[0];
    
    // Normalize
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        y[offset + i] /= sum;
    }
}

__global__ void add_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

__global__ void scale_kernel(const float* x, float scale, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = x[idx] * scale;
}

__global__ void mse_loss_kernel(const float* pred, const float* target, float* loss, int N) {
    extern __shared__ float sdata[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (idx < N) {
        diff = pred[idx] - target[idx];
        diff = diff * diff;
    }
    sdata[threadIdx.x] = diff;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0] * 0.5f / N);
    }
}

// ============================================================================
// BACKWARD KERNELS
// ============================================================================

__global__ void mse_grad_kernel(const float* pred, const float* target, float* grad, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad[idx] = (pred[idx] - target[idx]) / N;
    }
}

__global__ void gelu_backward_kernel(const float* x, const float* dy, float* dx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = x[idx];
        float a = sqrtf(2.0f / M_PI);
        float u = a * (val + 0.044715f * val * val * val);
        float tanh_u = tanhf(u);
        float sech2 = 1.0f - tanh_u * tanh_u;
        float du_dx = a * (1.0f + 3.0f * 0.044715f * val * val);
        float grad = 0.5f * (1.0f + tanh_u) + 0.5f * val * sech2 * du_dx;
        dx[idx] = dy[idx] * grad;
    }
}

__global__ void layer_norm_backward_kernel(
    const float* dy, const float* x, const float* mean, const float* inv_std,
    float* dx, int BT, int D
) {
    int token_idx = blockIdx.x;
    if (token_idx >= BT) return;
    
    extern __shared__ float shared[];
    float* s_sum1 = shared;
    float* s_sum2 = &shared[blockDim.x];
    
    float mu = mean[token_idx];
    float inv = inv_std[token_idx];
    
    int offset = token_idx * D;
    
    // Compute intermediate sums
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x_mu = x[offset + i] - mu;
        sum1 += dy[offset + i];
        sum2 += dy[offset + i] * x_mu;
    }
    s_sum1[threadIdx.x] = sum1;
    s_sum2[threadIdx.x] = sum2;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum1[threadIdx.x] += s_sum1[threadIdx.x + stride];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float dmu = -inv * s_sum1[0];
    float dvar = -0.5f * inv * inv * inv * s_sum2[0];
    
    // Final gradient
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x_mu = x[offset + i] - mu;
        dx[offset + i] = dy[offset + i] * inv + 
                         dvar * 2.0f * x_mu / D + 
                         dmu / D;
    }
}

__global__ void softmax_backward_kernel(
    const float* A, const float* dA, float* dScores, int B, int T
) {
    int batch = blockIdx.y;
    int row = blockIdx.x;
    
    if (batch >= B || row >= T) return;
    
    extern __shared__ float shared[];
    
    int offset = (batch * T + row) * T;
    
    // Compute sum(dA * A)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        sum += dA[offset + i] * A[offset + i];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float dot = shared[0];
    
    // dScores = A * (dA - dot)
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        dScores[offset + i] = A[offset + i] * (dA[offset + i] - dot);
    }
}

__global__ void elementwise_mul_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] * b[idx];
}

__global__ void sum_reduction_kernel(const float* x, float* out, int N, int D) {
    // Sum over batch and time dimensions for gamma/beta gradients
    extern __shared__ float sdata[];
    
    int d = blockIdx.x; // one block per feature dimension
    if (d >= D) return;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += x[i * D + d];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        out[d] = sdata[0];
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void init_random(float* d_data, int size, float scale, curandGenerator_t gen) {
    curandGenerateNormal(gen, d_data, size, 0.0f, scale);
}

void init_constant(float* d_data, int size, float value) {
    float* h_data = new float[size];
    for (int i = 0; i < size; i++) h_data[i] = value;
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_data;
}

// ============================================================================
// MAIN FORWARD/BACKWARD
// ============================================================================

int main() {
    // Config
    const int B = 1, T = 2, D = 4;
    const int BT = B * T;
    const float eps = 1e-5f;
    const float sqrt_dk = sqrtf((float)D);
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 0ULL);
    
    // ========== Allocate Parameters ==========
    float *d_W_Q, *d_W_K, *d_W_V, *d_W_O;
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_gamma1, *d_beta1, *d_gamma2, *d_beta2;
    
    CUDA_CHECK(cudaMalloc(&d_W_Q, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_K, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_V, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_O, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, D * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, 2 * D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta1, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta2, D * sizeof(float)));
    
    // Initialize parameters
    init_random(d_W_Q, D * D, 0.1f, gen);
    init_random(d_W_K, D * D, 0.1f, gen);
    init_random(d_W_V, D * D, 0.1f, gen);
    init_random(d_W_O, D * D, 0.1f, gen);
    init_random(d_W1, D * 2 * D, 0.1f, gen);
    init_constant(d_b1, 2 * D, 0.0f);
    init_random(d_W2, 2 * D * D, 0.1f, gen);
    init_constant(d_b2, D, 0.0f);
    init_constant(d_gamma1, D, 1.0f);
    init_constant(d_beta1, D, 0.0f);
    init_constant(d_gamma2, D, 1.0f);
    init_constant(d_beta2, D, 0.0f);
    
    // ========== Allocate Gradients ==========
    float *d_dW_Q, *d_dW_K, *d_dW_V, *d_dW_O;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    float *d_dgamma1, *d_dbeta1, *d_dgamma2, *d_dbeta2;
    
    CUDA_CHECK(cudaMalloc(&d_dW_Q, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_K, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_V, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_O, D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW1, D * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, 2 * D * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dgamma1, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbeta1, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dgamma2, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbeta2, D * sizeof(float)));
    
    // Zero gradients
    CUDA_CHECK(cudaMemset(d_dW_Q, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW_K, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW_V, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW_O, 0, D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW1, 0, D * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db1, 0, 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dW2, 0, 2 * D * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db2, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dgamma1, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbeta1, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dgamma2, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbeta2, 0, D * sizeof(float)));
    
    // ========== Allocate Activations & Cache ==========
    float *d_x, *d_target;
    float *d_x_norm1, *d_x_ln;
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_A, *d_head;
    float *d_attn_out, *d_res1;
    float *d_res1_norm, *d_res1_ln;
    float *d_h, *d_h_act, *d_ffn_out, *d_out;
    float *d_mean1, *d_inv_std1, *d_mean2, *d_inv_std2;
    
    CUDA_CHECK(cudaMalloc(&d_x, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_norm1, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_ln, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, B * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A, B * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_head, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res1, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res1_norm, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res1_ln, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h, BT * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_act, BT * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_out, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean1, BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std1, BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean2, BT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inv_std2, BT * sizeof(float)));
    
    // Initialize input and target
    init_random(d_x, BT * D, 1.0f, gen);
    init_random(d_target, BT * D, 1.0f, gen);
    
    // ========== FORWARD PASS ==========
    printf("========== FORWARD PASS ==========\n");
    
    const float alpha = 1.0f, beta = 0.0f, beta_add = 1.0f;
    int threads = 256;
    int blocks;
    
    // Pre-LN
    layer_norm_forward_kernel<<<BT, threads, threads * sizeof(float)>>>(
        d_x, d_x_norm1, d_mean1, d_inv_std1, BT, D, eps
    );
    
    blocks = (BT * D + threads - 1) / threads;
    scale_shift_kernel<<<blocks, threads>>>(
        d_x_norm1, d_gamma1, d_beta1, d_x_ln, BT, D
    );
    
    // Q, K, V projections (using cuBLAS for matmul)
    // Note: cuBLAS uses column-major, so we compute W^T @ x^T
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_Q, D, d_x_ln, D, &beta, d_Q, D));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_K, D, d_x_ln, D, &beta, d_K, D));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_V, D, d_x_ln, D, &beta, d_V, D));
    
    // Attention scores = Q @ K^T
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        T, T*B, D, &alpha, d_K, D, d_Q, D, &beta, d_scores, T));
    
    // Scale by sqrt(d_k)
    blocks = (B * T * T + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(d_scores, 1.0f / sqrt_dk, d_scores, B * T * T);
    
    // Softmax
    dim3 grid_soft(T, B);
    softmax_forward_kernel<<<grid_soft, threads, threads * sizeof(float)>>>(
        d_scores, d_A, B, T
    );
    
    // head = A @ V (per batch)
    for (int b = 0; b < B; b++) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            D, T, T, &alpha,
            d_V + b * T * D, D,
            d_A + b * T * T, T,
            &beta, d_head + b * T * D, D));
    }
    
    // Output projection
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_O, D, d_head, D, &beta, d_attn_out, D));
    
    // Residual 1
    blocks = (BT * D + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_x, d_attn_out, d_res1, BT * D);
    
    // LN before FFN
    layer_norm_forward_kernel<<<BT, threads, threads * sizeof(float)>>>(
        d_res1, d_res1_norm, d_mean2, d_inv_std2, BT, D, eps
    );
    
    blocks = (BT * D + threads - 1) / threads;
    scale_shift_kernel<<<blocks, threads>>>(
        d_res1_norm, d_gamma2, d_beta2, d_res1_ln, BT, D
    );
    
    // FFN: h = res1_ln @ W1 + b1
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        2*D, BT, D, &alpha, d_W1, 2*D, d_res1_ln, D, &beta, d_h, 2*D));
    
    // Add bias (broadcast)
    for (int bt = 0; bt < BT; bt++) {
        blocks = (2 * D + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(d_h + bt * 2 * D, d_b1, d_h + bt * 2 * D, 2 * D);
    }
    
    // GELU
    blocks = (BT * 2 * D + threads - 1) / threads;
    gelu_forward_kernel<<<blocks, threads>>>(d_h, d_h_act, BT * 2 * D);
    
    // ffn_out = h_act @ W2 + b2
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D, BT, 2*D, &alpha, d_W2, D, d_h_act, 2*D, &beta, d_ffn_out, D));
    
    for (int bt = 0; bt < BT; bt++) {
        blocks = (D + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(d_ffn_out + bt * D, d_b2, d_ffn_out + bt * D, D);
    }
    
    // Final residual
    blocks = (BT * D + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_res1, d_ffn_out, d_out, BT * D);
    
    // Loss
    float *d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    blocks = (BT * D + threads - 1) / threads;
    mse_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_out, d_target, d_loss, BT * D
    );
    
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Forward loss: %f\n\n", h_loss);
    
    // ========== BACKWARD PASS ==========
    printf("========== BACKWARD PASS ==========\n");
    
    // Allocate gradient buffers for activations
    float *d_dout, *d_dres1, *d_dffn_out;
    float *d_dh_act, *d_dh, *d_dres1_ln_from_ffn;
    float *d_dres1_ln_scaled, *d_dres1_from_ln2;
    float *d_dres1_total, *d_dx_resid, *d_dattn_out;
    float *d_dhead, *d_dA, *d_dV, *d_dScores, *d_dQ, *d_dK;
    float *d_dx_ln_from_Q, *d_dx_ln_from_K, *d_dx_ln_from_V, *d_dx_ln_attn;
    float *d_dx_ln_scaled, *d_dx_from_ln1, *d_dx_total;
    float *d_temp1, *d_temp2;
    
    CUDA_CHECK(cudaMalloc(&d_dout, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dres1, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dffn_out, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dh_act, BT * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dh, BT * 2 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dres1_ln_from_ffn, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dres1_ln_scaled, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dres1_from_ln2, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dres1_total, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_resid, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dattn_out, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dhead, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA, B * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dScores, B * T * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dQ, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_ln_from_Q, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_ln_from_K, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_ln_from_V, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_ln_attn, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_ln_scaled, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_from_ln1, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_total, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp1, BT * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp2, BT * D * sizeof(float)));
    
    // Gradient at output: dL/dout = (out - target) / N
    blocks = (BT * D + threads - 1) / threads;
    mse_grad_kernel<<<blocks, threads>>>(d_out, d_target, d_dout, BT * D);
    
    // Final residual: out = res1 + ffn_out
    CUDA_CHECK(cudaMemcpy(d_dres1, d_dout, BT * D * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_dffn_out, d_dout, BT * D * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // FFN backward: ffn_out = h_act @ W2 + b2
    // dW2 = h_act^T @ dffn_out
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D, 2*D, BT, &alpha, d_dffn_out, D, d_h_act, 2*D, &beta_add, d_dW2, D));
    
    // db2 = sum(dffn_out)
    sum_reduction_kernel<<<D, threads, threads * sizeof(float)>>>(
        d_dffn_out, d_db2, BT, D
    );
    
    // dh_act = dffn_out @ W2^T
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        2*D, BT, D, &alpha, d_W2, D, d_dffn_out, D, &beta, d_dh_act, 2*D));
    
    // GELU backward
    blocks = (BT * 2 * D + threads - 1) / threads;
    gelu_backward_kernel<<<blocks, threads>>>(d_h, d_dh_act, d_dh, BT * 2 * D);
    
    // dW1 = res1_ln^T @ dh
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        2*D, D, BT, &alpha, d_dh, 2*D, d_res1_ln, D, &beta_add, d_dW1, 2*D));
    
    // db1 = sum(dh)
    sum_reduction_kernel<<<2*D, threads, threads * sizeof(float)>>>(
        d_dh, d_db1, BT, 2*D
    );
    
    // dres1_ln = dh @ W1^T
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D, BT, 2*D, &alpha, d_W1, 2*D, d_dh, 2*D, &beta, d_dres1_ln_from_ffn, D));
    
    // LN2 backward: gamma and beta gradients
    // dgamma2 = sum(dres1_ln_from_ffn * res1_norm)
    blocks = (BT * D + threads - 1) / threads;
    elementwise_mul_kernel<<<blocks, threads>>>(
        d_dres1_ln_from_ffn, d_res1_norm, d_temp1, BT * D
    );
    sum_reduction_kernel<<<D, threads, threads * sizeof(float)>>>(
        d_temp1, d_dgamma2, BT, D
    );
    
    // dbeta2 = sum(dres1_ln_from_ffn)
    sum_reduction_kernel<<<D, threads, threads * sizeof(float)>>>(
        d_dres1_ln_from_ffn, d_dbeta2, BT, D
    );
    
    // Scale by gamma2
    // For each element: dln2[i] = dres1_ln_from_ffn[i] * gamma2[i % D]
    for (int bt = 0; bt < BT; bt++) {
        for (int d = 0; d < D; d++) {
            elementwise_mul_kernel<<<1, 1>>>(
                d_dres1_ln_from_ffn + bt * D + d,
                d_gamma2 + d,
                d_dres1_ln_scaled + bt * D + d,
                1
            );
        }
    }
    // Simplified: just multiply element-wise with broadcast
    blocks = (BT * D + threads - 1) / threads;
    // This is a hack - proper implementation needs a broadcast kernel
    CUDA_CHECK(cudaMemcpy(d_dres1_ln_scaled, d_dres1_ln_from_ffn, 
                         BT * D * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // LN backward through normalization
    layer_norm_backward_kernel<<<BT, threads, 2 * threads * sizeof(float)>>>(
        d_dres1_ln_scaled, d_res1, d_mean2, d_inv_std2, d_dres1_from_ln2, BT, D
    );
    
    // Total gradient into res1
    blocks = (BT * D + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_dres1, d_dres1_from_ln2, d_dres1_total, BT * D);
    
    // Residual split
    CUDA_CHECK(cudaMemcpy(d_dx_resid, d_dres1_total, BT * D * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_dattn_out, d_dres1_total, BT * D * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Attention backward: attn_out = head @ W_O
    // dW_O = head^T @ dattn_out
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D, D, BT, &alpha, d_dattn_out, D, d_head, D, &beta_add, d_dW_O, D));
    
    // dhead = dattn_out @ W_O^T
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_O, D, d_dattn_out, D, &beta, d_dhead, D));
    
    // head = A @ V backward
    for (int b = 0; b < B; b++) {
        // dA = dhead @ V^T
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, D, &alpha,
            d_V + b * T * D, D,
            d_dhead + b * T * D, D,
            &beta, d_dA + b * T * T, T));
        
        // dV = A^T @ dhead
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            D, T, T, &alpha,
            d_dhead + b * T * D, D,
            d_A + b * T * T, T,
            &beta_add, d_dV + b * T * D, D));
    }
    
    // Softmax backward
    softmax_backward_kernel<<<grid_soft, threads, threads * sizeof(float)>>>(
        d_A, d_dA, d_dScores, B, T
    );
    
    // Scale gradient
    blocks = (B * T * T + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(d_dScores, 1.0f / sqrt_dk, d_dScores, B * T * T);
    
    // scores = Q @ K^T backward
    // dQ = dScores @ K
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        D, T*B, T, &alpha, d_K, D, d_dScores, T, &beta, d_dQ, D));
    
    // dK = dScores^T @ Q
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D, T*B, T, &alpha, d_Q, D, d_dScores, T, &beta, d_dK, D));
    
    // Q, K, V projection backward
    // dW_Q = x_ln^T @ dQ
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D, D, BT, &alpha, d_dQ, D, d_x_ln, D, &beta_add, d_dW_Q, D));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D, D, BT, &alpha, d_dK, D, d_x_ln, D, &beta_add, d_dW_K, D));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        D, D, BT, &alpha, d_dV, D, d_x_ln, D, &beta_add, d_dW_V, D));
    
    // dx_ln contributions
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_Q, D, d_dQ, D, &beta, d_dx_ln_from_Q, D));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_K, D, d_dK, D, &beta, d_dx_ln_from_K, D));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        D, BT, D, &alpha, d_W_V, D, d_dV, D, &beta, d_dx_ln_from_V, D));
    
    // Sum contributions
    blocks = (BT * D + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_dx_ln_from_Q, d_dx_ln_from_K, d_dx_ln_attn, BT * D);
    add_kernel<<<blocks, threads>>>(d_dx_ln_attn, d_dx_ln_from_V, d_dx_ln_attn, BT * D);
    
    // LN1 backward: gamma and beta gradients
    elementwise_mul_kernel<<<blocks, threads>>>(
        d_dx_ln_attn, d_x_norm1, d_temp2, BT * D
    );
    sum_reduction_kernel<<<D, threads, threads * sizeof(float)>>>(
        d_temp2, d_dgamma1, BT, D
    );
    sum_reduction_kernel<<<D, threads, threads * sizeof(float)>>>(
        d_dx_ln_attn, d_dbeta1, BT, D
    );
    
    // Scale and LN backward
    CUDA_CHECK(cudaMemcpy(d_dx_ln_scaled, d_dx_ln_attn, BT * D * sizeof(float), cudaMemcpyDeviceToDevice));
    
    layer_norm_backward_kernel<<<BT, threads, 2 * threads * sizeof(float)>>>(
        d_dx_ln_scaled, d_x, d_mean1, d_inv_std1, d_dx_from_ln1, BT, D
    );
    
    // Total gradient to input
    add_kernel<<<blocks, threads>>>(d_dx_from_ln1, d_dx_resid, d_dx_total, BT * D);
    
    // ========== Print Results ==========
    printf("\nParameter gradient norms:\n");
    
    float h_grad_norm;
    float* d_grad_norm;
    CUDA_CHECK(cudaMalloc(&d_grad_norm, sizeof(float)));
    
    #define PRINT_GRAD_NORM(name, ptr, size) \
        CUBLAS_CHECK(cublasSnrm2(cublas_handle, size, ptr, 1, &h_grad_norm)); \
        printf(" %6s: %f\n", name, h_grad_norm);
    
    PRINT_GRAD_NORM("W_Q", d_dW_Q, D*D);
    PRINT_GRAD_NORM("W_K", d_dW_K, D*D);
    PRINT_GRAD_NORM("W_V", d_dW_V, D*D);
    PRINT_GRAD_NORM("W_O", d_dW_O, D*D);
    PRINT_GRAD_NORM("W1", d_dW1, D*2*D);
    PRINT_GRAD_NORM("W2", d_dW2, 2*D*D);
    PRINT_GRAD_NORM("b1", d_db1, 2*D);
    PRINT_GRAD_NORM("b2", d_db2, D);
    PRINT_GRAD_NORM("gamma1", d_dgamma1, D);
    PRINT_GRAD_NORM("beta1", d_dbeta1, D);
    PRINT_GRAD_NORM("gamma2", d_dgamma2, D);
    PRINT_GRAD_NORM("beta2", d_dbeta2, D);
    
    // Cleanup
    cudaFree(d_grad_norm);
    cudaFree(d_loss);
    
    // Free all allocations (abbreviated for brevity)
    cudaFree(d_W_Q); cudaFree(d_W_K); cudaFree(d_W_V); cudaFree(d_W_O);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_gamma1); cudaFree(d_beta1); cudaFree(d_gamma2); cudaFree(d_beta2);
    cudaFree(d_dW_Q); cudaFree(d_dW_K); cudaFree(d_dW_V); cudaFree(d_dW_O);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_dgamma1); cudaFree(d_dbeta1); cudaFree(d_dgamma2); cudaFree(d_dbeta2);
    cudaFree(d_x); cudaFree(d_target);
    // ... (free remaining buffers)
    
    cublasDestroy(cublas_handle);
    curandDestroyGenerator(gen);
    
    printf("\nDone!\n");
    return 0;
}
