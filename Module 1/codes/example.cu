
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <cstdio>
#include <cmath>

namespace {

// Simple particle physics computation on GPU
__global__ void compute_forces_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Access particle data
    float x = data[idx * 4 + 0];
    float y = data[idx * 4 + 1];

    // Simple force calculation (distance from origin)
    float r = sqrtf(x*x + y*y);
    float force = 1.0f / (r + 0.1f);  // Avoid division by zero

    // Store forces
    data[idx * 4 + 2] = -force * x/r;  // fx
    data[idx * 4 + 3] = -force * y/r;  // fy
}

// Helper function to run kernel
template<typename VectorType>
void compute_forces(VectorType& data) {
    int n = data.size() / 4;
    float* raw_ptr = thrust::raw_pointer_cast(data.data());

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    compute_forces_kernel<<<num_blocks, block_size>>>(raw_ptr, n);
    cudaDeviceSynchronize();
}

// Simple benchmark comparing memory transfer approaches
void measure_transfer_overhead() {
    const int N = 1000000;  // 1 million particles

    // Allocate data
    thrust::host_vector<float> h_data(N * 4);
    thrust::device_vector<float> d_data = h_data;

    // Fill with some data
    for(int i = 0; i < N * 4; i++) {
        h_data[i] = static_cast<float>(i);
    }

    // Test 1: Compute only
    auto t1 = std::chrono::high_resolution_clock::now();
    compute_forces(d_data);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Test 2: Compute + transfers 
    auto t3 = std::chrono::high_resolution_clock::now();
    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
    compute_forces(d_data);
    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
    auto t4 = std::chrono::high_resolution_clock::now();

    double compute_time = std::chrono::duration<double>(t2-t1).count();
    double total_time = std::chrono::duration<double>(t4-t3).count();

    printf("\nMemory Transfer Analysis:\n");
    printf("Compute only: %g seconds\n", compute_time);
    printf("With transfers: %g seconds\n", total_time);
    printf("Transfer overhead: %g seconds\n", total_time - compute_time);
}

// Demonstrate universal vector vs explicit memory management
void compare_memory_approaches() {
    const int N = 100000;  // 100k particles
    printf("\nComparing memory management approaches with %d particles...\n", N);

    // Test 1: Universal vector (automatic transfers)
    {
        thrust::universal_vector<float> data(N * 4, 1.0f);

        auto t1 = std::chrono::high_resolution_clock::now();

        // Do some work with automatic memory management
        for(int i = 0; i < 10; i++) {
            compute_forces(data);
            // Simulate CPU work every few iterations
            if(i % 3 == 0) {
                float sum = thrust::reduce(thrust::host, data.begin(), data.end());
                if(sum < -1e10) printf("Unlikely sum: %f\n", sum);
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(t2-t1).count();
        printf("Universal vector time: %g seconds\n", time);
    }

    // Test 2: Explicit memory management
    {
        thrust::host_vector<float> h_data(N * 4, 1.0f);
        thrust::device_vector<float> d_data = h_data;

        auto t1 = std::chrono::high_resolution_clock::now();

        // Do same work with explicit memory management
        for(int i = 0; i < 10; i++) {
            compute_forces(d_data);
            // Only transfer when needed
            if(i % 3 == 0) {
                thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
                float sum = thrust::reduce(thrust::host, h_data.begin(), h_data.end());
                if(sum < -1e10) printf("Unlikely sum: %f\n", sum);
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(t2-t1).count();
        printf("Explicit memory time: %g seconds\n", time);
    }
}

} // anonymous namespace

int main() {
    // Show overhead of memory transfers
    measure_transfer_overhead();

    // Compare different memory management approaches
    compare_memory_approaches();

    return 0;
}
