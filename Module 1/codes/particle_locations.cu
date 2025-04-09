#include <thrust/execution_policy.h>
#include <cstdio>

// Helper function to print execution location, callable from both host and device
__host__ __device__ void execution_location(const char* location) {
#ifdef __CUDA_ARCH__
    printf("Currently executing on: GPU (%s)\n", location);
#else
    printf("Currently executing on: CPU (%s)\n", location);
#endif
}

int main() {

    execution_location("???");

    thrust::for_each_n(thrust::device,
                       thrust::counting_iterator<int>(0), 1,
                       [=] __host__ __device__ (int) {
                           execution_location("???");
                       });


    thrust::for_each_n(thrust::host,
                       thrust::counting_iterator<int>(0), 1,
                       [=] __host__ __device__ (int) {
                           execution_location("???");
                       });

    // Runs on CPU
    execution_location("???");

    return 0;
}
