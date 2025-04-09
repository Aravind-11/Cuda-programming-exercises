#include <thrust/universal_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <cstdio>

// This is a helper function that will print where we are executing
void where_am_i(const char* location) {
    printf("Executing on: %s\n", location);
}

int main() {
    // Where does this line execute?
    where_am_i("???");

    // Create a vector with a single element
    thrust::universal_vector<int> vec{1};

    // Run a function on the GPU
    thrust::for_each(thrust::device, vec.begin(), vec.end(),
                   [=] __host__ __device__(int) { where_am_i("???"); });

    // Run a function on the CPU
    thrust::for_each(thrust::host, vec.begin(), vec.end(),
                   [=] __host__ __device__(int) { where_am_i("???"); });

    // Where does this line execute?
    where_am_i("???");

    return 0;
}
