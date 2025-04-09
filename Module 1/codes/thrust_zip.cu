#include <thrust/device_vector.h>  // Use device_vector instead of universal_vector for compatibility
#include <thrust/iterator/zip_iterator.h>  // Correct path to zip_iterator
#include <thrust/tuple.h>  // Explicitly include tuple
#include <thrust/execution_policy.h>
#include <cstdio>

struct Particle {
    float x, y;    // position
};

int main() {
    // Create two sets of particles
    thrust::device_vector<Particle> before{
        {0.0f, 0.0f},
        {1.0f, 2.0f},
        {-1.0f, -1.0f}
    };

    thrust::device_vector<Particle> after{
        {0.1f, 0.05f},
        {0.95f, 2.02f},
        {-0.97f, -0.93f}
    };

    // Create a zip iterator to combine both vectors
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(before.begin(), after.begin()));

    // Access the first particle pair
    auto pair = *zip;
    Particle p1 = thrust::get<0>(pair);  // Use copy instead of reference for device memory
    Particle p2 = thrust::get<1>(pair);  // Use copy instead of reference for device memory

    printf("First particle before: (%.2f, %.2f)\n", p1.x, p1.y);
    printf("First particle after: (%.2f, %.2f)\n", p2.x, p2.y);

    return 0;
}
