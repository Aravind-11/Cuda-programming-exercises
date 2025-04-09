#include <thrust/device_vector.h>  // Use device_vector instead of universal_vector for better compatibility
#include <thrust/iterator/zip_iterator.h>  // Correct path for zip iterator
#include <thrust/iterator/transform_iterator.h>  // Correct path for transform iterator
#include <thrust/tuple.h>  // Explicitly include tuple header
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cmath>

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

    // Create a zip iterator
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(before.begin(), after.begin()));

    // Create a transform iterator that computes displacement
    auto displacement = thrust::make_transform_iterator(zip, 
        [] __host__ __device__ (const thrust::tuple<Particle, Particle>& t) {
            const Particle& p1 = thrust::get<0>(t);
            const Particle& p2 = thrust::get<1>(t);

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            return sqrt(dx*dx + dy*dy);
        });

    // Print the first few displacements
    printf("Particle 0 displacement: %.4f\n", displacement[0]);
    printf("Particle 1 displacement: %.4f\n", displacement[1]);
    printf("Particle 2 displacement: %.4f\n", displacement[2]);

    return 0;
}
