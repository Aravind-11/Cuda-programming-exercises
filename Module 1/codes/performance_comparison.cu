
#include <thrust/device_vector.h>  // Changed from universal_vector
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>  // Fixed path
#include <thrust/iterator/transform_iterator.h>  // Fixed path
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>  // Added missing header
#include <chrono>
#include <cstdio>
#include <cmath>

struct Particle {
    float x, y;    // position
};

// Naive approach with temporary storage
float naive_max_displacement(const thrust::device_vector<Particle>& before, 
                           const thrust::device_vector<Particle>& after) 
{
    // Allocate vector to store displacements
    thrust::device_vector<float> displacements(before.size());
    // Compute displacements
    thrust::transform(thrust::device, 
                     before.begin(), before.end(),
                     after.begin(),
                     displacements.begin(),
                     [] __host__ __device__ (const Particle& p1, const Particle& p2) {  // Fixed attributes
                         float dx = p2.x - p1.x;
                         float dy = p2.y - p1.y;
                         return sqrt(dx*dx + dy*dy);
                     });
    // Find maximum displacement
    return thrust::reduce(thrust::device, 
                         displacements.begin(), 
                         displacements.end(), 
                         0.0f, thrust::maximum<float>{});
}

// Optimized approach using iterators
float optimized_max_displacement(const thrust::device_vector<Particle>& before, 
                                const thrust::device_vector<Particle>& after) 
{
    // Create a zip iterator
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(before.begin(), after.begin()));

    // Create a transform iterator
    auto displacement = thrust::make_transform_iterator(zip, 
        [] __host__ __device__ (const thrust::tuple<Particle, Particle>& t) {  // Fixed attributes
            const Particle& p1 = thrust::get<0>(t);
            const Particle& p2 = thrust::get<1>(t);

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            return sqrt(dx*dx + dy*dy);
        });

    // Find maximum displacement
    return thrust::reduce(thrust::device, 
                         displacement, 
                         displacement + before.size(),
                         0.0f, thrust::maximum<float>{});
}

int main() 
{
    // Create large particle arrays (16 million particles)
    // Reduced size for testing - 16M particles may be too much for some systems
    const int NUM_PARTICLES = 1 << 20;  // Using 1M particles instead of 16M for initial testing

    thrust::device_vector<Particle> before(NUM_PARTICLES);
    thrust::device_vector<Particle> after(NUM_PARTICLES);

    // Initialize with some values
    // Note: This initialization is inefficient - should use thrust::transform for GPU
    thrust::device_vector<int> indices(NUM_PARTICLES);
    thrust::sequence(indices.begin(), indices.end());

    thrust::transform(indices.begin(), indices.end(), before.begin(),
        [] __host__ __device__ (int i) {
            Particle p;
            p.x = i;
            p.y = NUM_PARTICLES - i;
            return p;
        });

    thrust::transform(indices.begin(), indices.end(), after.begin(),
        [] __host__ __device__ (int i) {
            Particle p;
            p.x = i + 0.5f;
            p.y = NUM_PARTICLES - i - 0.2f;
            return p;
        });

    // Measure naive approach time (include allocation)
    auto start_naive = std::chrono::high_resolution_clock::now();
    float naive_result = naive_max_displacement(before, after);
    auto end_naive = std::chrono::high_resolution_clock::now();

    // Measure optimized approach time
    auto start_optimized = std::chrono::high_resolution_clock::now();
    float optimized_result = optimized_max_displacement(before, after);
    auto end_optimized = std::chrono::high_resolution_clock::now();

    // Calculate durations in milliseconds
    float naive_time = std::chrono::duration<float, std::milli>(end_naive - start_naive).count();
    float optimized_time = std::chrono::duration<float, std::milli>(end_optimized - start_optimized).count();

    // Print results
    printf("Naive approach: %.3f ms, result = %.6f\n", naive_time, naive_result);
    printf("Optimized approach: %.3f ms, result = %.6f\n", optimized_time, optimized_result);
    printf("Speedup: %.2fx\n", naive_time / optimized_time);

    return 0;
}
