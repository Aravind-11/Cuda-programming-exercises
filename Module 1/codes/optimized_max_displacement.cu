
#include <thrust/device_vector.h>  // Changed from universal_vector
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>  // Fixed path
#include <thrust/iterator/transform_iterator.h>  // Fixed path
#include <thrust/reduce.h>
#include <thrust/tuple.h>  // Added missing header
#include <cstdio>
#include <cmath>

struct Particle {
    float x, y;    // position
    float vx, vy;  // velocity
};

// Optimized approach using iterators
float optimized_max_displacement(const thrust::device_vector<Particle>& before, 
                                const thrust::device_vector<Particle>& after) 
{
    // Create a zip iterator to combine both vectors
    auto zip = thrust::make_zip_iterator(thrust::make_tuple(before.begin(), after.begin()));

    // Create a transform iterator that computes displacement
    auto displacement = thrust::make_transform_iterator(zip, 
        [] __host__ __device__ (const thrust::tuple<Particle, Particle>& t) {  // Fixed attributes
            const Particle& p1 = thrust::get<0>(t);
            const Particle& p2 = thrust::get<1>(t);

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;
            return sqrt(dx*dx + dy*dy);
        });

    // Find maximum displacement directly from the transform iterator
    return thrust::reduce(thrust::device, 
                         displacement, 
                         displacement + before.size(),
                         0.0f, thrust::maximum<float>{});
}

int main() 
{
    float dt = 0.1f;  // time step

    // Create two arrays to store particle states in alternating steps
    thrust::device_vector<Particle> particles[2] = {  // Changed from universal_vector
        // Initial state
        {
            {0.0f, 0.0f, 1.0f, 0.5f},    // Particle 1
            {1.0f, 2.0f, -0.5f, 0.2f},   // Particle 2
            {-1.0f, -1.0f, 0.3f, 0.7f}   // Particle 3
        },
        // Empty state for next timestep
        {}
    };
    particles[1].resize(particles[0].size());

    // Update function - applies velocity to position
    auto update_particle = [=] __host__ __device__ (const Particle& p) {  // Fixed attributes
        Particle updated = p;
        updated.x += p.vx * dt;
        updated.y += p.vy * dt;
        return updated;
    };

    std::printf("step  max_displacement\n");
    for (int step = 0; step < 3; step++) {
        // Get references to current and next state arrays
        thrust::device_vector<Particle>& current = particles[step % 2];
        thrust::device_vector<Particle>& next = particles[(step + 1) % 2];

        // Update particle positions
        thrust::transform(thrust::device, 
                         current.begin(), current.end(), 
                         next.begin(), 
                         update_particle);

        // Calculate maximum displacement
        float max_disp = optimized_max_displacement(current, next);
        std::printf("%d     %.4f\n", step, max_disp);
    }

return 0;
}
