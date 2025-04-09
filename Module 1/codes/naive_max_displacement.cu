#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio>
#include <cmath>

struct Particle {
    float x, y;    // position
    float vx, vy;  // velocity
};

// Calculate distance between two positions
float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}

// Naive approach to find maximum displacement
float naive_max_displacement(const thrust::universal_vector<Particle>& before, 
                           const thrust::universal_vector<Particle>& after) 
{
    // Allocate vector to store displacements (temporary storage)
    thrust::universal_vector<float> displacements(before.size());

    // Compute displacements
    thrust::transform(thrust::device, 
                     before.begin(), before.end(),           // first input sequence  
                     after.begin(),                          // second input sequence
                     displacements.begin(),                  // output
                     [=] __host__ __device__ (const Particle& p1, const Particle& p2) {
                         // Calculate displacement between positions
                         return distance(p1.x, p1.y, p2.x, p2.y);
                     });

    // Find maximum displacement
    return thrust::reduce(thrust::device, 
                         displacements.begin(), 
                         displacements.end(), 
                         0.0f, thrust::maximum<float>{});
}

int main() 
{
    float dt = 0.1f;  // time step

    // Create two arrays to store particle states in alternating steps
    thrust::universal_vector<Particle> particles[2] = {
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
    auto update_particle = [=] __host__ __device__ (const Particle& p) {
        Particle updated = p;
        updated.x += p.vx * dt;
        updated.y += p.vy * dt;
        return updated;
    };

    std::printf("step  max_displacement\n");
    for (int step = 0; step < 3; step++) {
        // Get references to current and next state arrays
        thrust::universal_vector<Particle>& current = particles[step % 2];
        thrust::universal_vector<Particle>& next = particles[(step + 1) % 2];

        // Update particle positions
        thrust::transform(thrust::device, 
                         current.begin(), current.end(), 
                         next.begin(), 
                         update_particle);

        // Calculate maximum displacement
        float max_disp = naive_max_displacement(current, next);
        std::printf("%d     %.4f\n", step, max_disp);
    }

    return 0;
}
