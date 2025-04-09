#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda/std/mdspan>
#include <cstdio>

struct ParticleData {
    float x, y;
    float vx, vy;
};

int main() {
    float dt = 0.1f;
    const int NUM_PARTICLES = 3;

    // Store particle data in a single array
    thrust::universal_vector<float> particle_data{
        // x    y     vx    vy
        0.0f,  0.0f,  1.0f, 0.5f,   // Particle 0
        1.0f,  2.0f, -0.5f, 0.2f,   // Particle 1
        -1.0f, -1.0f, 0.3f, 0.7f    // Particle 2
    };

    // Create mdspan view of particle data
    float* data_ptr = thrust::raw_pointer_cast(particle_data.data());
    cuda::std::mdspan particles(data_ptr, NUM_PARTICLES, 4);  // 4 components per particle

    // Print initial state
    printf("Step 0:\n");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
               i, particles(i,0), particles(i,1), particles(i,2), particles(i,3));
    }

    // Run simulation
    for (int step = 1; step <= 3; step++) {
        thrust::transform(
            thrust::device,
            particle_data.begin(), 
            particle_data.end() - 3,  // Don't transform last particle's velocity
            particle_data.begin(),
            [dt, particles] __device__ (float val) {
                int idx = &val - particles.data_handle();
                int particle_idx = idx / 4;
                int component = idx % 4;

                if (component < 2) {  // Position components
                    return val + dt * particles(particle_idx, component + 2);
                }
                return val;  // Velocity components unchanged
            }
        );

        printf("\nStep %d:\n", step);
        for (int i = 0; i < NUM_PARTICLES; i++) {
            printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
                   i, particles(i,0), particles(i,1), particles(i,2), particles(i,3));
        }
    }
}
