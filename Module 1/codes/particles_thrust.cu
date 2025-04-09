#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cstdio>

struct Particle {
    float x, y;    // position
    float vx, vy;  // velocity
};

// Define a functor for updating positions
struct UpdatePosition {
    const float dt;

    UpdatePosition(float _dt) : dt(_dt) {}

    __host__ __device__
    Particle operator()(const Particle& p) const {
        Particle updated = p;
        updated.x += p.vx * dt;
        updated.y += p.vy * dt;
        return updated;
    }
};

int main() {
    // Simulation parameters
    float dt = 0.1f;  // time step

    // Create particles on the host (CPU)
    thrust::host_vector<Particle> h_particles = {
        {0.0f, 0.0f, 1.0f, 0.5f},
        {1.0f, 2.0f, -0.5f, 0.2f},
        {-1.0f, -1.0f, 0.3f, 0.7f}
    };

    // Copy particles to the device (GPU)
    thrust::device_vector<Particle> d_particles = h_particles;

    // Create our transformation functor
    UpdatePosition updater(dt);

    // Print initial state
    printf("Step 0:\n");
    for (int i = 0; i < h_particles.size(); i++) {
        printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
                i, h_particles[i].x, h_particles[i].y, h_particles[i].vx, h_particles[i].vy);
    }

    // Run simulation for 3 steps
    for (int step = 1; step <= 3; step++) {
        // Update positions on the GPU
        thrust::transform(thrust::device, 
                         d_particles.begin(), d_particles.end(), 
                         d_particles.begin(), 
                         updater);

        // Copy results back to the host
        thrust::copy(d_particles.begin(), d_particles.end(), h_particles.begin());

        // Print results
        printf("\nStep %d:\n", step);
        for (int i = 0; i < h_particles.size(); i++) {
            printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
                   i, h_particles[i].x, h_particles[i].y, h_particles[i].vx, h_particles[i].vy);
        }
    }

    return 0;
}
