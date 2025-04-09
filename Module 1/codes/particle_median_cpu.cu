#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cstdio>

struct Particle {
    float x, y;    // position
    float vx, vy;  // velocity
};

// Calculate magnitude of velocity vector
float velocity_magnitude(float vx, float vy) {
    return sqrt(vx*vx + vy*vy);
}

// CPU implementation of median calculation
float median_velocity(thrust::universal_vector<Particle> particles) 
{
    // Extract velocity magnitudes
    thrust::universal_vector<float> velocities(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
        velocities[i] = velocity_magnitude(particles[i].vx, particles[i].vy);
    }

    // Use standard C++ algorithm to sort on CPU
    std::sort(velocities.begin(), velocities.end());

    // Return the median value
    return velocities[velocities.size() / 2];
}

int main() 
{
    // Simulation parameters
    float dt = 0.1f;  // time step
    float damping = 0.98f;  // velocity damping factor

    // Initial particle states
    thrust::universal_vector<Particle> particles{
        {0.0f, 0.0f, 1.0f, 0.5f},    // Particle 1
        {1.0f, 2.0f, -0.5f, 0.2f},   // Particle 2
        {-1.0f, -1.0f, 0.3f, 0.7f},  // Particle 3
        {2.0f, -2.0f, -0.1f, -0.8f}, // Particle 4
        {3.0f, 1.0f, -0.4f, 0.6f}    // Particle 5
    };

    // Update function (runs on GPU)
    auto update_particle = [=] __host__ __device__ (Particle p) { 
        // Update position based on velocity
        p.x += p.vx * dt;
        p.y += p.vy * dt;

        // Apply damping to velocities
        p.vx *= damping;
        p.vy *= damping;

        return p;
    };

    // Simulation loop
    std::printf("step  median_velocity\n");
    for (int step = 0; step < 3; step++) {
        // Update particles on GPU
        thrust::transform(thrust::device, 
                         particles.begin(), particles.end(), 
                         particles.begin(), 
                         update_particle);

        // Calculate median velocity on CPU
        float median_vel = median_velocity(particles);

        std::printf("%d     %.4f\n", step, median_vel);
    }
}
