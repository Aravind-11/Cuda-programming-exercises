#include <cstdio>
#include <vector>
#include <cmath>

struct Particle {
    float x, y;    // position
    float vx, vy;  // velocity
};

int main() {
    // Simulation parameters
    float dt = 0.1f;  // time step

    // Create some particles with initial positions and velocities
    std::vector<Particle> particles = {
        {0.0f, 0.0f, 1.0f, 0.5f},
        {1.0f, 2.0f, -0.5f, 0.2f},
        {-1.0f, -1.0f, 0.3f, 0.7f}
    };

    // Print initial state
    printf("Step 0:\n");
    for (int i = 0; i < particles.size(); i++) {
        printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
                i, particles[i].x, particles[i].y, particles[i].vx, particles[i].vy);
    }

    // Run simulation for 3 steps
    for (int step = 1; step <= 3; step++) {
        // Update each particle position based on its velocity
        for (int i = 0; i < particles.size(); i++) {
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
        }

        // Print results
        printf("\nStep %d:\n", step);
        for (int i = 0; i < particles.size(); i++) {
            printf("Particle %d: pos=(%.2f, %.2f) vel=(%.2f, %.2f)\n", 
                   i, particles[i].x, particles[i].y, particles[i].vx, particles[i].vy);
        }
    }

    return 0;
}
